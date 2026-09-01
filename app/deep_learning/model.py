"""Neural network architecture for equation recognition.

These classes are extracted verbatim from the original research/training
code so that `full_checkpoint.pt` (saved as encoder/seq_model/decoder
state_dicts) continues to load without modification. Do not change layer
shapes, names, or forward logic here without re-validating against the
checkpoint.

Pipeline: image -> LWDSCSA_Encoder -> SequenceModel -> LuongDecoder (per step)
"""
import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(x_cat)
        attn = self.sigmoid(attn)
        return x * attn


class LWDSCSA_Encoder(nn.Module):
    """Lightweight Depthwise-Separable-Conv + Spatial-Attention encoder,
    followed by a row-wise BiLSTM to produce a sequence of column features."""

    def __init__(self, in_channels=1, hidden_dims=(32, 64, 128), strides=(2, 2, 1), row_hidden_dim=64):
        super().__init__()
        assert len(hidden_dims) == len(strides)
        layers = []
        prev_dim = in_channels
        for hd, s in zip(hidden_dims, strides):
            layers.append(DepthwiseSeparableConv(prev_dim, hd, 3, stride=s, padding=1))
            prev_dim = hd
        self.dsc_blocks = nn.Sequential(*layers)
        self.spatial_att = SpatialAttention()
        self.row_hidden_dim = row_hidden_dim
        self.row_bilstm = nn.LSTM(
            input_size=hidden_dims[-1],
            hidden_size=row_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        feat = self.dsc_blocks(x)
        feat = self.spatial_att(feat)
        B, C, H, W = feat.shape
        feat_reshape = feat.permute(0, 2, 3, 1).contiguous().view(B * H, W, C)
        row_out, _ = self.row_bilstm(feat_reshape)
        row_out = row_out.reshape(B, H, W, 2 * self.row_hidden_dim)
        row_out = row_out.permute(0, 2, 1, 3).contiguous()
        row_out = row_out.reshape(B, W, H * 2 * self.row_hidden_dim)
        return row_out


class SequenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        output, _ = self.bilstm(x)
        return output


class LuongAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_hidden, encoder_outputs):
        attn_energies = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = torch.softmax(attn_energies, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hidden_dim, dec_hidden_dim, batch_first=True)
        self.attn = LuongAttention()
        self.out = nn.Linear(dec_hidden_dim, vocab_size)
        self.hidden_proj = nn.Linear(dec_hidden_dim, enc_hidden_dim)
        self.dec_hidden_dim = dec_hidden_dim

    def forward(self, input_token, prev_hidden, encoder_outputs):
        emb = self.embedding(input_token).unsqueeze(1)
        hidden_for_attn = prev_hidden[0].squeeze(0)
        hidden_for_attn = self.hidden_proj(hidden_for_attn)
        context, attn_weights = self.attn(hidden_for_attn, encoder_outputs)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([emb, context], dim=2)
        output, hidden = self.rnn(rnn_input, prev_hidden)
        output = output.squeeze(1)
        logits = self.out(output)
        return logits, hidden, attn_weights


def build_models(vocab_size, config):
    """Construct the three model components with the hyperparameters that
    match the trained checkpoint. Weights are NOT loaded here."""
    encoder = LWDSCSA_Encoder(
        in_channels=1,
        hidden_dims=[32, 64, 128],
        strides=[2, 2, 1],
        row_hidden_dim=config.ROW_BI_DIM,
    )
    seq_model = SequenceModel(
        input_dim=5120,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    decoder = LuongDecoder(
        vocab_size,
        emb_dim=config.EMB_DIM,
        enc_hidden_dim=2 * config.HIDDEN_DIM,
        dec_hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
    )
    return encoder, seq_model, decoder
