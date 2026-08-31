"""Structural tests for the network definitions: shapes only, random weights.

These do NOT load full_checkpoint.pt and do NOT validate prediction
correctness -- only that the architecture wires together with the
hyperparameters in app/config.py the same way it did in the original
training/inference code (input_dim=5120 for the sequence model derives
from IMG_HEIGHT=160 and the encoder's stride/hidden-dim configuration).
"""
import torch

from app.config import Config
from app.deep_learning.model import build_models


def test_encoder_seq_decoder_forward_shapes():
    vocab_size = 10
    encoder, seq_model, decoder = build_models(vocab_size, Config)

    encoder.eval(); seq_model.eval(); decoder.eval()

    dummy_image = torch.rand(1, 1, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    with torch.no_grad():
        features = encoder(dummy_image)
        # W dimension after strides [2, 2, 1]: IMG_WIDTH / 4
        assert features.shape[0] == 1
        assert features.shape[1] == Config.IMG_WIDTH // 4

        encoder_outputs = seq_model(features)
        assert encoder_outputs.shape == (1, Config.IMG_WIDTH // 4, 2 * Config.HIDDEN_DIM)

        hidden = (
            torch.zeros(1, 1, decoder.dec_hidden_dim),
            torch.zeros(1, 1, decoder.dec_hidden_dim),
        )
        input_token = torch.zeros(1, dtype=torch.long)
        logits, new_hidden, attn_weights = decoder(input_token, hidden, encoder_outputs)
        assert logits.shape == (1, vocab_size)
        assert attn_weights.shape == (1, Config.IMG_WIDTH // 4)
