"""Model loading and decoding logic. Framework-agnostic (no Flask imports)."""
import json
import logging

import torch

from .model import build_models
from .postprocessing import tokens_to_latex

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the checkpoint or vocab cannot be loaded."""


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError(
            "DEVICE=cuda was requested but no CUDA-capable GPU is available on this machine."
        )
    return torch.device(requested)


class EquationRecognizer:
    """Holds the loaded encoder/seq_model/decoder in memory and exposes
    greedy and beam-search LaTeX decoding. Constructed once at application
    startup (see app/__init__.py) and reused across requests."""

    def __init__(self, config):
        self.config = config
        self.device = resolve_device(config.DEVICE)

        try:
            with open(config.VOCAB_PATH, encoding="utf-8") as f:
                self.vocab = json.load(f)
        except OSError as exc:
            raise ModelLoadError(f"Could not read vocab file at {config.VOCAB_PATH}: {exc}") from exc

        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        self.sos_token_id = self.vocab["<SOS>"]
        self.eos_token_id = self.vocab["<EOS>"]

        self.encoder, self.seq_model, self.decoder = build_models(len(self.vocab), config)
        self.encoder.to(self.device)
        self.seq_model.to(self.device)
        self.decoder.to(self.device)

        try:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=True)
        except FileNotFoundError as exc:
            raise ModelLoadError(
                f"Checkpoint not found at {config.MODEL_PATH}. "
                "See README 'Model Download/Setup' for how to obtain it."
            ) from exc

        self.encoder.load_state_dict(ckpt["encoder"])
        self.seq_model.load_state_dict(ckpt["seq_model"])
        self.decoder.load_state_dict(ckpt["decoder"])

        self.encoder.eval()
        self.seq_model.eval()
        self.decoder.eval()

        logger.info("Model loaded on device=%s vocab_size=%d", self.device, len(self.vocab))

    def _encode(self, image_tensor: torch.Tensor):
        features = self.encoder(image_tensor.to(self.device))
        return self.seq_model(features)

    def decode_greedy(self, image_tensor: torch.Tensor) -> str:
        with torch.no_grad():
            encoder_outputs = self._encode(image_tensor)
            hidden = (
                torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device),
                torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device),
            )
            input_token = torch.full((1,), self.sos_token_id, dtype=torch.long, device=self.device)
            pred_ids = []
            for _ in range(self.config.MAX_LEN):
                logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
                next_token = logits.argmax(dim=1)
                if next_token.item() == self.eos_token_id:
                    break
                pred_ids.append(next_token.item())
                input_token = next_token
        return tokens_to_latex(self.idx_to_token, pred_ids)

    def decode_beam(self, image_tensor: torch.Tensor, beam_width: int | None = None) -> str:
        beam_width = beam_width or self.config.BEAM_WIDTH
        with torch.no_grad():
            encoder_outputs = self._encode(image_tensor)
            h_0 = torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device)
            c_0 = torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device)
            beams = [([self.sos_token_id], (h_0, c_0), 0.0, False)]
            for _ in range(self.config.MAX_LEN):
                new_beams = []
                for tokens, h, score, finished in beams:
                    if finished:
                        new_beams.append((tokens, h, score, True))
                        continue
                    input_token = torch.tensor([tokens[-1]], dtype=torch.long, device=self.device)
                    logits, h_new, _ = self.decoder(input_token, h, encoder_outputs)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                    for log_p, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                        next_tokens = tokens + [idx]
                        next_score = score + log_p
                        next_finished = idx == self.eos_token_id
                        new_beams.append((next_tokens, h_new, next_score, next_finished))
                beams = sorted(new_beams, key=lambda b: b[2], reverse=True)[:beam_width]
                if all(b[3] for b in beams):
                    break
            best_tokens = beams[0][0][1:]
            if self.eos_token_id in best_tokens:
                best_tokens = best_tokens[: best_tokens.index(self.eos_token_id)]
        return tokens_to_latex(self.idx_to_token, best_tokens)
