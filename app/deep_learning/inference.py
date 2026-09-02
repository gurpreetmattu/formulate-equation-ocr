"""Model loading and decoding logic. Framework-agnostic (no Flask imports)."""
import json
import logging
import pickle

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
        except pickle.UnpicklingError:
            # weights_only=True's safe unpickler allowlist isn't guaranteed identical
            # across torch builds/platforms (e.g. the CUDA wheel used in the deployed
            # container vs. a CPU wheel used locally). The checkpoint is our own
            # trusted artifact (tracked in this repo), so fall back to the full loader
            # rather than hard-crashing the app on environments where the safe path
            # rejects it.
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device, weights_only=False)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.seq_model.load_state_dict(ckpt["seq_model"])
        self.decoder.load_state_dict(ckpt["decoder"])

        self.encoder.eval()
        self.seq_model.eval()
        self.decoder.eval()

        self._validate_input_shape()

        logger.info("Model loaded on device=%s vocab_size=%d", self.device, len(self.vocab))

    def _validate_input_shape(self):
        """Fails loudly at startup if config.IMG_HEIGHT/IMG_WIDTH don't
        match the shapes this checkpoint was actually trained with (the
        encoder's feature width is hardcoded into seq_model's input_dim in
        model.py -- see build_models), instead of surfacing a confusing
        tensor-shape error on a user's first real request."""
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, self.config.IMG_HEIGHT, self.config.IMG_WIDTH, device=self.device)
                self._encode(dummy)
        except Exception as exc:
            raise ModelLoadError(
                f"Configured IMG_HEIGHT={self.config.IMG_HEIGHT}/IMG_WIDTH={self.config.IMG_WIDTH} "
                f"do not match the shapes this checkpoint was trained with: {exc}"
            ) from exc

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
        return self.decode_beam_candidates(image_tensor, beam_width)[0]

    def decode_beam_candidates(self, image_tensor: torch.Tensor, beam_width: int | None = None) -> list[str]:
        """Returns every finished beam's LaTeX, best (highest length-normalized
        score) first. The top beam is usually the most accurate, but it can
        occasionally be syntactically invalid LaTeX (e.g. unbalanced \\left/
        \\right) -- callers that need convertible output should try each
        candidate in order rather than assuming candidates[0] always parses."""
        beam_width = beam_width or self.config.BEAM_WIDTH
        with torch.no_grad():
            encoder_outputs = self._encode(image_tensor)
            h_0 = torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device)
            c_0 = torch.zeros(1, 1, self.decoder.dec_hidden_dim, device=self.device)
            beams = [([self.sos_token_id], (h_0, c_0), 0.0, False)]
            for _ in range(self.config.MAX_LEN):
                active = [b for b in beams if not b[3]]
                if not active:
                    break
                finished = [b for b in beams if b[3]]

                # Run every still-active beam through the decoder in one
                # batched call instead of one Python-loop call per beam --
                # the model already supports arbitrary batch size (see
                # LuongDecoder.forward), decode_beam just wasn't using it.
                input_tokens = torch.tensor([b[0][-1] for b in active], dtype=torch.long, device=self.device)
                h_batch = torch.cat([b[1][0] for b in active], dim=1)
                c_batch = torch.cat([b[1][1] for b in active], dim=1)
                enc_batch = encoder_outputs.expand(len(active), -1, -1)

                logits, (h_new, c_new), _ = self.decoder(input_tokens, (h_batch, c_batch), enc_batch)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)

                new_beams = list(finished)
                for i, (tokens, _prev_h, score, _finished) in enumerate(active):
                    beam_hidden = (h_new[:, i:i + 1, :], c_new[:, i:i + 1, :])
                    for log_p, idx in zip(topk_log_probs[i].tolist(), topk_indices[i].tolist()):
                        next_tokens = tokens + [idx]
                        next_score = score + log_p
                        next_finished = idx == self.eos_token_id
                        new_beams.append((next_tokens, beam_hidden, next_score, next_finished))
                # Rank by length-normalized score, not the raw cumulative
                # log-prob sum. Summing log-probs without normalizing
                # systematically favors shorter sequences (every extra
                # token can only make the sum more negative), which is
                # exactly why beam search under-performed greedy decoding
                # in eval -- this ranks candidates fairly regardless of how
                # many tokens they've generated so far.
                beams = sorted(new_beams, key=lambda b: b[2] / len(b[0]), reverse=True)[:beam_width]
                if all(b[3] for b in beams):
                    break
            candidates = []
            for tokens, _hidden, _score, _finished in beams:
                tokens = tokens[1:]
                if self.eos_token_id in tokens:
                    tokens = tokens[: tokens.index(self.eos_token_id)]
                candidates.append(tokens_to_latex(self.idx_to_token, tokens))
        return candidates
