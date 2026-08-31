"""Structural checkpoint sanity check: confirms full_checkpoint.pt loads and its
state_dict keys match the model classes in app/deep_learning/model.py.

This validates *loadability*, not prediction correctness. Run after cloning /
pulling the checkpoint via Git LFS, or after any change to app/deep_learning/model.py.

Usage: python scripts/verify_checkpoint.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from app.config import Config
from app.deep_learning.model import build_models


def main():
    import json

    with open(Config.VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)

    encoder, seq_model, decoder = build_models(len(vocab), Config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint on device={device} ...")
    ckpt = torch.load(Config.MODEL_PATH, map_location=device, weights_only=True)

    missing_keys = {"encoder", "seq_model", "decoder"} - set(ckpt.keys())
    if missing_keys:
        print(f"FAIL: checkpoint is missing top-level keys: {missing_keys}")
        sys.exit(1)

    encoder.load_state_dict(ckpt["encoder"])
    seq_model.load_state_dict(ckpt["seq_model"])
    decoder.load_state_dict(ckpt["decoder"])

    n_params = sum(p.numel() for p in encoder.parameters()) \
        + sum(p.numel() for p in seq_model.parameters()) \
        + sum(p.numel() for p in decoder.parameters())

    print("OK: checkpoint state_dicts loaded without shape/key mismatches.")
    print(f"Vocab size: {len(vocab)}")
    print(f"Total parameters: {n_params:,}")
    if device.type == "cpu":
        print("NOTE: this only validates CPU loadability, not GPU behavior "
              "or prediction correctness. Not verified: GPU inference.")


if __name__ == "__main__":
    main()
