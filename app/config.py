"""Runtime configuration, sourced from environment variables.

All values have sane local-dev defaults so the app runs out of the box,
but nothing machine-specific (absolute paths, credentials) is hardcoded.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    # --- Model artifacts ---
    MODEL_PATH = os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "full_checkpoint.pt"))
    VOCAB_PATH = os.environ.get("VOCAB_PATH", str(BASE_DIR / "models" / "vocab.json"))

    # --- Device ---
    # "auto" picks CUDA when available, otherwise CPU. Can be forced via env
    # (e.g. DEVICE=cpu to force CPU even on a GPU box).
    DEVICE = os.environ.get("DEVICE", "auto")

    # --- Model / inference hyperparameters (must match training-time values) ---
    IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT", 160))
    IMG_WIDTH = int(os.environ.get("IMG_WIDTH", 1024))
    ROW_BI_DIM = int(os.environ.get("ROW_BI_DIM", 64))
    HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 512))
    EMB_DIM = int(os.environ.get("EMB_DIM", 384))
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 2))
    DROPOUT = float(os.environ.get("DROPOUT", 0.3))
    BEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", 5))
    MAX_LEN = int(os.environ.get("MAX_LEN", 160))

    # --- Flask ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = _bool_env("DEBUG", False)
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8080))

    # --- Uploads ---
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 8)) * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "tif"}

    # --- Examples gallery ---
    EXAMPLES_DIR = os.environ.get("EXAMPLES_DIR", str(BASE_DIR / "examples"))
