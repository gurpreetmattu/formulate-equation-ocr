"""Flask application factory.

The Deep Learning model is loaded exactly once here, at process startup,
and stored on the app so every request reuses the same in-memory model
instead of reloading weights per request (expensive: GPU allocation,
disk I/O, deserialization).
"""
import base64
import io
import logging

from flask import Flask
from PIL import Image

from app.config import Config
from app.deep_learning.inference import EquationRecognizer, ModelLoadError
from app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)


def encode_preview_png(image_array) -> str:
    """Encodes a uint8 ndarray as a base64 PNG data URI for inline <img> use."""
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.jinja_env.filters["b64png"] = encode_preview_png

    logging.basicConfig(level=logging.INFO)

    try:
        recognizer = EquationRecognizer(config_class)
        app.extensions = getattr(app, "extensions", {})
        app.extensions["inference_service"] = InferenceService(recognizer, config_class)
        app.extensions["model_load_error"] = None
    except ModelLoadError as exc:
        # Fail loudly in logs but let the app boot so health checks / error
        # pages can explain the problem instead of the process crash-looping.
        logger.error("Model failed to load at startup: %s", exc)
        app.extensions = getattr(app, "extensions", {})
        app.extensions["inference_service"] = None
        app.extensions["model_load_error"] = str(exc)

    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
