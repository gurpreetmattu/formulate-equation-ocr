"""Flask application factory.

The Deep Learning model is loaded exactly once here, at process startup,
and stored on the app so every request reuses the same in-memory model
instead of reloading weights per request (expensive: GPU allocation,
disk I/O, deserialization).
"""
import base64
import io
import logging

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config
from app.deep_learning.inference import EquationRecognizer, ModelLoadError
from app.services.inference_service import InferenceService

logger = logging.getLogger(__name__)

# Content-Security-Policy scoped to what this app actually loads: itself,
# MathJax from cdnjs, and the Inter font from Google Fonts. No wildcards.
_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'"
)


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

    # The app is deployed behind a reverse proxy (Docker/Cloud Run, see
    # README) -- without this, request.is_secure and generated URLs would
    # reflect the proxy hop instead of the original client request.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    logging.basicConfig(level=logging.INFO)

    @app.after_request
    def _set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = _CSP
        return response

    def _wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(404)
    def _not_found(_exc):
        if _wants_json():
            return jsonify({"error": "Not found."}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        max_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        message = f"File is too large. Maximum upload size is {max_mb} MB."
        if _wants_json():
            return jsonify({"error": message}), 413
        return render_template("errors/generic.html", code=413, message=message), 413

    @app.errorhandler(500)
    @app.errorhandler(Exception)
    def _server_error(exc):
        if isinstance(exc, HTTPException):
            # Let other registered HTTPException handlers (404, 413) and
            # well-formed 4xx responses pass through unchanged.
            return exc
        logger.exception("Unhandled server error")
        if _wants_json():
            return jsonify({"error": "Internal server error."}), 500
        return render_template("errors/generic.html", code=500,
                                message="Something went wrong on our end."), 500

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
