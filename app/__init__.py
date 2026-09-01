"""Flask application factory.

The Deep Learning model is loaded exactly once here, at process startup,
and stored on the app so every request reuses the same in-memory model
instead of reloading weights per request (expensive: GPU allocation,
disk I/O, deserialization).
"""
import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import Config
from app.deep_learning.inference import EquationRecognizer, ModelLoadError
from app.extensions import limiter
from app.services.inference_service import InferenceService, encode_preview_png

logger = logging.getLogger(__name__)

# Content-Security-Policy scoped to what this app actually loads: itself,
# MathJax from cdnjs, and the Inter font from Google Fonts. No wildcards.
# img-src needs blob: in addition to 'self'/data: because the upload
# dropzone previews the selected file via URL.createObjectURL() (both a
# real <input type=file> selection and the quick-start sample chips,
# which fetch an image and hand it through the same preview path).
_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'"
)


def _load_examples_meta(examples_dir):
    """Returns (meta_dict, warning). meta_dict is {} if examples.json is missing."""
    meta_path = os.path.join(examples_dir, "examples.json")
    if not os.path.exists(meta_path):
        return {}, "No example metadata found."
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f), None


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.jinja_env.filters["b64png"] = encode_preview_png

    # The app is deployed behind a reverse proxy (Docker/Cloud Run, see
    # README) -- without this, request.is_secure and generated URLs would
    # reflect the proxy hop instead of the original client request.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    logging.basicConfig(level=logging.INFO)

    limiter.init_app(app)

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

    @app.errorhandler(429)
    def _rate_limited(_exc):
        message = "Too many requests. Please wait a moment and try again."
        if _wants_json():
            return jsonify({"error": message}), 429
        return render_template("errors/generic.html", code=429, message=message), 429

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

    app.extensions = getattr(app, "extensions", {})

    # Read once at startup instead of on every request -- the examples/
    # directory is a static, build-time asset (see Dockerfile), not
    # something that changes while the process is running.
    examples_meta, examples_warning = _load_examples_meta(config_class.EXAMPLES_DIR)
    app.extensions["examples_meta"] = examples_meta
    app.extensions["examples_warning"] = examples_warning

    try:
        recognizer = EquationRecognizer(config_class)
        app.extensions["inference_service"] = InferenceService(recognizer, config_class)
        app.extensions["model_load_error"] = None
    except ModelLoadError as exc:
        # Fail loudly in logs but let the app boot so health checks / error
        # pages can explain the problem instead of the process crash-looping.
        logger.error("Model failed to load at startup: %s", exc)
        app.extensions["inference_service"] = None
        app.extensions["model_load_error"] = str(exc)

    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
