"""HTTP layer only: request/response handling. No Deep Learning code here."""
import json
import os

from flask import Blueprint, Response, current_app, jsonify, render_template, request, send_from_directory

from app.deep_learning.preprocessing import PreprocessingError

bp = Blueprint("routes", __name__)

# How many example images to surface as one-click "try a sample" chips on
# the Recognize page, in examples.json's insertion order.
QUICK_START_SAMPLE_COUNT = 3


def _allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


def _service_or_error():
    service = current_app.extensions.get("inference_service")
    if service is None:
        return None, current_app.extensions.get("model_load_error") or "Model is not loaded."
    return service, None


def _load_examples_meta():
    """Returns (meta_dict, warning). meta_dict is {} if examples.json is missing."""
    meta_path = os.path.join(current_app.config["EXAMPLES_DIR"], "examples.json")
    if not os.path.exists(meta_path):
        return {}, "No example metadata found."
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f), None


@bp.route("/")
@bp.route("/recognize")
def recognize_page():
    _, load_error = _service_or_error()
    examples_meta, _warning = _load_examples_meta()
    samples = [
        {"filename": fname, "desc": meta.get("desc", fname)}
        for fname, meta in list(examples_meta.items())[:QUICK_START_SAMPLE_COUNT]
        if os.path.exists(os.path.join(current_app.config["EXAMPLES_DIR"], fname))
    ]
    return render_template("recognize.html", load_error=load_error, samples=samples)


@bp.route("/examples/<path:filename>")
def serve_example_image(filename):
    examples_meta, _warning = _load_examples_meta()
    if filename not in examples_meta:
        return jsonify({"error": "Not found."}), 404
    return send_from_directory(current_app.config["EXAMPLES_DIR"], filename)


@bp.route("/api/predict", methods=["POST"])
def predict():
    service, load_error = _service_or_error()
    if service is None:
        return jsonify({"error": load_error}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name must be 'image')."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Allowed: png, jpg, jpeg, bmp, tiff."}), 400

    image_bytes = file.read()
    try:
        result = service.predict_from_bytes(image_bytes)
    except PreprocessingError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        current_app.logger.exception("Inference failed")
        return jsonify({"error": "Inference failed due to an internal error."}), 500

    from app import encode_preview_png

    return jsonify({
        "preview_image": encode_preview_png(result.preview_image),
        "greedy_latex": result.greedy_latex,
        "greedy_mathml": result.greedy_mathml,
        "beam_latex": result.beam_latex,
        "beam_mathml": result.beam_mathml,
    })


@bp.route("/examples")
def examples_page():
    service, load_error = _service_or_error()
    examples_dir = current_app.config["EXAMPLES_DIR"]
    examples_meta, warning = _load_examples_meta()

    if not examples_meta:
        return render_template("examples.html", examples=[], load_error=load_error, warning=warning)

    from app import encode_preview_png

    examples = []
    for fname, meta in examples_meta.items():
        img_path = os.path.join(examples_dir, fname)
        entry = {"filename": fname, "desc": meta.get("desc", fname), "ground_truth": meta.get("latex", "")}
        if not os.path.exists(img_path):
            entry["error"] = "Image file missing on disk."
            examples.append(entry)
            continue

        if service is None:
            entry["error"] = load_error
            examples.append(entry)
            continue

        with open(img_path, "rb") as fimg:
            image_bytes = fimg.read()
        try:
            result = service.predict_from_bytes(image_bytes)
            entry.update({
                "preview_image": encode_preview_png(result.preview_image),
                "greedy_latex": result.greedy_latex,
                "greedy_mathml": result.greedy_mathml,
                "beam_latex": result.beam_latex,
                "beam_mathml": result.beam_mathml,
            })
        except Exception as exc:
            entry["error"] = str(exc)
        examples.append(entry)

    return render_template("examples.html", examples=examples, load_error=load_error, warning=None)


@bp.route("/about")
def about_page():
    return render_template("about.html")


@bp.route("/robots.txt")
def robots_txt():
    return Response("User-agent: *\nAllow: /\n", mimetype="text/plain")


@bp.route("/healthz")
def healthz():
    service, load_error = _service_or_error()
    if service is None:
        return jsonify({"status": "unhealthy", "error": load_error}), 503
    return jsonify({"status": "ok", "device": str(service.recognizer.device)})
