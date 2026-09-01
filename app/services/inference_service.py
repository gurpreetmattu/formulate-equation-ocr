"""Application-layer service sitting between Flask routes and the DL model.

Owns request-scoped concerns (validating uploads, calling preprocessing,
invoking the model, converting to MathML) so routes.py stays HTTP-only.
"""
import base64
import io
from dataclasses import dataclass

from PIL import Image

from app.deep_learning.inference import EquationRecognizer
from app.deep_learning.postprocessing import latex_to_mathml
from app.deep_learning.preprocessing import preprocess_equation_image_for_inference


def encode_preview_png(image_array) -> str:
    """Encodes a uint8 ndarray as a base64 PNG data URI for inline <img> use."""
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class PredictionResult:
    greedy_latex: str
    greedy_mathml: str
    beam_latex: str
    beam_mathml: str
    preview_image: "object"  # uint8 ndarray, for rendering the preprocessed input


class InferenceService:
    def __init__(self, recognizer: EquationRecognizer, config):
        self.recognizer = recognizer
        self.config = config

    def predict_from_bytes(self, image_bytes: bytes) -> PredictionResult:
        """Raises PreprocessingError on invalid/unreadable images."""
        img_tensor, preview_image = preprocess_equation_image_for_inference(
            image_bytes, self.config.IMG_WIDTH, self.config.IMG_HEIGHT
        )

        try:
            greedy_latex = self.recognizer.decode_greedy(img_tensor)
        except Exception as exc:
            greedy_latex = f"[Prediction error: {exc}]"
        greedy_mathml = latex_to_mathml(greedy_latex)

        try:
            beam_latex = self.recognizer.decode_beam(img_tensor, self.config.BEAM_WIDTH)
        except Exception as exc:
            beam_latex = f"[Prediction error: {exc}]"
        beam_mathml = latex_to_mathml(beam_latex)

        return PredictionResult(
            greedy_latex=greedy_latex,
            greedy_mathml=greedy_mathml,
            beam_latex=beam_latex,
            beam_mathml=beam_mathml,
            preview_image=preview_image,
        )
