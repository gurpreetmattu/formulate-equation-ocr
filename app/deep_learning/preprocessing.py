"""Image preprocessing for equation-recognition inference.

Ported verbatim from the original app: crop to content bounding box,
sharpen/contrast-enhance, Otsu-binarize, resize to the model's fixed
input resolution, normalize to [0, 1]. Do not change this without
re-validating against the trained checkpoint, since the model was
trained on images produced by this exact pipeline.
"""
import io

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance


class PreprocessingError(ValueError):
    """Raised when an uploaded image cannot be turned into a model input."""


def preprocess_equation_image_for_inference(image_bytes: bytes, img_width: int, img_height: int):
    """Returns (model_input_tensor[1,1,H,W] float32 in [0,1], display_image uint8 ndarray)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        raise PreprocessingError(f"Could not read image: {exc}") from exc

    img_np = np.array(img)
    _, thresh = cv2.threshold(img_np, 240, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(255 - thresh)
    if coords is None:
        raise PreprocessingError("Image appears blank — no equation content detected.")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img_np[y:y + h, x:x + w]
    pil_img = Image.fromarray(cropped)
    sharpness = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    contrast = ImageEnhance.Contrast(sharpness).enhance(1.5)
    img = np.array(contrast)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    img_norm = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return img_tensor, img
