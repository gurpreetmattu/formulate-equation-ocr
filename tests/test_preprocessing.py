import pytest
import torch

from app.deep_learning.preprocessing import PreprocessingError, preprocess_equation_image_for_inference

IMG_W, IMG_H = 1024, 160


def test_output_tensor_shape_and_dtype(sample_equation_png_bytes):
    tensor, display_img = preprocess_equation_image_for_inference(sample_equation_png_bytes, IMG_W, IMG_H)
    assert tensor.shape == (1, 1, IMG_H, IMG_W)
    assert tensor.dtype == torch.float32
    assert display_img.shape == (IMG_H, IMG_W)


def test_output_normalized_to_unit_range(sample_equation_png_bytes):
    tensor, _ = preprocess_equation_image_for_inference(sample_equation_png_bytes, IMG_W, IMG_H)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_blank_image_raises_preprocessing_error(blank_png_bytes):
    with pytest.raises(PreprocessingError):
        preprocess_equation_image_for_inference(blank_png_bytes, IMG_W, IMG_H)


def test_garbage_bytes_raise_preprocessing_error():
    with pytest.raises(PreprocessingError):
        preprocess_equation_image_for_inference(b"not an image", IMG_W, IMG_H)
