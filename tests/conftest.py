import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_equation_png_bytes() -> bytes:
    """A tiny synthetic 'equation-like' image: black strokes on a white background."""
    arr = np.full((80, 300), 255, dtype=np.uint8)
    arr[30:50, 50:250] = 0  # a dark horizontal band standing in for glyphs
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def blank_png_bytes() -> bytes:
    arr = np.full((80, 300), 255, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
