"""End-to-end CPU inference smoke test using a real example image and the
trained checkpoint. Prints predictions; does not assert exact correctness
(no GPU-trained reference output is available in this environment), but
proves the full pipeline (preprocess -> encoder -> seq_model -> decoder ->
MathML) runs without error on real data.

Usage: python scripts/smoke_test_inference.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Config
from app.deep_learning.inference import EquationRecognizer
from app.deep_learning.postprocessing import latex_to_mathml
from app.deep_learning.preprocessing import preprocess_equation_image_for_inference


def main():
    examples_dir = Path(Config.EXAMPLES_DIR)
    image_path = examples_dir / "1.png"
    if not image_path.exists():
        print(f"FAIL: example image not found at {image_path}")
        sys.exit(1)

    print("Loading model (CPU) ...")
    t0 = time.time()
    recognizer = EquationRecognizer(Config)
    print(f"Model loaded in {time.time() - t0:.1f}s on device={recognizer.device}")

    image_bytes = image_path.read_bytes()
    img_tensor, _ = preprocess_equation_image_for_inference(image_bytes, Config.IMG_WIDTH, Config.IMG_HEIGHT)

    t0 = time.time()
    greedy = recognizer.decode_greedy(img_tensor)
    print(f"Greedy decode ({time.time() - t0:.1f}s): {greedy[:200]}")

    t0 = time.time()
    beam = recognizer.decode_beam(img_tensor)
    print(f"Beam decode ({time.time() - t0:.1f}s):   {beam[:200]}")

    print(f"Greedy MathML: {latex_to_mathml(greedy)[:200]}")
    print("OK: full CPU inference pipeline ran without error.")
    print("NOTE: prediction correctness / latency on GPU is NOT verified here.")


if __name__ == "__main__":
    main()
