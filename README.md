# Automated Printed Equation Recognition & LaTeX/MathML Conversion

A deep-learning application that converts images of **printed mathematical equations**
into **LaTeX** and **MathML**, served through a production Flask web application.

## Overview

Upload a scanned/printed equation image → a custom encoder-decoder network predicts
the LaTeX token sequence → `latex2mathml` converts it to MathML → both are shown
alongside a rendered preview of the equation.

## Architecture

```
Browser
   |
   v
Flask Web Layer (app/routes.py)
   |
   v
Inference Service (app/services/inference_service.py)
   |
   +--> Preprocessing (app/deep_learning/preprocessing.py)
   |
   v
Deep Learning Model (app/deep_learning/model.py, inference.py) -- PyTorch
   |
   v
models/full_checkpoint.pt
   |
   v
Postprocessing / MathML conversion (app/deep_learning/postprocessing.py)
   |
   v
JSON response -> rendered in browser (MathJax)
```

The model is constructed and its weights loaded **once**, at application startup
(`app/__init__.py`), and kept in memory for the life of the process — it is not
reloaded per request.

## Model

- **Encoder** — `LWDSCSA_Encoder`: a stack of depthwise-separable convolutions with a
  spatial-attention block, followed by a row-wise bidirectional LSTM that turns the 2D
  feature map into a sequence of column features.
- **Sequence model** — a 2-layer bidirectional LSTM (`SequenceModel`) refining the
  encoder's output sequence.
- **Decoder** — `LuongDecoder`: an LSTM decoder with Luong (multiplicative) attention
  over the encoder sequence, run step-by-step to emit LaTeX tokens.
- **Input**: a single-channel (grayscale) image, cropped to content, contrast/sharpness
  enhanced, Otsu-binarized, and resized to a fixed 160×1024 (H×W) tensor normalized to `[0, 1]`.
- **Output**: a sequence of LaTeX token IDs, decoded via either **greedy** decoding or
  **beam search** (both are computed and shown side by side), stopped at `<EOS>` or
  `MAX_LEN` tokens, then joined into a LaTeX string and converted to MathML.
- **Checkpoint** — `models/full_checkpoint.pt`: a `torch.save`d dict with three plain
  `state_dict`s (`encoder`, `seq_model`, `decoder`; no optimizer state, no custom
  pickled classes). It is loaded with `torch.load(..., weights_only=True)` for safer
  deserialization. The model classes in `app/deep_learning/model.py` must match this
  checkpoint's saved shapes exactly — do not change layer dimensions there without
  re-validating against the checkpoint.
- **Vocabulary** — `models/vocab.json`: a 544-token LaTeX-token → integer-ID mapping
  (includes `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`), trained on the im2latex-100k dataset.

Preserving this architecture and preprocessing exactly (rather than retraining or
redesigning) was the explicit goal of this refactor — see `app/deep_learning/`.

## Features

- End-to-end OCR: image → LaTeX & MathML
- Greedy and beam-search decoding shown side by side
- Rendered LaTeX preview (MathJax) alongside raw LaTeX/MathML source
- Example gallery comparing predictions against ground truth
- JSON API (`/api/predict`) usable independently of the web UI
- `/healthz` endpoint reporting whether the model loaded and which device it's on

## Technology Stack

- Python, PyTorch (model + inference)
- OpenCV / Pillow (image preprocessing)
- `latex2mathml` (LaTeX → MathML conversion)
- Flask + Gunicorn (web application / production server)
- Docker (containerization)
- Google Cloud Run with GPU (deployment target)

## Project Structure

```
app/
  config.py                  Environment-driven configuration
  routes.py                  Flask routes (HTTP only)
  __init__.py                App factory; loads the model once at startup
  services/
    inference_service.py     Glue between routes and the DL pipeline
  deep_learning/
    model.py                 Network architecture (unchanged from training code)
    preprocessing.py         Image -> tensor pipeline
    inference.py              Model loading + greedy/beam decoding
    postprocessing.py        Token joining + LaTeX -> MathML fixups
  templates/, static/        Web UI
models/                      full_checkpoint.pt, vocab.json (see below)
training/                    Original training/preprocessing notebooks + data-prep logs
legacy/                      Original Streamlit prototype, kept for reference only
examples/                    Sample images + ground truth used by the Recognize page's quick-start samples
tests/                       Pytest suite (see "Running Tests")
wsgi.py                      Production entrypoint (Gunicorn) / local dev runner
Dockerfile, .dockerignore    Container build
```

`dataset/` and `sample/` (raw training images/CSVs, several hundred MB) are **not**
committed to this repository — see `training/notebooks/` for how they were used, and
regenerate/obtain them separately if you need to retrain.

## Requirements

- Python 3.11 or 3.12 (3.14 is not yet broadly supported by the pinned PyTorch wheel)
- **GPU is optional for inference** — the app runs on CPU by default (`DEVICE=auto`
  picks CUDA if available, else CPU). Beam search on CPU is slower than on GPU but
  functionally identical.
- If using a GPU: an NVIDIA GPU with a CUDA 12.1-compatible driver, and the CUDA build
  of PyTorch (see Docker section) instead of the default CPU wheel.
- ~2 GB free disk for the model + dependencies; the checkpoint itself is ~394 MB.

## Local Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements-dev.txt   # includes runtime deps + test/lint tools
cp .env.example .env                  # adjust as needed
```

### Model Download/Setup

The checkpoint (`models/full_checkpoint.pt`, ~394 MB) and vocab (`models/vocab.json`)
are tracked with **Git LFS** in this repository. After cloning:

```bash
git lfs install
git lfs pull
```

If you obtained this repo as a plain zip (no LFS), place `full_checkpoint.pt` and
`vocab.json` under `models/` yourself, or point `MODEL_PATH`/`VOCAB_PATH` in `.env`
at wherever you've stored them (e.g. a Google Cloud Storage bucket mounted locally,
or a path from a GitHub Release asset).

## Running the Application

```bash
python wsgi.py
# or, for auto-reload during development:
flask --app wsgi run --debug
```

Visit `http://localhost:8080`. Health check: `http://localhost:8080/healthz`.

## Running Tests

```bash
pytest
```

The suite covers preprocessing (tensor shape/dtype/range, error handling on
unreadable/blank images), postprocessing (MathML fixups, token joining), model
construction (forward-pass shapes with the configured hyperparameters, using random
weights — **not** the trained checkpoint), and Flask route wiring. It runs entirely
on CPU and does not require a GPU.

## Docker

```bash
docker build -t equation-ocr .
docker run --rm -p 8080:8080 --env-file .env equation-ocr
# with GPU (requires nvidia-container-toolkit):
docker run --rm --gpus all -p 8080:8080 --env-file .env equation-ocr
```

The image is based on an official NVIDIA CUDA runtime image so the same build works
GPU-accelerated on Cloud Run/GKE or CPU-only anywhere else — device selection is
automatic at startup. Model artifacts under `models/` are baked into the image at
build time; for a smaller image or independently updatable weights, adapt the
Dockerfile to fetch them from Cloud Storage at container startup instead.

## Google Cloud Run Deployment

Cloud Run supports attaching **NVIDIA L4 GPUs** to services (not just batch jobs),
which is the appropriate target for this GPU-optional-but-GPU-accelerated PyTorch
workload — a plain CPU-only Cloud Run service would work (the app falls back to
CPU automatically) but beam-search latency will be materially higher than on GPU.

```bash
gcloud run deploy equation-ocr \
  --source . \
  --region us-central1 \
  --gpu 1 --gpu-type nvidia-l4 \
  --memory 16Gi --cpu 4 \
  --min-instances 0 --max-instances 3 \
  --timeout 300 \
  --set-env-vars DEVICE=auto
```

Notes:

- `--min-instances 0` means cold starts pay the full model-load + CUDA-init cost
  (typically several seconds to tens of seconds); set `--min-instances 1` to keep a
  warm instance if latency matters more than idle cost.
- Concurrency should stay low (Cloud Run default is 80; consider `--concurrency 4-8`)
  since a single loaded model instance serializes GPU work — this is a single-model,
  single-process-per-container design, not a multi-tenant batching server.
- Request timeout is raised to accommodate beam search over long sequences on cold
  or CPU-only instances.
- Container image size and model artifact size both affect cold-start time; if that
  becomes a bottleneck, move the checkpoint to Cloud Storage and load it at startup
  instead of baking it into the image.

## Environment Variables

See `.env.example` for the full list (`MODEL_PATH`, `VOCAB_PATH`, `DEVICE`,
`SECRET_KEY`, `DEBUG`, `HOST`, `PORT`, `MAX_UPLOAD_MB`, `EXAMPLES_DIR`).

## Production Hardening

This is a small, single-purpose tool (upload an image, get LaTeX/MathML back) — not
a multi-user product with accounts or persisted data. The hardening applied reflects
that scope rather than a generic checklist:

**Covered:**
- Security response headers (`X-Content-Type-Options`, `X-Frame-Options`, a
  `Content-Security-Policy` scoped to the exact origins the app loads — no wildcards)
  and `ProxyFix` middleware so scheme/host are correct behind the Docker/Cloud Run
  reverse proxy (`app/__init__.py`).
- Consistent error responses: unknown routes and oversized/failed requests return a
  styled page for browser navigation or a JSON `{"error": ...}` body for `/api/*`,
  instead of Flask's raw default error pages.
- Client-side upload validation (file type and size) sourced from the same
  `app/config.py` values the server enforces (`ALLOWED_EXTENSIONS`,
  `MAX_CONTENT_LENGTH`) — rejected immediately in the browser rather than after a
  round trip that ends in a 413.
- A client-side request timeout (`AbortController`, 90s) so a hung request has a
  visible failure state instead of spinning forever, plus robust handling of a
  non-JSON error response instead of a raw parse exception reaching the user.
- `robots.txt`, favicon, and Open Graph/Twitter meta tags (this is a public tool with
  nothing to hide from crawlers or unlink-preview).

**Deliberately out of scope (and why):**
- **No authentication/authorization, admin area, or CSRF token** — there is no
  session/account state for any of these to protect; every request is anonymous and
  stateless.
- **No React/Next.js/TypeScript rewrite** — Flask/Jinja/vanilla JS is the right-sized
  stack for one form and one JSON endpoint; a framework migration would add build
  tooling and complexity with no corresponding product need.
- **No upload progress bar** — typical inputs are KB-sized crops against an 8MB cap;
  transfer time is negligible next to the multi-second CPU inference time the
  existing loading state already covers.
- **No Playwright/E2E suite** — the core flows (upload, drag-and-drop, keyboard
  operability, error states) have been manually verified end-to-end in a real
  browser; a full E2E framework has no CI pipeline to run in yet, so it's listed
  under Next Steps rather than added speculatively.

## Limitations

- Best on clear, high-contrast, printed (not handwritten) equations; performance on
  handwriting or noisy scans is not guaranteed.
- Beam search is meaningfully slower on CPU than GPU; expect higher latency on
  CPU-only deployments.
- The example gallery re-runs inference on every page load rather than caching
  results — fine for a handful of examples, not designed for a large gallery.
- **GPU-accelerated inference, CUDA initialization, and end-to-end prediction
  correctness with `full_checkpoint.pt` have not been verified in this environment**
  (no GPU was available during this refactor) — see Validation below.

## Validation Status

**Verified on the machine this was refactored on (CPU only, no GPU):**
- Python syntax / imports across `app/`
- Flask app factory boots and all routes respond (`/`, `/examples`, `/about`,
  `/healthz`, `/api/predict` error paths)
- Preprocessing unit tests (shape, dtype, normalization range, error handling)
- Postprocessing unit tests (MathML fixups, token joining)
- Model construction + forward-pass shape tests with the configured hyperparameters
  (random weights, not the trained checkpoint)
- `full_checkpoint.pt` loads successfully via `torch.load(..., weights_only=True)`
  on CPU and its keys match the `encoder`/`seq_model`/`decoder` state_dicts expected
  by `app/deep_learning/model.py` (`scripts/verify_checkpoint.py`)
- **Real end-to-end CPU inference** on an actual example image
  (`scripts/smoke_test_inference.py`): greedy and beam decoding both ran without
  error and produced LaTeX matching `examples/examples.json`'s ground truth almost
  token-for-token
- **`docker build`** completed successfully against the production `Dockerfile`
  (CUDA 12.1 base image, CUDA-enabled torch wheel, Gunicorn) — image size ~5.9GB
- **`docker run`** of that image, on this GPU-less machine: container correctly
  detected no NVIDIA driver, `DEVICE=auto`/`DEVICE=cpu` fell back to CPU as
  designed, Gunicorn booted, the model loaded (~10-15s cold start), and
  `/healthz` plus a real `/api/predict` request (via `curl -F image=@examples/1.png`)
  both returned correct results identical to the non-Docker CPU run above

**Not verified — requires GPU:**
- CUDA initialization and `DEVICE=cuda` code path (the container build includes the
  CUDA-enabled torch wheel, but this machine has no NVIDIA driver to exercise it)
- GPU model loading and GPU tensor operations
- Actual inference latency/throughput on GPU
- GPU memory usage under load
- Prediction correctness end-to-end on GPU (only CPU-loaded weights were exercised)
- Cloud Run GPU deployment behavior (cold start timing, concurrency behavior under
  real GPU load, `--gpu`/`--gpu-type` flags against a real project/quota)

## Contact

Lead developer: Gurpreet Singh — gurpreetsinghmattu2002@gmail.com
