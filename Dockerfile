# Production image for the equation-recognition Flask app.
#
# Base: official NVIDIA CUDA runtime image so this same image runs correctly
# on Cloud Run with GPU (NVIDIA L4) *and* falls back to CPU-only execution
# on GPU-less environments (torch.cuda.is_available() -> False, DEVICE=auto
# in app/config.py picks CPU automatically). If you only ever deploy to a
# CPU-only target, you can switch the base image to `python:3.11-slim` and
# drop the CUDA toolkit lines to shrink the image significantly.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps: python3.11 + libs required by opencv-python-headless / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
# CUDA-enabled torch wheel matching the base image's CUDA 12.1 runtime,
# installed first so the plain PyPI "torch==2.4.1" pin below is already
# satisfied and pip does not overwrite it with a CPU-only wheel.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY wsgi.py .
COPY examples/ ./examples/
# The trained checkpoint and vocab are expected under models/ at build time,
# or mounted/downloaded at deploy time -- see README "Model Download/Setup".
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Cloud Run sets $PORT at runtime; default to 8080 for local `docker run`.
# Single worker: the model is loaded once into memory per worker process,
# and GPU memory is not safely shared across multiple worker processes.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 wsgi:app"]
