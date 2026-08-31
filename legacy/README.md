# Legacy Streamlit Prototype

This directory preserves the original Streamlit application exactly as it existed
before the Flask/production refactor, for reference only. It is **not** part of the
deployed application and is excluded from the Docker build (`.dockerignore`).

- `streamlit_app.py` — the original, working root-level `app.py` (CPU-forced,
  relative-path-free of the stale `C:\Users\gurpr\...` paths).
- `streamlit_app_dev_copy.py` — an older development copy (`code/app.py`) with
  hardcoded machine-specific paths and CUDA device selection; kept only in case it
  contains context not present in `streamlit_app.py`.
- `.streamlit/config.toml` — the Streamlit theme config used by the prototype.

To run the legacy app (requires `streamlit` installed separately, not part of
`requirements.txt`):

```bash
pip install streamlit
streamlit run legacy/streamlit_app.py
```
