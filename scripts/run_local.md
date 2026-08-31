# Quick local run reference

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements-dev.txt
cp .env.example .env
python scripts/verify_checkpoint.py   # optional: confirm checkpoint loads
python wsgi.py
```
