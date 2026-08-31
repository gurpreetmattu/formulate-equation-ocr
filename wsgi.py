"""WSGI entrypoint for production servers (Gunicorn) and `python wsgi.py` for local dev."""
from app import create_app
from app.config import Config

app = create_app()

if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
