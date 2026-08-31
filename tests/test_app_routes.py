"""Flask route tests. These exercise app startup and HTTP wiring; they do not
assert on prediction correctness (that requires a GPU -- see README)."""
import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_recognize_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_about_page_loads(client):
    resp = client.get("/about")
    assert resp.status_code == 200


def test_examples_page_loads(client):
    resp = client.get("/examples")
    assert resp.status_code == 200


def test_healthz_reports_status(client):
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    body = resp.get_json()
    assert "status" in body


def test_predict_without_image_returns_400(client):
    resp = client.post("/api/predict", data={})
    assert resp.status_code in (400, 503)


def test_predict_with_bad_extension_returns_400(client, sample_equation_png_bytes):
    import io
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(sample_equation_png_bytes), "eq.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (400, 503)


def test_unknown_page_route_returns_styled_404(client):
    resp = client.get("/this-route-does-not-exist")
    assert resp.status_code == 404
    assert resp.content_type.startswith("text/html")
    assert b"Page not found" in resp.data


def test_unknown_api_route_returns_json_404(client):
    resp = client.get("/api/this-does-not-exist")
    assert resp.status_code == 404
    assert resp.get_json()["error"]


def test_oversized_upload_returns_json_413(sample_equation_png_bytes):
    import io

    from app import create_app
    from app.config import Config

    class TinyLimitConfig(Config):
        MAX_CONTENT_LENGTH = 100  # bytes -- smaller than any real image

    app = create_app(TinyLimitConfig)
    app.config.update(TESTING=True)
    with app.test_client() as client:
        resp = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(sample_equation_png_bytes), "eq.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413
        assert "too large" in resp.get_json()["error"].lower()


def test_security_headers_present(client):
    resp = client.get("/")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in resp.headers


def test_robots_txt(client):
    resp = client.get("/robots.txt")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/plain")
    assert b"User-agent" in resp.data
