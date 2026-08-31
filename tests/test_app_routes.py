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
