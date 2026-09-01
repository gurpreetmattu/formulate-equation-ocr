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


def test_predict_with_valid_image_returns_expected_shape(client, sample_equation_png_bytes):
    """End-to-end happy path: a valid image against whatever model is
    actually loaded (CPU here). 503 is accepted only for environments where
    no checkpoint is present (see README); this does NOT assert the
    predicted LaTeX is *correct* -- only that a real inference call
    succeeds and returns the response shape the frontend depends on."""
    import io
    resp = client.post(
        "/api/predict",
        data={"image": (io.BytesIO(sample_equation_png_bytes), "eq.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (200, 503)
    if resp.status_code == 503:
        return
    body = resp.get_json()
    for key in ("preview_image", "greedy_latex", "greedy_mathml", "beam_latex", "beam_mathml"):
        assert key in body
    assert body["preview_image"].startswith("data:image/png;base64,")
    assert isinstance(body["greedy_latex"], str)
    assert isinstance(body["beam_latex"], str)


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


def test_predict_rate_limit_returns_json_429(sample_equation_png_bytes):
    """Exercises actual enforcement (not just that the decorator is wired
    up): TESTING is deliberately left unset here, since the exempt_when
    check in routes.py bypasses rate limiting whenever TESTING is True --
    every other test in this file relies on exactly that bypass."""
    import io

    from app import create_app
    from app.config import Config
    from app.extensions import limiter

    class OneRequestPerMinuteConfig(Config):
        RATE_LIMIT_PREDICT = "1 per minute"

    app = create_app(OneRequestPerMinuteConfig)
    limiter.reset()  # isolate from any counters other tests may have left behind
    with app.test_client() as client:
        first = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(sample_equation_png_bytes), "eq.png")},
            content_type="multipart/form-data",
        )
        assert first.status_code in (200, 503)

        second = client.post(
            "/api/predict",
            data={"image": (io.BytesIO(sample_equation_png_bytes), "eq.png")},
            content_type="multipart/form-data",
        )
        assert second.status_code == 429
        assert "too many requests" in second.get_json()["error"].lower()
    limiter.reset()  # leave a clean slate for tests that run after this one


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


def test_serve_example_image_known_file_returns_image(client):
    resp = client.get("/examples/1.png")
    assert resp.status_code == 200
    assert resp.content_type.startswith("image/")


def test_serve_example_image_unknown_file_returns_404(client):
    resp = client.get("/examples/not-a-real-file.png")
    assert resp.status_code == 404
    assert resp.get_json()["error"]


def test_recognize_page_includes_sample_chips(client):
    resp = client.get("/")
    assert b"sample-chip" in resp.data


def test_theme_cookie_reflected_in_html_attribute(client):
    client.set_cookie("theme", "dark")
    resp = client.get("/")
    assert b'data-theme="dark"' in resp.data


def test_no_theme_cookie_renders_empty_attribute(client):
    resp = client.get("/")
    assert b'data-theme=""' in resp.data
