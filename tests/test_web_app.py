import re

import pytest
from fastapi.testclient import TestClient

from videoatlas.web.app import create_app
from videoatlas.web.settings import WebSettings

BASE = "http://localhost"
CSRF = re.compile(r'<meta name="csrf-token" content="([^"]+)"')


@pytest.fixture
def app(tmp_path):
    return create_app(WebSettings(data_dir=tmp_path), start_runner=False)


@pytest.fixture
def client(app):
    with TestClient(app, base_url=BASE) as instance:
        yield instance


def login(client, app):
    response = client.post("/login", data={"key": app.state.key}, follow_redirects=False)
    assert response.status_code == 303
    page = client.get("/")
    return CSRF.search(page.text).group(1)


def finish_doctor(app, status, result=None, error=None):
    store = app.state.store
    job_id = store.enqueue("doctor")
    store.claim_next()
    store.finish(job_id, status, result=result, error=error)
    return job_id


def test_unknown_host_rejected(client):
    response = client.get("/login", headers={"host": "evil.example"})
    assert response.status_code == 400


def test_home_requires_login(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_htmx_request_without_session_gets_hx_redirect(client):
    response = client.get("/fragments/status", headers={"HX-Request": "true"}, follow_redirects=False)
    assert response.status_code == 401
    assert response.headers["HX-Redirect"] == "/login"


def test_login_wrong_key_rejected(client):
    response = client.post("/login", data={"key": "wrong"}, follow_redirects=False)
    assert response.status_code == 401
    assert "set-cookie" not in response.headers
    assert "did not match" in response.text


def test_login_sets_strict_httponly_cookie(client, app):
    response = client.post("/login", data={"key": app.state.key}, follow_redirects=False)
    cookie = response.headers["set-cookie"].lower()
    assert "va_session=" in cookie
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/" in cookie


def test_login_with_foreign_origin_rejected(client, app):
    response = client.post("/login", data={"key": app.state.key}, headers={"Origin": "http://evil.example"},
                           follow_redirects=False)
    assert response.status_code == 403


def test_oversized_form_rejected(client):
    response = client.post("/login", data={"key": "x" * 10000}, follow_redirects=False)
    assert response.status_code == 413


def test_logout_ends_session(client, app):
    token = login(client, app)
    response = client.post("/logout", data={"csrf_token": token}, follow_redirects=False)
    assert response.status_code == 303
    assert client.get("/", follow_redirects=False).status_code == 303


def test_post_without_csrf_rejected(client, app):
    login(client, app)
    response = client.post("/jobs/doctor", follow_redirects=False)
    assert response.status_code == 403
    assert app.state.store.recent() == []


def test_post_with_wrong_csrf_rejected(client, app):
    login(client, app)
    response = client.post("/jobs/doctor", headers={"X-CSRF-Token": "wrong"}, follow_redirects=False)
    assert response.status_code == 403


def test_post_with_own_origin_accepted(client, app):
    token = login(client, app)
    response = client.post("/jobs/doctor", headers={"X-CSRF-Token": token, "Origin": "http://localhost",
                                                    "Sec-Fetch-Site": "same-origin"}, follow_redirects=False)
    assert response.status_code == 303


def test_null_origin_rejected(client, app):
    token = login(client, app)
    response = client.post("/jobs/doctor", headers={"X-CSRF-Token": token, "Origin": "null"}, follow_redirects=False)
    assert response.status_code == 403


def test_post_with_foreign_origin_rejected(client, app):
    token = login(client, app)
    response = client.post("/jobs/doctor", headers={"X-CSRF-Token": token, "Origin": "http://evil.example"},
                           follow_redirects=False)
    assert response.status_code == 403
    assert app.state.store.recent() == []


def test_cross_site_fetch_rejected(client, app):
    token = login(client, app)
    response = client.post("/jobs/doctor", headers={"X-CSRF-Token": token, "Sec-Fetch-Site": "cross-site"},
                           follow_redirects=False)
    assert response.status_code == 403


def test_security_headers(client):
    response = client.get("/login")
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "default-src 'none'" in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["cache-control"] == "no-store"


def test_pages_reference_no_external_scripts(client, app):
    pages = [client.get("/login").text]
    login(client, app)
    pages.append(client.get("/").text)
    for page in pages:
        sources = re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', page)
        assert sources, "expected local assets"
        assert all(source.startswith("/static/") for source in sources), sources
        assert "<script>" not in page and "style=" not in page


def test_static_htmx_served(client):
    response = client.get("/static/vendor/htmx-2.0.11.min.js")
    assert response.status_code == 200
    assert "htmx" in response.text[:2000]


def test_check_gpu_enqueues_one_doctor_job(client, app):
    token = login(client, app)
    headers = {"X-CSRF-Token": token, "HX-Request": "true"}
    first = client.post("/jobs/doctor", headers=headers)
    second = client.post("/jobs/doctor", headers=headers)
    assert first.status_code == second.status_code == 200
    jobs = app.state.store.recent()
    assert [(job.kind, job.status) for job in jobs] == [("doctor", "queued")]
    assert "GPU check" in second.text


def test_check_gpu_without_htmx_redirects_home(client, app):
    token = login(client, app)
    response = client.post("/jobs/doctor", data={"csrf_token": token}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_stop_button_requests_stop(client, app):
    token = login(client, app)
    job_id = app.state.store.enqueue("doctor")
    response = client.post(f"/jobs/{job_id}/stop", headers={"X-CSRF-Token": token, "HX-Request": "true"})
    assert response.status_code == 200
    assert app.state.store.get(job_id).status == "stopped"


def test_stop_unknown_job_is_404(client, app):
    token = login(client, app)
    response = client.post("/jobs/999/stop", headers={"X-CSRF-Token": token})
    assert response.status_code == 404


def test_home_shows_gpu_not_checked(client, app):
    login(client, app)
    assert "Not checked yet" in client.get("/").text


def test_home_shows_gpu_ok(client, app):
    finish_doctor(app, "succeeded", result={"providers": {"detector": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                                                           "face01": ["CUDAExecutionProvider", "CPUExecutionProvider"]},
                                             "embedding_model": "face01", "runtime": "onnxruntime"})
    login(client, app)
    page = client.get("/").text
    assert "All models run on the GPU" in page
    assert "detector" in page and "face01" in page


def test_home_shows_cpu_warning(client, app):
    finish_doctor(app, "succeeded", result={"providers": {"detector": ["CPUExecutionProvider"]}})
    login(client, app)
    assert "Some models run on the CPU" in client.get("/").text


def test_home_shows_failed_check(client, app):
    finish_doctor(app, "failed", error="error: GPU was requested, but these models run on the CPU: face01")
    login(client, app)
    page = client.get("/").text
    assert "GPU check failed" in page
    assert "these models run on the CPU: face01" in page


def test_home_escapes_job_output(client, app):
    finish_doctor(app, "failed", error="error: <script>alert(1)</script>")
    login(client, app)
    page = client.get("/").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_home_shows_disk_space(client, app):
    login(client, app)
    assert re.search(r"[\d.]+ GB free of [\d.]+ GB", client.get("/").text)


def test_status_fragment_polls_faster_while_job_active(client, app):
    login(client, app)
    idle = client.get("/fragments/status", headers={"HX-Request": "true"}).text
    assert "every 15s" in idle
    app.state.store.enqueue("doctor")
    busy = client.get("/fragments/status", headers={"HX-Request": "true"}).text
    assert "every 2s" in busy
