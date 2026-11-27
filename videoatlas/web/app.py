"""FastAPI application for the browser GUI.

Every page except the sign-in page needs a session. Every request that changes
something needs the session's CSRF token, and a same-origin request. The server
answers only to the Host names of a loopback address.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .jobs import JOB_KINDS, Job, JobRunner, JobStore
from .security import Session, SessionStore, keys_match, load_or_create_key
from .settings import LOOPBACK_HOSTS, WebSettings
from .status import ago, disk_summary, gpu_summary, progress_percent

HERE = Path(__file__).resolve().parent
SESSION_COOKIE = "va_session"
MAX_FORM_BYTES = 4096
# Host names a browser may use: the server's own loopback addresses, written as Starlette
# parses a Host header (an IPv6 address keeps its brackets). An IP literal cannot be rebound by DNS.
ALLOWED_HOSTS = [f"[{host}]" if ":" in host else host for host in LOOPBACK_HOSTS]
CONTENT_SECURITY_POLICY = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                           "connect-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    # "no-referrer" would make browsers send "Origin: null" on form posts, which the same-origin check refuses.
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class LoginRequired(Exception):
    """Raised when a page needs a session and the request has none."""


async def read_form(request: Request) -> dict[str, str]:
    """A small URL-encoded form body, read once. Larger bodies are refused."""
    cached = getattr(request.state, "form", None)
    if cached is not None:
        return cached
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_FORM_BYTES:
            raise HTTPException(413, "The form is too large.")
    fields = {name: values[0] for name, values in parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True).items()}
    request.state.form = fields
    return fields


def check_same_origin(request: Request) -> None:
    expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
    origin = request.headers.get("origin")
    if origin is not None and origin != expected:
        raise HTTPException(403, "Requests from other sites are refused.")
    if request.headers.get("sec-fetch-site") in ("cross-site", "same-site"):
        raise HTTPException(403, "Requests from other sites are refused.")


def job_view(job: Job) -> dict:
    kind = JOB_KINDS.get(job.kind)
    return {"id": job.id, "title": kind.title if kind else job.kind, "status": job.status, "active": job.active,
            "percent": progress_percent(job), "created": ago(job.created_at), "error": job.error}


def create_app(settings: WebSettings, runner: JobRunner | None = None, start_runner: bool = True) -> FastAPI:
    settings.validate()
    store = runner.store if runner is not None else JobStore(settings.jobs_db)
    runner = runner if runner is not None else JobRunner(store, settings)
    key = load_or_create_key(settings.key_file)
    sessions = SessionStore()
    templates = Jinja2Templates(directory=str(HERE / "templates"))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_runner:
            runner.start()
        try:
            yield
        finally:
            runner.close()

    app = FastAPI(title="VideoAtlas", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.store, app.state.runner = settings, store, runner
    app.state.key, app.state.sessions = key, sessions

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if not request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, _error: LoginRequired):
        if request.headers.get("HX-Request"):
            return Response(status_code=401, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    def current_session(request: Request) -> Session:
        session = sessions.get(request.cookies.get(SESSION_COOKIE))
        if session is None:
            raise LoginRequired()
        return session

    async def changing_session(request: Request, session: Session = Depends(current_session)) -> Session:
        check_same_origin(request)
        token = request.headers.get("x-csrf-token")
        if token is None:
            token = (await read_form(request)).get("csrf_token", "")
        if not hmac.compare_digest(token.encode("utf-8"), session.csrf.encode("utf-8")):
            raise HTTPException(403, "The request is missing a valid CSRF token.")
        return session

    def status_context(session: Session) -> dict:
        jobs = [job_view(job) for job in store.recent(10)]
        return {"csrf": session.csrf, "gpu": gpu_summary(store.latest_finished("doctor")),
                "disk": disk_summary(settings.data_dir), "jobs": jobs, "active": any(job["active"] for job in jobs)}

    def after_change(request: Request, session: Session):
        if request.headers.get("HX-Request"):
            return templates.TemplateResponse(request, "_status.html", status_context(session))
        return RedirectResponse("/", status_code=303)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        if sessions.get(request.cookies.get(SESSION_COOKIE)) is not None:
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    async def login(request: Request):
        check_same_origin(request)
        given = (await read_form(request)).get("key", "").strip()
        if not keys_match(given, key):
            return templates.TemplateResponse(request, "login.html", {"error": "The access key did not match."}, status_code=401)
        session = sessions.create()
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(SESSION_COOKIE, session.token, max_age=sessions.ttl_seconds, path="/", httponly=True, samesite="strict")
        return response

    @app.post("/logout")
    async def logout(session: Session = Depends(changing_session)):
        sessions.delete(session.token)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
        return response

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request, session: Session = Depends(current_session)):
        return templates.TemplateResponse(request, "home.html", status_context(session))

    @app.get("/fragments/status", response_class=HTMLResponse)
    async def status_fragment(request: Request, session: Session = Depends(current_session)):
        return templates.TemplateResponse(request, "_status.html", status_context(session))

    @app.post("/jobs/doctor")
    async def check_gpu(request: Request, session: Session = Depends(changing_session)):
        store.enqueue_unique("doctor")
        return after_change(request, session)

    @app.post("/jobs/{job_id}/stop")
    async def stop_job(request: Request, job_id: int, session: Session = Depends(changing_session)):
        if store.get(job_id) is None:
            raise HTTPException(404, "No such job.")
        store.request_stop(job_id)
        return after_change(request, session)

    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    return app
