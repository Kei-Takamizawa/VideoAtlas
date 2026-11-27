# Browser GUI Stage 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A loopback-only browser GUI server, protected by an access key, that shows whether every model runs on the GPU and runs GPU jobs one at a time in a separate process.

**Architecture:** A FastAPI app renders Jinja2 pages; vendored htmx refreshes fragments by polling. A single runner thread takes queued jobs from a small SQLite database and runs each one as a `python -m videoatlas.cli ...` subprocess, so model crashes never take down the GUI and the GPU never runs two jobs at once. The web package never imports the analysis engine.

**Scope change from the design doc:** stage 1 no longer extracts a Qt-free service layer from `controller.py`. Photo grouping now runs through the CLI (`videoatlas.cli`), so the GUI only needs to start CLI jobs and read their output folders. The extraction moves to the stage that brings the video library into the GUI.

**Tech Stack:** Python 3.12, FastAPI 0.162.7, Starlette 1.10.0, Uvicorn 0.54.0, Jinja2 3.1.6, htmx 2.0.11 (vendored), SQLite, pytest + httpx for tests.

**Spec:** User-approved design doc "VideoAtlas ブラウザGUI 設計書" (private Claude Doc, 2026-11-26), stage 1 of its five-stage plan; `docs/design-specification.md` (amended by Task 5).

## Global Constraints

- Bind only to a loopback address (`127.0.0.1`, `::1`, `localhost`); refuse any other host. Default port 8765.
- No new LAN port; access is through `ssh -N -L 8765:127.0.0.1:8765 videoatlas.incus`.
- Every page and fragment except `/login` and `/static/*` requires a session; every non-GET request requires a valid CSRF token and a same-origin `Origin` header when one is sent.
- Allowed `Host` values: `localhost`, `127.0.0.1`.
- No external CDN, no inline scripts or styles; CSP `script-src 'self'`.
- Dependencies pinned with hashes in `requirements-web.txt`, every package released at least 14 days before pinning.
- GPU checks always use `--device cuda`; a CPU fallback is shown as a failure, never hidden.
- UI text in English. Logs never contain face images, embeddings, or the access key.
- Do not modify existing analysis modules in this stage; the web package calls the CLI as a subprocess.

## Review Focus

1. An access key file copied with loose permissions (for example 0644) → the server tightens it to 0600 before use. Test in Task 1.
2. Double-clicking "Check GPU" → only one doctor job is queued while one is queued or running. Test in Task 2.
3. A job command that prints non-JSON or huge output → the job fails with a short reason; the runner keeps working and stores at most 1 MiB. Test in Task 2.
4. The server stops while a job runs → the child process is terminated, and on the next start the job shows as `interrupted`. Tests in Task 2.
5. The session expires while the home page polls → htmx requests get `HX-Redirect: /login` instead of the login page swapped into a fragment. Test in Task 3.

---

### Task 1: Settings and access security

**Files:**
- Create: `videoatlas/web/__init__.py`, `videoatlas/web/settings.py`, `videoatlas/web/security.py`
- Test: `tests/test_web_security.py`

**Interfaces:**
- Produces: `WebSettings(data_dir: Path, host: str = "127.0.0.1", port: int = 8765, python: str = sys.executable, repo_root: Path = <repo root>)` with `validate() -> None` (raises `ValueError` for a non-loopback host or a port outside 1024–65535) and properties `web_dir -> data_dir / "web"`, `jobs_db -> web_dir / "jobs.sqlite3"`, `key_file -> web_dir / "access-key"`. `default_data_dir() -> Path` returns `$VIDEOATLAS_DATA_DIR` or `~/videoatlas-data`.
- Produces: `load_or_create_key(path: Path) -> str` (43-character URL-safe token, file mode 0600, tightens looser modes), `rotate_key(path: Path) -> str`, `keys_match(given: str, expected: str) -> bool` (constant time).
- Produces: `SessionStore(ttl_seconds: int = 7 * 24 * 3600)` with `create() -> Session`, `get(token: str | None) -> Session | None`, `delete(token: str) -> None`; `Session(token: str, csrf: str, created: float)`.

- [ ] **Step 1: Write failing tests** `test_settings_reject_non_loopback_host` (`"0.0.0.0"` and `"192.168.1.5"` raise), `test_access_key_created_owner_only` (mode `0o600`, `len >= 43`), `test_access_key_permissions_tightened` (pre-existing 0644 file becomes 0600, value unchanged), `test_rotate_key_changes_value`, `test_session_expires` (ttl 0 → `get` returns None), `test_keys_match`.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_web_security.py -q` → FAIL (module missing).
- [ ] **Step 3: Implement** with `secrets.token_urlsafe(32)`, `os.open(..., 0o600)` for creation, `os.chmod` to tighten, `hmac.compare_digest`.
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `Add loopback-only web settings, access key, and sessions`.

### Task 2: Job store and runner

**Files:**
- Create: `videoatlas/web/jobs.py`
- Test: `tests/test_web_jobs.py`

**Interfaces:**
- Produces: `Job` dataclass (`id, kind, status, created_at, started_at, finished_at, progress_done, progress_total, result, error, log_tail, stop_requested`); statuses `queued | running | succeeded | failed | stopped | interrupted`.
- Produces: `JobStore(path: Path, kinds: dict[str, JobKind] = JOB_KINDS)` with `enqueue(kind: str, params: dict | None = None) -> int` (raises `ValueError` for an unknown kind), `enqueue_unique(kind) -> int` (returns the existing queued/running job id when there is one), `get(id) -> Job | None`, `recent(limit=10) -> list[Job]`, `latest_finished(kind) -> Job | None`, `claim_next() -> Job | None`, `set_progress(id, done, total, log_tail)`, `finish(id, status, result=None, error=None)`, `request_stop(id) -> bool`, `mark_interrupted() -> int`.
- Produces: `JobKind(name: str, title: str, argv: Callable[[dict], list[str]], json_result: bool)`; `JOB_KINDS = {"doctor": JobKind("doctor", "GPU check", lambda p: ["-m", "videoatlas.cli", "doctor", "--device", "cuda"], True)}`.
- Produces: `JobRunner(store: JobStore, settings: WebSettings, kinds: dict[str, JobKind] = JOB_KINDS, poll_seconds: float = 0.5)` with `start()`, `close(timeout=15, record=True)` (`record=False` leaves the database as a crash would, for tests); it marks interrupted jobs at construction.

- [ ] **Step 1: Write failing tests** using a test-only `JobKind` whose argv runs `python -c` scripts:
  - `test_jobs_run_one_at_a_time` (two 0.3 s jobs never overlap: second `started_at >= ` first `finished_at`)
  - `test_progress_parsed_from_stderr` (`"3/10 images"` → `progress_done == 3`, `progress_total == 10`)
  - `test_json_result_stored` (stdout `{"ok": 1}` → `result == {"ok": 1}`)
  - `test_failed_job_records_error_line` (exit 1 with `error: GPU was requested` → `status == "failed"`, `error` contains that line)
  - `test_invalid_json_fails_job_and_runner_continues` (non-JSON stdout → failed, next job still succeeds)
  - `test_output_capped` (2 MiB stdout → stored text ≤ 1 MiB)
  - `test_stop_terminates_running_job` (status `stopped` within 5 s)
  - `test_close_terminates_child_and_restart_marks_interrupted`
  - `test_enqueue_unique_deduplicates`, `test_unknown_kind_rejected`, `test_doctor_kind_requires_cuda`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_web_jobs.py -q` → FAIL.
- [ ] **Step 3: Implement.** SQLite in WAL mode with a new connection per call (thread safe). `claim_next` uses `BEGIN IMMEDIATE`. The runner starts each child with `subprocess.Popen([settings.python, *argv], cwd=settings.repo_root, start_new_session=True, env={..., "PYTHONUNBUFFERED": "1"})`, reads stderr in a helper thread (regex `^(\d+)/(\d+)\b`, last 50 lines kept), stops with SIGTERM then SIGKILL after 10 s, and caps stdout at 1 MiB.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `Run GUI jobs one at a time in a separate process`.

### Task 3: Web app, pages, and security middleware

**Files:**
- Create: `videoatlas/web/app.py`, `videoatlas/web/status.py`, `videoatlas/web/templates/{base,login,home,_status}.html`, `videoatlas/web/static/{app.css,app.js}`, `videoatlas/web/static/vendor/htmx-2.0.11.min.js`, `videoatlas/web/static/vendor/README.md`
- Test: `tests/test_web_app.py`

**Interfaces:**
- Consumes: Task 1 and Task 2 interfaces.
- Produces: `create_app(settings: WebSettings, runner: JobRunner | None = None, start_runner: bool = True) -> FastAPI`. Routes: `GET /login`, `POST /login` (form field `key`), `POST /logout`, `GET /`, `GET /fragments/status`, `POST /jobs/doctor`, `POST /jobs/{job_id}/stop`, static under `/static`.
- Produces: `gpu_summary(job: Job | None) -> dict` (`state` is `unknown | ok | cpu | failed`, plus `models: list[(name, provider)]`, `error`), `disk_summary(path: Path) -> dict` (`free_gb`, `total_gb`).

- [ ] **Step 1: Write failing tests** (client `base_url="http://localhost"`):
  - `test_unknown_host_rejected` (`Host: evil.example` → 400)
  - `test_home_requires_login` (303 to `/login`); `test_htmx_request_without_session_gets_hx_redirect`
  - `test_login_wrong_key_rejected` (401, no cookie); `test_login_sets_strict_httponly_cookie`
  - `test_post_without_csrf_rejected` (403); `test_post_with_foreign_origin_rejected` (403)
  - `test_security_headers` (CSP has `script-src 'self'` and `frame-ancestors 'none'`; `nosniff`; `no-store` on HTML)
  - `test_pages_reference_no_external_scripts`
  - `test_check_gpu_enqueues_one_doctor_job`; `test_stop_button_requests_stop`
  - `test_home_shows_gpu_ok`, `test_home_shows_cpu_warning`, `test_home_shows_failed_check`, `test_home_shows_disk_space`
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_web_app.py -q` → FAIL.
- [ ] **Step 3: Implement.** `TrustedHostMiddleware(allowed_hosts=["localhost", "127.0.0.1"])`; a middleware adds security headers; a dependency checks the `va_session` cookie; unsafe methods check the `X-CSRF-Token` header or `csrf_token` form field and the `Origin` header. Parse form bodies with `urllib.parse.parse_qs` (limit 4 KiB) instead of adding `python-multipart`. `app.js` copies the CSRF meta tag into htmx requests. htmx config meta: `includeIndicatorStyles=false`, `allowEval=false`, `allowScriptTags=false`. The status fragment polls every 2 s.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `Add the browser GUI home page with GPU status and jobs`.

### Task 4: Command line, auto-start, and access instructions

**Files:**
- Create: `videoatlas/web/__main__.py`, `Scripts/videoatlas-web.service`, `docs/WEB_GUI.md`
- Test: `tests/test_web_cli.py`

**Interfaces:**
- Produces: `python -m videoatlas.web serve [--data-dir DIR] [--port N]`, `print-key`, `rotate-key`; `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Write failing tests** `test_print_key_creates_and_prints_key`, `test_rotate_key_prints_new_key`, `test_serve_rejects_non_loopback_host` (exit code 2 with a message, without starting a server).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (`uvicorn.run(create_app(settings), host, port, proxy_headers=False, server_header=False)`). **Step 4: Run** → PASS.
- [ ] **Step 5: Write** the systemd user unit (`Restart=on-failure`, `NoNewPrivileges=yes`, `PrivateTmp=yes`) and `docs/WEB_GUI.md` (start, key, SSH tunnel, security model, what stage 1 does not do yet).
- [ ] **Step 6: Commit** `Add web GUI commands, systemd unit, and access guide`.

### Task 5: Specification amendment and docs

**Files:**
- Modify: `docs/design-specification.md` (pointer lines in §2 and §23, new "Amendments" section A1), `docs/DEVELOPMENT.md` (repo map rows, web test command), `docs/IMPLEMENTATION_STATUS.md` (dated stage 1 entry with the checks actually run)

- [ ] **Step 1:** Add amendment A1 (2026-11-27): processing on a Linux server with an NVIDIA GPU in an isolated container; GUI in the browser, loopback-only, reached by SSH tunnel; PySide6 GUI frozen until browser parity; personal use unchanged. Do not rewrite the original sections.
- [ ] **Step 2:** Run the full web suite and `git diff --check`; take a Playwright screenshot of the login and home pages for review.
- [ ] **Step 3: Commit** `Amend the specification for the browser GUI`.
