import time

import pytest

from videoatlas.web.jobs import JOB_KINDS, MAX_OUTPUT_BYTES, JobKind, JobRunner, JobStore
from videoatlas.web.settings import WebSettings

SCRIPT = JobKind("script", "Test script", lambda params: ["-c", params["code"]], json_result=False)
JSON_SCRIPT = JobKind("json-script", "Test JSON script", lambda params: ["-c", params["code"]], json_result=True)
KINDS = {kind.name: kind for kind in (SCRIPT, JSON_SCRIPT)}


@pytest.fixture
def settings(tmp_path):
    return WebSettings(data_dir=tmp_path)


@pytest.fixture
def store(settings):
    return JobStore(settings.jobs_db, KINDS)


@pytest.fixture
def runner(store, settings):
    instance = JobRunner(store, settings, KINDS, poll_seconds=0.05)
    instance.start()
    yield instance
    instance.close()


def wait_for(store, job_id, statuses=("succeeded", "failed", "stopped"), timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id)
        if job.status in statuses:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stayed {store.get(job_id).status}")


def test_unknown_kind_rejected(store):
    with pytest.raises(ValueError, match="Unknown job"):
        store.enqueue("rm -rf /")


def test_doctor_kind_requires_cuda():
    argv = JOB_KINDS["doctor"].argv({})
    assert argv[:3] == ["-m", "videoatlas.cli", "doctor"]
    assert argv[argv.index("--device") + 1] == "cuda"


def test_enqueue_unique_deduplicates(store):
    first = store.enqueue_unique("script", {"code": "pass"})
    assert store.enqueue_unique("script", {"code": "pass"}) == first
    store.finish(first, "succeeded")
    assert store.enqueue_unique("script", {"code": "pass"}) != first


def test_jobs_run_one_at_a_time(store, runner):
    code = "import time; time.sleep(0.3)"
    first = store.enqueue("script", {"code": code})
    second = store.enqueue("script", {"code": code})
    first_job = wait_for(store, first)
    second_job = wait_for(store, second)
    assert first_job.status == second_job.status == "succeeded"
    assert second_job.started_at >= first_job.finished_at


def test_progress_parsed_from_stderr(store, runner):
    code = "import sys, time; print('3/10 images (1.0 images/s)', file=sys.stderr, flush=True); time.sleep(5)"
    job_id = store.enqueue("script", {"code": code})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and store.get(job_id).progress_total is None:
        time.sleep(0.02)
    job = store.get(job_id)
    assert (job.progress_done, job.progress_total) == (3, 10)
    assert "3/10 images" in job.log_tail
    store.request_stop(job_id)
    wait_for(store, job_id)


def test_json_result_stored(store, runner):
    job_id = store.enqueue("json-script", {"code": "print('{\"ok\": 1}')"})
    job = wait_for(store, job_id)
    assert job.status == "succeeded"
    assert job.result == {"ok": 1}


def test_failed_job_records_error_line(store, runner):
    code = "import sys; print('warning: noise', file=sys.stderr); print('error: GPU was requested, but these models run on the CPU: FACE01', file=sys.stderr); sys.exit(1)"
    job = wait_for(store, store.enqueue("script", {"code": code}))
    assert job.status == "failed"
    assert job.error == "error: GPU was requested, but these models run on the CPU: FACE01"


def test_invalid_json_fails_job_and_runner_continues(store, runner):
    bad = wait_for(store, store.enqueue("json-script", {"code": "print('not json')"}))
    assert bad.status == "failed"
    assert "JSON" in bad.error
    good = wait_for(store, store.enqueue("script", {"code": "pass"}))
    assert good.status == "succeeded"


def test_output_capped(store, runner):
    code = "import sys; sys.stdout.write('x' * (2 * 1024 * 1024))"
    job = wait_for(store, store.enqueue("json-script", {"code": code}))
    assert job.status == "failed"
    assert "too large" in job.error
    assert len(job.log_tail.encode()) <= MAX_OUTPUT_BYTES


def test_stop_terminates_running_job(store, runner):
    job_id = store.enqueue("script", {"code": "import time; time.sleep(60)"})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and store.get(job_id).status != "running":
        time.sleep(0.02)
    assert store.request_stop(job_id)
    job = wait_for(store, job_id, timeout=5)
    assert job.status == "stopped"


def test_stop_queued_job(store):
    job_id = store.enqueue("script", {"code": "pass"})
    assert store.request_stop(job_id)
    assert store.get(job_id).status == "stopped"
    assert store.claim_next() is None


def test_stop_finished_job_is_refused(store):
    job_id = store.enqueue("script", {"code": "pass"})
    store.finish(job_id, "succeeded")
    assert not store.request_stop(job_id)


def test_close_terminates_child_and_restart_marks_interrupted(store, settings, tmp_path):
    marker = tmp_path / "still-running"
    code = f"import time, pathlib; time.sleep(1.5); pathlib.Path({str(marker)!r}).write_text('x')"
    runner = JobRunner(store, settings, KINDS, poll_seconds=0.05)
    runner.start()
    job_id = store.enqueue("script", {"code": code})
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and store.get(job_id).status != "running":
        time.sleep(0.02)
    # Simulate a crash of the server process: the row stays "running".
    runner.close(record=False)
    time.sleep(2)
    assert not marker.exists(), "the child process outlived the runner"
    assert store.get(job_id).status == "running"
    JobRunner(JobStore(settings.jobs_db, KINDS), settings, KINDS)
    assert store.get(job_id).status == "interrupted"


def test_latest_finished_and_recent(store):
    first = store.enqueue("script", {"code": "pass"})
    store.finish(first, "failed", error="error: boom")
    second = store.enqueue("script", {"code": "pass"})
    store.finish(second, "succeeded", result={"a": 1})
    third = store.enqueue("script", {"code": "pass"})
    assert store.latest_finished("script").id == second
    assert [job.id for job in store.recent(2)] == [third, second]
