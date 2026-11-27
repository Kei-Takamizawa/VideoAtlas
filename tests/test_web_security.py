import stat
import time

import pytest

from videoatlas.web.security import SessionStore, keys_match, load_or_create_key, rotate_key
from videoatlas.web.settings import WebSettings


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5", "example.com", "::"])
def test_settings_reject_non_loopback_host(tmp_path, host):
    with pytest.raises(ValueError, match="loopback"):
        WebSettings(data_dir=tmp_path, host=host).validate()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_settings_accept_loopback_host(tmp_path, host):
    WebSettings(data_dir=tmp_path, host=host).validate()


@pytest.mark.parametrize("port", [0, 80, 70000])
def test_settings_reject_bad_port(tmp_path, port):
    with pytest.raises(ValueError, match="port"):
        WebSettings(data_dir=tmp_path, port=port).validate()


def test_settings_paths(tmp_path):
    settings = WebSettings(data_dir=tmp_path)
    assert settings.web_dir == tmp_path / "web"
    assert settings.jobs_db == tmp_path / "web" / "jobs.sqlite3"
    assert settings.key_file == tmp_path / "web" / "access-key"


def test_access_key_created_owner_only(tmp_path):
    path = tmp_path / "web" / "access-key"
    key = load_or_create_key(path)
    assert len(key) >= 43
    assert _mode(path) == 0o600
    assert load_or_create_key(path) == key


def test_access_key_permissions_tightened(tmp_path):
    path = tmp_path / "access-key"
    path.write_text("x" * 43 + "\n", encoding="utf-8")
    path.chmod(0o644)
    assert load_or_create_key(path) == "x" * 43
    assert _mode(path) == 0o600


def test_short_access_key_rejected(tmp_path):
    path = tmp_path / "access-key"
    path.write_text("short\n", encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="too short"):
        load_or_create_key(path)


def test_rotate_key_changes_value(tmp_path):
    path = tmp_path / "access-key"
    first = load_or_create_key(path)
    second = rotate_key(path)
    assert second != first
    assert load_or_create_key(path) == second
    assert _mode(path) == 0o600


def test_keys_match():
    assert keys_match("abc", "abc")
    assert not keys_match("abd", "abc")
    assert not keys_match("", "abc")


def test_session_lifecycle():
    store = SessionStore()
    session = store.create()
    assert store.get(session.token) is session
    assert session.csrf and session.csrf != session.token
    assert store.get(None) is None
    assert store.get("unknown") is None
    store.delete(session.token)
    assert store.get(session.token) is None


def test_session_expires():
    store = SessionStore(ttl_seconds=0)
    session = store.create()
    time.sleep(0.01)
    assert store.get(session.token) is None


def test_settings_reject_missing_engine_python(tmp_path):
    with pytest.raises(ValueError, match="engine Python"):
        WebSettings(data_dir=tmp_path, python=str(tmp_path / "missing-python")).validate()
