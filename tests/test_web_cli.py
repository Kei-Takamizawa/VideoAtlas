import stat

from videoatlas.web.__main__ import main


def test_print_key_creates_and_prints_key(tmp_path, capsys):
    assert main(["print-key", "--data-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out.strip()
    key_file = tmp_path / "web" / "access-key"
    assert printed == key_file.read_text(encoding="utf-8").strip()
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600


def test_rotate_key_prints_new_key(tmp_path, capsys):
    main(["print-key", "--data-dir", str(tmp_path)])
    first = capsys.readouterr().out.strip()
    assert main(["rotate-key", "--data-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert first not in output
    assert (tmp_path / "web" / "access-key").read_text(encoding="utf-8").strip() in output


def test_serve_rejects_non_loopback_host(tmp_path, capsys, monkeypatch):
    started = []
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: started.append(args))
    assert main(["serve", "--data-dir", str(tmp_path), "--host", "0.0.0.0"]) == 2
    assert started == []
    assert "loopback" in capsys.readouterr().err


def test_serve_starts_uvicorn_on_loopback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    assert main(["serve", "--data-dir", str(tmp_path), "--port", "8799"]) == 0
    assert calls == [{"host": "127.0.0.1", "port": 8799, "proxy_headers": False, "server_header": False,
                      "date_header": False, "access_log": False}]
