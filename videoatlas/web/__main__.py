"""Run the browser GUI server.

    python -m videoatlas.web serve [--data-dir DIR] [--port 8765]
    python -m videoatlas.web print-key [--data-dir DIR]
    python -m videoatlas.web rotate-key [--data-dir DIR]

The server listens on a loopback address only. Reach it from another computer
through an SSH tunnel, for example ``ssh -N -L 8765:127.0.0.1:8765 videoatlas.incus``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .security import load_or_create_key, rotate_key
from .settings import WebSettings, default_data_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m videoatlas.web", description="VideoAtlas browser GUI.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=None, help="data folder (default: $VIDEOATLAS_DATA_DIR or ~/videoatlas-data)")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", parents=[common], help="start the GUI server on a loopback address")
    serve.add_argument("--host", default="127.0.0.1", help="loopback address to listen on (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    serve.add_argument("--engine-python", type=Path, default=None,
                       help="Python that runs analysis jobs (default: this Python). Lets the GUI use its own environment.")
    serve.add_argument("--engine-root", type=Path, default=None,
                       help="repository folder whose videoatlas.cli runs the jobs (default: this repository)")
    commands.add_parser("print-key", parents=[common], help="print the access key, creating it when missing")
    commands.add_parser("rotate-key", parents=[common], help="replace the access key; restart the server afterwards")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = (args.data_dir or default_data_dir()).expanduser().resolve()
    if args.command == "serve":
        engine = {}
        if args.engine_python is not None:
            engine["python"] = str(args.engine_python.expanduser().absolute())
        if args.engine_root is not None:
            engine["repo_root"] = args.engine_root.expanduser().resolve()
        settings = WebSettings(data_dir=data_dir, host=args.host, port=args.port, **engine)
        try:
            settings.validate()
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        import uvicorn

        from .app import create_app
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, proxy_headers=False,
                    server_header=False, date_header=False, access_log=False)
        return 0
    key_file = WebSettings(data_dir=data_dir).key_file
    if args.command == "print-key":
        print(load_or_create_key(key_file))
        return 0
    print(rotate_key(key_file))
    print("Restart the GUI server: it keeps accepting the old key and its sessions until it restarts.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
