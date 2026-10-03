import sys

if sys.platform != "win32":
    raise SystemExit("VideoAtlas requires Windows 10 or 11.")

from .ui import main
if __name__ == "__main__":
    raise SystemExit(main())
