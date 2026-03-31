import argparse
import json
import tempfile
import time
from pathlib import Path


def _default_session_file() -> Path:
    return Path(tempfile.gettempdir()) / "noble_active_session.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the active session file used by the Windows camera watcher.")
    parser.add_argument("--session-id", default="", help="Active session id. Leave empty with --disable to stop watcher dispatch.")
    parser.add_argument(
        "--session-file",
        default=str(_default_session_file()),
        help="Path to the watcher control JSON file.",
    )
    parser.add_argument("--disable", action="store_true", help="Disable camera dispatch without deleting the session file.")
    args = parser.parse_args()

    payload = {
        "session_id": "" if args.disable else args.session_id.strip(),
        "enabled": not args.disable,
        "updated_at": time.time(),
    }
    path = Path(args.session_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
