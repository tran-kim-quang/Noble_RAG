import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _default_vision_identify_url() -> str:
    explicit = os.getenv("VISION_IDENTIFY_URL")
    if explicit:
        return explicit
    base = os.getenv("VISION_SERVICE_URL", "http://127.0.0.1:8020").rstrip("/")
    return f"{base}/vision/identify"


def capture_photo(camera_index: int = 0) -> str:
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("Missing dependency: opencv-python") from e

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    window = "Noble Camera - SPACE: capture | Q/ESC: quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    captured_path: Optional[str] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from camera")

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:  # SPACE
                out_dir = Path(tempfile.gettempdir()) / "noble_camera_capture"
                out_dir.mkdir(parents=True, exist_ok=True)
                captured_path = str(out_dir / f"capture_{int(time.time())}.jpg")
                cv2.imwrite(captured_path, frame)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not captured_path:
        raise RuntimeError("Capture cancelled by user")
    return captured_path


def call_vision_identify(
    image_path: str,
    session_id: str,
    identify_url: Optional[str] = None,
    source: str = "camera",
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("Missing dependency: requests") from e

    url = identify_url or _default_vision_identify_url()
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        data = {"session_id": session_id, "source": source}
        resp = requests.post(url, data=data, files=files, timeout=timeout_sec)

    if resp.status_code >= 400:
        raise RuntimeError(f"Vision identify failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


def capture_and_identify(
    session_id: Optional[str] = None,
    camera_index: int = 0,
    identify_url: Optional[str] = None,
    keep_photo: bool = False,
) -> Dict[str, Any]:
    sid = session_id or f"cam_{uuid.uuid4().hex[:12]}"
    image_path = capture_photo(camera_index=camera_index)
    try:
        result = call_vision_identify(image_path=image_path, session_id=sid, identify_url=identify_url)
    finally:
        if not keep_photo:
            try:
                os.remove(image_path)
            except OSError:
                pass

    return {
        "session_id": result.get("session_id", sid),
        "customer_id": result.get("customer_id"),
        "is_existing_customer": result.get("is_existing_customer"),
        "gender_estimate": result.get("gender_estimate", "unknown"),
        "age_group_estimate": result.get("age_group_estimate", "unknown"),
        "match_score": result.get("match_score"),
        "customer_context": result.get("customer_context", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture user photo and call Vision identify endpoint.")
    parser.add_argument("--session-id", default=None, help="Session ID for identify request.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index.")
    parser.add_argument("--identify-url", default=None, help="Override identify endpoint URL.")
    parser.add_argument("--keep-photo", action="store_true", help="Keep captured image file in /tmp.")
    args = parser.parse_args()

    output = capture_and_identify(
        session_id=args.session_id,
        camera_index=args.camera_index,
        identify_url=args.identify_url,
        keep_photo=args.keep_photo,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
