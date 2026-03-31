import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from action.camera import (
    _load_state,
    _max_faces,
    _min_face_size,
    _reidentify_cooldown_sec,
    _similarity_threshold,
    _stable_face_seconds,
    _switch_min_frames,
    _write_temp_frame,
    call_vision_identify,
)
from common.opencv_face_runtime import OpenCVFaceRuntime


def _default_session_file() -> Path:
    return Path(tempfile.gettempdir()) / "noble_active_session.json"


def _camera_loop_sleep() -> float:
    return 0.02


def _write_session_control(session_file: Path, session_id: str, enabled: bool) -> None:
    payload = {
        "session_id": session_id.strip(),
        "enabled": bool(enabled),
        "updated_at": time.time(),
    }
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_active_session(session_id: Optional[str], session_file: Optional[Path]) -> Optional[str]:
    if session_id:
        return session_id.strip() or None
    if not session_file or not session_file.exists():
        return None
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("enabled") is False:
        return None
    value = str(data.get("session_id") or "").strip()
    return value or None


def _start_control_server(bind_host: str, port: int, session_file: Path) -> ThreadingHTTPServer:
    class WatcherControlHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send_json(404, {"ok": False, "detail": "not found"})
                return
            self._send_json(200, {"ok": True, "session_file": str(session_file)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/session":
                self._send_json(404, {"ok": False, "detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"ok": False, "detail": "invalid json"})
                return

            enabled = bool(data.get("enabled", True))
            session_id = str(data.get("session_id") or "").strip()
            if enabled and not session_id:
                self._send_json(400, {"ok": False, "detail": "session_id cannot be empty when enabled=true"})
                return

            _write_session_control(session_file, session_id, enabled)
            self._send_json(
                200,
                {
                    "ok": True,
                    "session_id": session_id,
                    "enabled": enabled,
                    "session_file": str(session_file),
                },
            )

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((bind_host, port), WatcherControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _open_capture(camera_source: Optional[str], camera_index: int):
    if camera_source:
        source = int(camera_source) if camera_source.isdigit() else camera_source
    else:
        source = camera_index
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    return cap


def _draw_preview(frame, faces, status_text: str) -> None:
    preview = frame.copy()
    for face in faces:
        x, y, w, h = [int(v) for v in face.raw[:4]]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (30, 220, 30), 2)
    cv2.putText(
        preview,
        status_text[:100],
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 240),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Noble Camera Watcher", preview)


def run_watcher(
    *,
    camera_index: int,
    camera_source: Optional[str],
    identify_url: str,
    session_id: Optional[str],
    session_file: Optional[Path],
    show_preview: bool,
    keep_photo: bool,
) -> None:
    runtime = OpenCVFaceRuntime()
    similarity_threshold = _similarity_threshold()
    stable_sec = _stable_face_seconds()
    switch_frames = _switch_min_frames()
    min_face_size = _min_face_size()
    max_faces = _max_faces()
    cooldown_sec = _reidentify_cooldown_sec()

    cap = _open_capture(camera_source, camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera source")

    current_session_id: Optional[str] = None
    ref_embedding = None
    last_identified_at = 0.0
    low_similarity_frames = 0
    missing_face_frames = 0
    stable_started_at: Optional[float] = None
    stable_best_frame = None
    stable_best_embedding = None
    suspect_user_switch = True
    status_text = "waiting session"

    try:
        while True:
            next_session_id = _read_active_session(session_id, session_file)
            if next_session_id != current_session_id:
                current_session_id = next_session_id
                low_similarity_frames = 0
                missing_face_frames = 0
                stable_started_at = None
                stable_best_frame = None
                stable_best_embedding = None
                suspect_user_switch = True
                if current_session_id:
                    state = _load_state(current_session_id)
                    ref_embedding = state.get("last_embedding")
                    last_identified_at = float(state.get("last_identified_at") or 0.0)
                    status_text = f"session={current_session_id}"
                else:
                    ref_embedding = None
                    last_identified_at = 0.0
                    status_text = "waiting session"

            ok, frame = cap.read()
            if not ok:
                time.sleep(_camera_loop_sleep())
                continue

            if not current_session_id:
                if show_preview:
                    _draw_preview(frame, [], status_text)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(_camera_loop_sleep())
                continue

            detected_faces = runtime.detect_faces(frame)
            valid_faces = [
                face
                for face in detected_faces
                if min(face.width, face.height) >= min_face_size
            ]
            if len(valid_faces) != 1:
                stable_started_at = None
                stable_best_frame = None
                stable_best_embedding = None
                low_similarity_frames = 0
                if len(valid_faces) == 0:
                    missing_face_frames += 1
                    status_text = f"{current_session_id}: no close face"
                    if ref_embedding is not None and missing_face_frames >= switch_frames:
                        suspect_user_switch = True
                else:
                    missing_face_frames = 0
                    status_text = f"{current_session_id}: multiple faces"
                if show_preview:
                    _draw_preview(frame, valid_faces, status_text)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                time.sleep(_camera_loop_sleep())
                continue

            missing_face_frames = 0
            face = valid_faces[0]
            feature = runtime.extract_feature(frame, face)

            if ref_embedding is not None:
                similarity = runtime.cosine_similarity(feature, runtime.from_list(ref_embedding))
                if not suspect_user_switch and similarity >= similarity_threshold:
                    status_text = f"{current_session_id}: same user {similarity:.3f}"
                    low_similarity_frames = 0
                    stable_started_at = None
                    stable_best_frame = None
                    stable_best_embedding = None
                elif similarity < similarity_threshold:
                    low_similarity_frames += 1
                    status_text = f"{current_session_id}: suspect switch {similarity:.3f}"
                    if low_similarity_frames >= switch_frames:
                        suspect_user_switch = True
                else:
                    low_similarity_frames = 0
            else:
                suspect_user_switch = True
                status_text = f"{current_session_id}: new face candidate"

            if suspect_user_switch:
                if stable_started_at is None:
                    stable_started_at = time.time()
                    stable_best_frame = frame.copy()
                    stable_best_embedding = feature
                else:
                    stable_best_frame = frame.copy()
                    stable_best_embedding = feature
                status_text = f"{current_session_id}: stabilizing"

            should_identify = (
                suspect_user_switch
                and stable_started_at is not None
                and stable_best_frame is not None
                and stable_best_embedding is not None
                and (time.time() - stable_started_at) >= stable_sec
                and (time.time() - last_identified_at) >= cooldown_sec
            )
            if should_identify:
                image_path = _write_temp_frame(stable_best_frame, prefix=f"watch_{current_session_id}")
                try:
                    result = call_vision_identify(
                        image_path=image_path,
                        session_id=current_session_id,
                        identify_url=identify_url,
                        source="camera_windows",
                    )
                finally:
                    if not keep_photo:
                        try:
                            os.remove(image_path)
                        except OSError:
                            pass
                ref_embedding = runtime.to_list(stable_best_embedding)
                last_identified_at = time.time()
                stable_started_at = None
                stable_best_frame = None
                stable_best_embedding = None
                low_similarity_frames = 0
                suspect_user_switch = False
                status_text = (
                    f"{current_session_id}: identified "
                    f"{result.get('customer_id', '')} existing={result.get('is_existing_customer')}"
                )
                print(json.dumps(result, ensure_ascii=False))

            if show_preview:
                _draw_preview(frame, valid_faces, status_text)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(_camera_loop_sleep())
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-running Windows camera watcher that tracks one close face and sends keyframes to Noble Vision."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Local webcam index.")
    parser.add_argument("--camera-source", default=None, help="Optional camera URL/file path instead of index.")
    parser.add_argument("--identify-url", required=True, help="Vision identify URL, e.g. http://localhost:8020/vision/identify")
    parser.add_argument("--session-id", default=None, help="Fixed session id. If omitted, use --session-file.")
    parser.add_argument(
        "--session-file",
        default=str(_default_session_file()),
        help="JSON file with {\"session_id\": \"...\", \"enabled\": true}.",
    )
    parser.add_argument("--control-host", default="127.0.0.1", help="Local watcher control bind host.")
    parser.add_argument("--control-port", type=int, default=8765, help="Local watcher control port.")
    parser.add_argument("--show-preview", action="store_true", help="Show preview window with tracking status.")
    parser.add_argument("--keep-photo", action="store_true", help="Keep captured keyframes in temp directory.")
    args = parser.parse_args()

    session_file = Path(args.session_file) if args.session_file else _default_session_file()
    control_server = _start_control_server(args.control_host, args.control_port, session_file)

    try:
        run_watcher(
            camera_index=args.camera_index,
            camera_source=args.camera_source,
            identify_url=args.identify_url,
            session_id=args.session_id,
            session_file=session_file,
            show_preview=args.show_preview,
            keep_photo=args.keep_photo,
        )
    finally:
        control_server.shutdown()
        control_server.server_close()


if __name__ == "__main__":
    main()
