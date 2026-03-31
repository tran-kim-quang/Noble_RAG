import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from common.opencv_face_runtime import OpenCVFaceRuntime


def _default_vision_identify_url() -> str:
    explicit = os.getenv("VISION_IDENTIFY_URL")
    if explicit:
        return explicit
    base = os.getenv("VISION_SERVICE_URL", "http://127.0.0.1:8020").rstrip("/")
    return f"{base}/vision/identify"


def _watcher_state_dir() -> Path:
    path = Path(os.getenv("VISION_WATCH_STATE_DIR", str(Path(tempfile.gettempdir()) / "noble_camera_state")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _watcher_state_path(session_id: str) -> Path:
    return _watcher_state_dir() / f"{session_id}.json"


def _load_state(session_id: str) -> Dict[str, Any]:
    path = _watcher_state_path(session_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_state(session_id: str, payload: Dict[str, Any]) -> None:
    _watcher_state_path(session_id).write_text(json.dumps(payload, ensure_ascii=False))


def _persist_identified_face(session_id: str, embedding: list[float], payload: Dict[str, Any]) -> None:
    _save_state(
        session_id,
        {
            "last_embedding": embedding,
            "last_customer_id": payload.get("customer_id"),
            "last_identified_at": time.time(),
            "last_match_score": payload.get("match_score"),
        },
    )


def _similarity_threshold() -> float:
    raw = os.getenv("VISION_FACE_SIMILARITY_THRESHOLD")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return OpenCVFaceRuntime.default_similarity_threshold()


def _stable_face_seconds() -> float:
    return float(os.getenv("VISION_FACE_STABLE_SEC", "1.0"))


def _switch_min_frames() -> int:
    return int(os.getenv("VISION_SWITCH_MIN_FRAMES", "5"))


def _min_face_size() -> float:
    return float(os.getenv("VISION_MIN_FACE_SIZE", "140"))


def _max_faces() -> int:
    return int(os.getenv("VISION_MAX_FACES", "1"))


def _reidentify_cooldown_sec() -> float:
    return float(os.getenv("VISION_REIDENTIFY_COOLDOWN_SEC", "10"))


def _camera_loop_timeout_sec() -> float:
    return float(os.getenv("VISION_CAMERA_LOOP_TIMEOUT_SEC", "20"))


def call_vision_identify(
    image_path: str,
    session_id: str,
    identify_url: Optional[str] = None,
    source: str = "camera",
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    import requests

    url = identify_url or _default_vision_identify_url()
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        data = {"session_id": session_id, "source": source}
        resp = requests.post(url, data=data, files=files, timeout=timeout_sec)

    if resp.status_code >= 400:
        raise RuntimeError(f"Vision identify failed ({resp.status_code}): {resp.text[:500]}")
    return resp.json()


def _write_temp_frame(frame, prefix: str) -> str:
    out_dir = Path(tempfile.gettempdir()) / "noble_camera_capture"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{int(time.time() * 1000)}.jpg"
    ok = cv2.imwrite(str(out_path), frame)
    if not ok:
        raise RuntimeError("failed to write captured frame")
    return str(out_path)


def _return_cached_same_user(session_id: str, state: Dict[str, Any], similarity: float) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "customer_id": state.get("last_customer_id"),
        "is_existing_customer": bool(state.get("last_customer_id")),
        "match_score": similarity,
        "camera_decision": "same_user_skip",
        "skipped_identify": True,
        "customer_context": {},
    }


def capture_and_identify(
    session_id: Optional[str] = None,
    camera_index: int = 0,
    identify_url: Optional[str] = None,
    keep_photo: bool = False,
    tracking_only: bool = False,
) -> Dict[str, Any]:
    sid = session_id or f"cam_{uuid.uuid4().hex[:12]}"
    runtime = OpenCVFaceRuntime()
    similarity_threshold = _similarity_threshold()
    stable_sec = _stable_face_seconds()
    switch_frames = _switch_min_frames()
    min_face_size = _min_face_size()
    max_faces = _max_faces()
    cooldown_sec = _reidentify_cooldown_sec()
    loop_timeout = _camera_loop_timeout_sec()

    state = _load_state(sid)
    ref_embedding = state.get("last_embedding")
    last_identified_at = float(state.get("last_identified_at") or 0.0)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    started_at = time.time()
    low_similarity_frames = 0
    missing_face_frames = 0
    stable_started_at: Optional[float] = None
    stable_best_frame = None
    stable_best_embedding = None
    suspect_user_switch = ref_embedding is None

    try:
        while time.time() - started_at < loop_timeout:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            detected_faces = runtime.detect_faces(frame)
            valid_faces = [
                face
                for face in detected_faces
                if min(face.width, face.height) >= min_face_size
            ]
            if not valid_faces:
                stable_started_at = None
                stable_best_frame = None
                stable_best_embedding = None
                missing_face_frames += 1
                if ref_embedding is not None and missing_face_frames >= switch_frames:
                    suspect_user_switch = True
                continue
            if len(valid_faces) > max_faces:
                stable_started_at = None
                stable_best_frame = None
                stable_best_embedding = None
                low_similarity_frames = 0
                missing_face_frames = 0
                continue

            missing_face_frames = 0
            face = valid_faces[0]
            feature = runtime.extract_feature(frame, face)

            if ref_embedding is not None:
                similarity = runtime.cosine_similarity(feature, runtime.from_list(ref_embedding))
                if not suspect_user_switch and similarity >= similarity_threshold:
                    if time.time() - last_identified_at < cooldown_sec:
                        return _return_cached_same_user(sid, state, similarity)
                    return _return_cached_same_user(sid, state, similarity)
                if similarity < similarity_threshold:
                    low_similarity_frames += 1
                    if low_similarity_frames >= switch_frames:
                        suspect_user_switch = True
                else:
                    low_similarity_frames = 0
            else:
                suspect_user_switch = True

            if not suspect_user_switch:
                continue

            if stable_started_at is None:
                stable_started_at = time.time()
                stable_best_frame = frame.copy()
                stable_best_embedding = feature
            else:
                stable_best_frame = frame.copy()
                stable_best_embedding = feature

            if time.time() - stable_started_at >= stable_sec:
                if tracking_only:
                    return {
                        "session_id": sid,
                        "camera_decision": "tracking_ready",
                        "skipped_identify": True,
                        "valid_face_count": len(valid_faces),
                        "min_face_size": min_face_size,
                        "stable_duration_sec": stable_sec,
                    }
                image_path = _write_temp_frame(stable_best_frame, prefix=f"capture_{sid}")
                try:
                    result = call_vision_identify(image_path=image_path, session_id=sid, identify_url=identify_url)
                finally:
                    if not keep_photo:
                        try:
                            os.remove(image_path)
                        except OSError:
                            pass
                result["camera_decision"] = "identified"
                result["skipped_identify"] = False
                _persist_identified_face(sid, runtime.to_list(stable_best_embedding), result)
                return result

        raise RuntimeError("camera loop timeout before stable face was confirmed")
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-detect stable customer face and call Vision identify endpoint.")
    parser.add_argument("--session-id", default=None, help="Session ID for identify request.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index.")
    parser.add_argument("--identify-url", default=None, help="Override identify endpoint URL.")
    parser.add_argument("--keep-photo", action="store_true", help="Keep captured image file in /tmp.")
    parser.add_argument("--tracking-only", action="store_true", help="Stop after tracking confirms one close face, without calling Vision identify.")
    args = parser.parse_args()

    output = capture_and_identify(
        session_id=args.session_id,
        camera_index=args.camera_index,
        identify_url=args.identify_url,
        keep_photo=args.keep_photo,
        tracking_only=args.tracking_only,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
