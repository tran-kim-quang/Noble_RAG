from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from collections import deque
import re
from difflib import SequenceMatcher
from datetime import datetime

import uvicorn

from client.rag_client import RAGClient
from client.whisper_client import WhisperClient
from input.streaming_voice import iter_voice_segments, list_input_devices, measure_input_level


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _is_blocked_text(text: str, blocked_phrases: list[str]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    return any(phrase in normalized for phrase in blocked_phrases)


def _looks_like_generic_outro(text: str) -> bool:
    normalized = _normalize_text(text)
    patterns = [
        r"\bsubscribe\b",
        r"\bhãy subscribe\b",
        r"\bđăng ký kênh\b",
        r"\bcảm ơn các bạn đã theo dõi\b",
        r"\bhẹn gặp lại\b",
        r"\bkhông bỏ lỡ những video hấp dẫn\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _run_web_ui() -> None:
    host = os.getenv("WEB_UI_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_UI_PORT", "8501"))
    uvicorn.run("ui.rag_web_ui:app", host=host, port=port, reload=False)


def _run_voice_workflow() -> None:
    whisper_url = os.getenv("WHISPER_SERVICE_URL", "http://localhost:8001")
    rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:9621")
    silence_timeout_sec = float(os.getenv("MIC_SILENCE_TIMEOUT_SEC", "3"))
    max_segment_sec = float(os.getenv("MIC_MAX_SEGMENT_SEC", "12"))
    speech_threshold = float(os.getenv("MIC_SPEECH_THRESHOLD", "0.006"))
    mic_device_index_raw = os.getenv("MIC_DEVICE_INDEX", "").strip()
    mic_device_index = int(mic_device_index_raw) if mic_device_index_raw else None
    language = os.getenv("WHISPER_LANGUAGE", "vi").strip() or None
    mic_debug = os.getenv("MIC_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    save_audio = os.getenv("VOICE_SAVE_AUDIO", "0").strip().lower() in {"1", "true", "yes", "on"}
    save_audio_dir = Path(os.getenv("VOICE_AUDIO_DIR", "input/captured_audio")).expanduser()
    repeat_window_sec = float(os.getenv("VOICE_REPEAT_WINDOW_SEC", "45"))
    repeat_limit = int(os.getenv("VOICE_REPEAT_LIMIT", "2"))
    blocked_phrases_raw = os.getenv(
        "VOICE_BLOCKLIST",
        "hãy subscribe cho kênh la la school|cảm ơn các bạn đã theo dõi|hẹn gặp lại",
    )
    blocked_phrases = [
        _normalize_text(item)
        for item in blocked_phrases_raw.split("|")
        if _normalize_text(item)
    ]
    recent_blocked: deque[tuple[float, str]] = deque(maxlen=20)
    recent_texts: deque[tuple[float, str]] = deque(maxlen=20)

    whisper_client = WhisperClient(url=whisper_url)
    rag_client = RAGClient(url=rag_url)

    if mic_device_index is None:
        try:
            devices, _ = list_input_devices()
            pulse_device = next((dev for dev in devices if "pulse" in str(dev["name"]).lower()), None)
            if pulse_device is not None:
                mic_device_index = int(pulse_device["index"])
        except RuntimeError:
            pass

    print(" Voice workflow started")
    print(f"- Whisper: {whisper_url}")
    print(f"- RAG: {rag_url}")
    print(f"- Silence timeout: {silence_timeout_sec}s")
    print(f"- Max segment: {max_segment_sec}s")
    print(f"- Speech threshold: {speech_threshold}")
    print(f"- Mic device: {mic_device_index if mic_device_index is not None else 'auto'}")
    print(f"- Mic debug: {'on' if mic_debug else 'off'}")
    print(f"- Save audio: {'on' if save_audio else 'off'}")
    if save_audio:
        print(f"- Audio dir: {save_audio_dir}")
    print("Nhấn Ctrl+C để dừng.\n")

    if save_audio:
        save_audio_dir.mkdir(parents=True, exist_ok=True)

    if not whisper_client.check_health():
        print("!! Whisper service không healthy. Kiểm tra URL/containers trước khi chạy.")
        return

    if not rag_client.check_health():
        print("!! RAG service không healthy. Kiểm tra URL/containers trước khi chạy.")
        return

    try:
        stats = measure_input_level(device=mic_device_index, seconds=2.0, sample_rate=16000)
        print(f"[mic-probe] rms={stats['rms']:.5f}, peak={stats['peak']:.5f} ({stats['seconds']:.1f}s)")
        if stats["rms"] < 0.002:
            print("[mic-probe] Tín hiệu rất thấp. Hãy kiểm tra mute/permission của microphone.")
    except Exception as exc:
        print(f"[mic-probe] không đo được mức âm: {exc}")

    try:
        for segment in iter_voice_segments(
            silence_timeout_sec=silence_timeout_sec,
            speech_rms_threshold=speech_threshold,
            device=mic_device_index,
            max_segment_sec=max_segment_sec,
            debug=mic_debug,
        ):
            print(f"[segment] captured {segment.duration_sec:.2f}s, gửi Whisper...")
            wav_bytes = segment.to_wav_bytes()

            if save_audio:
                segment_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                tmp_path = save_audio_dir / f"segment_{segment_id}.wav"
                tmp_path.write_bytes(wav_bytes)
                print(f"[segment] saved: {tmp_path}")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(wav_bytes)
                    tmp_path = Path(tmp_file.name)

            try:
                text = whisper_client.transcribe(
                    audio_path=str(tmp_path),
                    language=language,
                    word_timestamps=False,
                )
            finally:
                if not save_audio:
                    tmp_path.unlink(missing_ok=True)

            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            normalized = _normalize_text(text)
            now = time.monotonic()

            while recent_texts and (now - recent_texts[0][0]) > repeat_window_sec:
                recent_texts.popleft()
            while recent_blocked and (now - recent_blocked[0][0]) > repeat_window_sec:
                recent_blocked.popleft()

            if _is_blocked_text(text, blocked_phrases):
                print(f"[filter] Bỏ qua transcript nghi ngờ hallucination: {text}")
                recent_texts.append((now, normalized))
                recent_blocked.append((now, normalized))
                continue

            if _looks_like_generic_outro(text):
                print(f"[filter] Bỏ qua transcript generic/outro: {text}")
                recent_texts.append((now, normalized))
                recent_blocked.append((now, normalized))
                continue

            if recent_blocked:
                max_similarity = max(
                    SequenceMatcher(None, normalized, blocked_text).ratio()
                    for _, blocked_text in recent_blocked
                )
                if max_similarity >= 0.88:
                    print(f"[filter] Bỏ qua transcript gần giống câu bị chặn ({max_similarity:.2f}): {text}")
                    recent_texts.append((now, normalized))
                    recent_blocked.append((now, normalized))
                    continue

            same_count = sum(1 for _, item in recent_texts if item == normalized)
            if same_count >= repeat_limit:
                print(f"[filter] Bỏ qua transcript lặp lại quá nhiều: {text}")
                recent_texts.append((now, normalized))
                continue

            recent_texts.append((now, normalized))

            print(f"User: {text}")

            rag_result = rag_client.query(query_text=text, top_k=10, return_structured_output=False, timeout=90)
            if rag_result is None:
                print("RAG: (không lấy được phản hồi)")
                continue

            response_text = str(rag_result.get("response", "")).strip()
            if response_text:
                print(f"RAG: {response_text}\n")
            else:
                print("RAG: (response rỗng)\n")

    except RuntimeError as exc:
        print(f"!! Không thể mở microphone: {exc}")
    except KeyboardInterrupt:
        print("\nVoice workflow stopped by user")


def _run_voice_list_devices() -> None:
    print("Microphone devices (input):")
    try:
        devices, default_input = list_input_devices()
    except RuntimeError as exc:
        print(f"!! {exc}")
        return

    if default_input is None:
        print("- Default input: none")
    else:
        print(f"- Default input index: {default_input}")

    if not devices:
        print("- Không tìm thấy input device nào trong môi trường hiện tại.")
        print("- Nếu chạy trong WSL/container, cần mapping audio input từ host.")
        return

    for device in devices:
        default_mark = " (default)" if device["is_default"] else ""
        print(
            f"- [{device['index']}] {device['name']}"
            f" | channels={device['max_input_channels']}"
            f" | samplerate={device['default_samplerate']}"
            f"{default_mark}"
        )

    print("\nUse with: MIC_DEVICE_INDEX=<index> APP_MODE=voice poetry run python main.py")


def main() -> None:
    app_mode = os.getenv("APP_MODE", "web").strip().lower()
    if app_mode in {"voice-list-devices", "voice_list_devices", "list-mic", "list_mic"}:
        _run_voice_list_devices()
        return
    if app_mode == "voice":
        _run_voice_workflow()
        return
    _run_web_ui()


if __name__ == "__main__":
    main()