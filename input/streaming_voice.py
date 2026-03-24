from __future__ import annotations

import io
import queue
import time
import wave
from collections import deque
from dataclasses import dataclass
from typing import Generator

import numpy as np

try:
	import sounddevice as sd
except OSError:
	sd = None


@dataclass(slots=True)
class VoiceSegment:
	pcm_bytes: bytes
	sample_rate: int
	channels: int
	sample_width: int = 2

	@property
	def duration_sec(self) -> float:
		frame_size = self.channels * self.sample_width
		if frame_size <= 0:
			return 0.0
		total_frames = len(self.pcm_bytes) / frame_size
		return total_frames / float(self.sample_rate)

	def to_wav_bytes(self) -> bytes:
		wav_buffer = io.BytesIO()
		with wave.open(wav_buffer, "wb") as wav_file:
			wav_file.setnchannels(self.channels)
			wav_file.setsampwidth(self.sample_width)
			wav_file.setframerate(self.sample_rate)
			wav_file.writeframes(self.pcm_bytes)
		return wav_buffer.getvalue()


class ContinuousMicListener:
	def __init__(
		self,
		sample_rate: int = 16000,
		channels: int = 1,
		frame_ms: int = 30,
		silence_timeout_sec: float = 3.0,
		min_speech_sec: float = 0.25,
		pre_speech_sec: float = 0.35,
		speech_rms_threshold: float = 0.012,
		adaptive_multiplier: float = 1.15,
		adaptive_offset: float = 0.002,
		max_segment_sec: float = 12.0,
		device: int | None = None,
		debug: bool = False,
	) -> None:
		self.sample_rate = sample_rate
		self.channels = channels
		self.frame_ms = frame_ms
		self.silence_timeout_sec = silence_timeout_sec
		self.min_speech_sec = min_speech_sec
		self.pre_speech_sec = pre_speech_sec
		self.speech_rms_threshold = speech_rms_threshold
		self.adaptive_multiplier = adaptive_multiplier
		self.adaptive_offset = adaptive_offset
		self.max_segment_sec = max_segment_sec
		self.device = device
		self.debug = debug

		self._frame_size = int(self.sample_rate * self.frame_ms / 1000)
		if self._frame_size <= 0:
			raise ValueError("frame_ms creates invalid frame size")

	def listen(self) -> Generator[VoiceSegment, None, None]:
		if sd is None:
			raise RuntimeError(
				"sounddevice is installed but PortAudio system library is missing. "
				"Install it on host (e.g. sudo apt-get install portaudio19-dev)."
			)

		resolved_device = self._resolve_input_device()

		audio_queue: queue.Queue[bytes] = queue.Queue()
		pre_frames = max(1, int(self.pre_speech_sec * 1000 / self.frame_ms))
		pre_buffer: deque[bytes] = deque(maxlen=pre_frames)

		in_segment = False
		segment_frames: list[bytes] = []
		segment_voice_frames = 0
		segment_started_at = 0.0
		last_voice_at = 0.0
		noise_rms = 0.0
		last_debug_at = 0.0

		def callback(indata: bytes, frames: int, _time_info: dict, status: sd.CallbackFlags) -> None:
			if status:
				return
			if frames > 0:
				audio_queue.put(bytes(indata))

		try:
			with sd.RawInputStream(
				samplerate=self.sample_rate,
				channels=self.channels,
				dtype="int16",
				blocksize=self._frame_size,
				callback=callback,
				device=resolved_device,
			):
				while True:
					frame = audio_queue.get()
					rms = self._rms_from_int16(frame)
					now = time.monotonic()

					if not in_segment:
						if noise_rms == 0.0:
							noise_rms = rms
						else:
							noise_rms = (0.98 * noise_rms) + (0.02 * rms)

					dynamic_threshold = max(
						self.speech_rms_threshold,
						(noise_rms * self.adaptive_multiplier) + self.adaptive_offset,
					)
					has_voice = rms >= dynamic_threshold

					if self.debug and (now - last_debug_at) >= 1.0:
						print(
							f"[mic-debug] rms={rms:.5f} noise={noise_rms:.5f} "
							f"threshold={dynamic_threshold:.5f} voice={has_voice}"
						)
						last_debug_at = now

					if has_voice:
						last_voice_at = now
						segment_voice_frames += 1

					if not in_segment:
						pre_buffer.append(frame)
						if has_voice:
							in_segment = True
							segment_frames = list(pre_buffer)
							segment_started_at = now
					else:
						segment_frames.append(frame)
						silence_elapsed = now - last_voice_at
						segment_elapsed = now - segment_started_at if segment_started_at > 0 else 0.0
						should_flush_by_silence = silence_elapsed >= self.silence_timeout_sec
						should_flush_by_length = self.max_segment_sec > 0 and segment_elapsed >= self.max_segment_sec

						if should_flush_by_silence or should_flush_by_length:
							trim_frames = max(0, int(self.silence_timeout_sec * 1000 / self.frame_ms))
							if should_flush_by_silence and trim_frames > 0 and len(segment_frames) > trim_frames:
								effective_frames = segment_frames[:-trim_frames]
							else:
								effective_frames = segment_frames

							segment = self._finalize_segment(effective_frames, segment_voice_frames)
							in_segment = False
							segment_frames = []
							segment_voice_frames = 0
							segment_started_at = 0.0
							pre_buffer.clear()
							if segment is not None:
								yield segment
		except Exception as exc:
			raise RuntimeError(
				f"Không thể mở input device {resolved_device}. "
				"Hãy thử đặt MIC_DEVICE_INDEX hoặc kiểm tra quyền truy cập microphone. "
				f"Chi tiết: {exc}"
			) from exc

	def _resolve_input_device(self) -> int | None:
		if self.device is not None:
			return self.device

		try:
			default_devices = sd.default.device
			if isinstance(default_devices, (tuple, list)) and len(default_devices) >= 1:
				default_input = int(default_devices[0])
				if default_input >= 0:
					return default_input
		except Exception:
			pass

		try:
			devices = sd.query_devices()
		except Exception as exc:
			raise RuntimeError(f"Không thể đọc danh sách microphone từ PortAudio: {exc}") from exc

		for index, device_info in enumerate(devices):
			if int(device_info.get("max_input_channels", 0)) > 0:
				return index

		raise RuntimeError(
			"Không tìm thấy microphone input nào. "
			"Nếu chạy trong container/WSL, cần mapping thiết bị âm thanh từ host."
		)

	def _finalize_segment(self, frames: list[bytes], voice_frames: int) -> VoiceSegment | None:
		if not frames:
			return None

		total_duration = len(frames) * self.frame_ms / 1000
		voice_duration = voice_frames * self.frame_ms / 1000

		if total_duration < self.min_speech_sec and voice_duration < self.min_speech_sec:
			return None

		return VoiceSegment(
			pcm_bytes=b"".join(frames),
			sample_rate=self.sample_rate,
			channels=self.channels,
		)

	@staticmethod
	def _rms_from_int16(frame: bytes) -> float:
		if not frame:
			return 0.0
		pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
		if pcm.size == 0:
			return 0.0
		return float(np.sqrt(np.mean((pcm / 32768.0) ** 2)))


def iter_voice_segments(
	silence_timeout_sec: float = 3.0,
	speech_rms_threshold: float = 0.012,
	device: int | None = None,
	max_segment_sec: float = 12.0,
	debug: bool = False,
) -> Generator[VoiceSegment, None, None]:
	listener = ContinuousMicListener(
		silence_timeout_sec=silence_timeout_sec,
		speech_rms_threshold=speech_rms_threshold,
		device=device,
		max_segment_sec=max_segment_sec,
		debug=debug,
	)
	yield from listener.listen()


def measure_input_level(
	device: int | None = None,
	seconds: float = 2.0,
	sample_rate: int = 16000,
) -> dict[str, float]:
	if sd is None:
		raise RuntimeError(
			"sounddevice is installed but PortAudio system library is missing. "
			"Install it on host (e.g. sudo apt-get install portaudio19-dev)."
		)

	frames = max(1, int(sample_rate * seconds))
	recorded = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=device)
	sd.wait()
	arr = recorded[:, 0]
	rms = float(np.sqrt(np.mean(arr**2)))
	peak = float(np.max(np.abs(arr)))
	return {
		"rms": rms,
		"peak": peak,
		"seconds": float(seconds),
	}


def list_input_devices() -> tuple[list[dict[str, str | int | bool]], int | None]:
	if sd is None:
		raise RuntimeError(
			"sounddevice is installed but PortAudio system library is missing. "
			"Install it on host (e.g. sudo apt-get install portaudio19-dev)."
		)

	try:
		devices = sd.query_devices()
	except Exception as exc:
		raise RuntimeError(f"Không thể lấy danh sách audio devices: {exc}") from exc

	default_input: int | None = None
	try:
		default_devices = sd.default.device
		if isinstance(default_devices, (tuple, list)) and len(default_devices) >= 1:
			candidate = int(default_devices[0])
			if candidate >= 0:
				default_input = candidate
	except Exception:
		pass

	input_devices: list[dict[str, str | int | bool]] = []
	for index, device_info in enumerate(devices):
		max_input_channels = int(device_info.get("max_input_channels", 0))
		if max_input_channels <= 0:
			continue
		input_devices.append(
			{
				"index": index,
				"name": str(device_info.get("name", "unknown")),
				"max_input_channels": max_input_channels,
				"default_samplerate": int(float(device_info.get("default_samplerate", 0))),
				"is_default": index == default_input,
			}
		)

	return input_devices, default_input
