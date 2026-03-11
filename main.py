import sys

from client.whisper_client import WhisperClient


client = WhisperClient(url="http://localhost:8001")

ref_wav_path = "test/ref.wav"

# 3. Gọi hàm nhận diện
text = client.transcribe(str(ref_wav_path))

if text:
    print(f"Văn bản: {text}")
