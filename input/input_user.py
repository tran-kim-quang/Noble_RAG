import sys
from pathlib import Path

# Thêm thư mục gốc của project vào sys.path
root_path = Path(__file__).parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from client.whisper_client import WhisperClient

# 1. Khởi tạo client
client = WhisperClient(url="http://localhost:8001")

# 2. Lấy đường dẫn file âm thanh (nằm ở thư mục Noble_RAG/test/ref.wav)
current_dir = Path(__file__).parent
ref_wav_path = current_dir.parent / "test" / "ref.wav"

# 3. Gọi hàm nhận diện
text = client.transcribe(str(ref_wav_path))

if text:
    print(f"Văn bản: {text}")
