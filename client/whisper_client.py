import requests
import logging
from pathlib import Path
from typing import Optional, Dict, Any

class WhisperClient:
    def __init__(self, url: str = "http://localhost:8001"):
        self.url = url.rstrip("/")
        self.transcribe_url = f"{self.url}/transcribe"
        self.health_url = f"{self.url}/health"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("WhisperClient")

    def transcribe(
        self, 
        audio_path: str, 
        language: Optional[str] = None, 
        word_timestamps: bool = False
    ) -> Optional[str]:
        """
        Gửi file âm thanh tới Whisper Service và nhận kết quả là chuỗi văn bản.
        
        :param audio_path: Đường dẫn tới file âm thanh (wav, mp3, mp4...)
        :param language: Mã ngôn ngữ (ví dụ: 'vi', 'en'). Nếu None sẽ tự nhận diện.
        :param word_timestamps: Nếu True sẽ yêu cầu mốc thời gian từng từ.
        :return: Chuỗi văn bản đã được nhận diện hoặc None nếu lỗi.
        """
        path = Path(audio_path)
        if not path.exists():
            self.logger.error(f"File không tồn tại: {audio_path}")
            return None

        # Chuẩn bị dữ liệu gửi đi (Multipart Form Data)
        data = {
            "word_timestamps": str(word_timestamps).lower()
        }
        if language:
            data["language"] = language

        try:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, "audio/wav")} # Tên file, handler, mime-type
                
                response = requests.post(
                    self.transcribe_url, 
                    files=files, 
                    data=data,
                    timeout=300 # Chờ tối đa 5 phút cho file dài
                )
                
            response.raise_for_status() # Kiểm tra lỗi HTTP (4xx, 5xx)
            result = response.json()
            
            # Ghép các đoạn segment thành 1 chuỗi văn bản hoàn chỉnh
            full_text = " ".join([seg["text"] for seg in result.get("segments", [])]).strip()
            return full_text

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Lỗi khi gọi API: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi không xác định: {e}")
            return None

    def check_health(self) -> bool:
        """Kiểm tra xem service có đang chạy không."""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    client = WhisperClient("http://localhost:8001")
    
    # Giả sử file test nằm ở thư mục cha / test / ref.wav
    audio_file = str(Path(__file__).parent.parent / "test" / "ref.wav")
    
    print(f"--- Đang nhận diện file: {audio_file} ---")
    text = client.transcribe(audio_file, language="vi")
    
    if text:
        print(f"Kết quả: {text}")
    else:
        print("Không nhận diện được hoặc có lỗi xảy ra.")
