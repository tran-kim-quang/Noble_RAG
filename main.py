# import sys

# from client.whisper_client import WhisperClient


# client = WhisperClient(url="http://localhost:8001")

# ref_wav_path = "test/ref.wav"

# # 3. Gọi hàm nhận diện
# text = client.transcribe(str(ref_wav_path))

# if text:
#     print(f"Văn bản: {text}")


from client.rag_client import RAGClient

rag_client = RAGClient(url="http://localhost:8000")
rag_client.upload_document()