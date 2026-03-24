"""
RAG Client Library
Giao diện HTTP cho LightRAG Service
"""

import requests
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum


class StorageType(str, Enum):
    """Storage type options for document upload"""
    GRAPH = "graph"
    VECTOR = "vector"
    BOTH = "both"


class RAGClient:
    """
    Client library để kết nối với RAG Service
    
    Tương tự whisper_client.py, cung cấp interface để:
    - Upload document vào storage (graph hoặc vector)
    - Query RAG system với LLM response
    - Check service health
    """
    
    def __init__(self, url: str = "http://localhost:8001"):
        """
        Khởi tạo RAG Client
        
        Args:
            url: Base URL của RAG Service (default: http://localhost:8001)
        """
        self.url = url.rstrip("/")
        self.upload_url = f"{self.url}/upload-document"
        self.query_url = f"{self.url}/query"
        self.health_url = f"{self.url}/health"
        self.models_url = f"{self.url}/models"
        self.status_url = f"{self.url}/status"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("RAGClient")
    
    def upload_document(
        self,
        file_path: str,
        storage_type: StorageType = StorageType.GRAPH,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        upload_filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Upload document vào RAG system
        
        Args:
            file_path: Đường dẫn tới file document (txt, pdf, etc.)
            storage_type: Loại storage - "graph" (default), "vector", hoặc "both"
            metadata: Metadata liên kết với document (dict)
            timeout: Timeout cho request (seconds)
        
        Returns:
            Dictionary chứa {status, document_id, chunks_count, storage_type, message}
            hoặc None nếu lỗi
        
        Example:
            client = RAGClient()
            result = client.upload_document(
                "document.txt",
                storage_type=StorageType.GRAPH,
                metadata={"source": "user_upload", "language": "en"}
            )
            if result:
                print(f"Document ID: {result['document_id']}")
        """
        
        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"File không tồn tại: {file_path}")
            return None
        
        if not path.is_file():
            self.logger.error(f"Đây không phải file: {file_path}")
            return None
        
        try:
            # Chuẩn bị form data
            effective_filename = (upload_filename or path.name).strip() or path.name
            files = {
                "file": (effective_filename, open(path, "rb"), "text/plain")
            }
            
            data = {
                "storage_type": storage_type.value
            }
            
            # Thêm metadata nếu có
            if metadata:
                data["metadata"] = json.dumps(metadata)
            
            self.logger.info(
                f"Uploading document '{effective_filename}' "
                f"storage_type={storage_type.value} size={path.stat().st_size} bytes"
            )
            
            response = requests.post(
                self.upload_url,
                files=files,
                data=data,
                timeout=timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            self.logger.info(
                f"Document uploaded successfully: "
                f"document_id={result.get('document_id')} "
                f"chunks={result.get('chunks_count')}"
            )
            
            return result
        
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout sau {timeout}s")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        return_structured_output: bool = False,
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Query RAG system với LLM response
        
        Args:
            query_text: Câu hỏi / query
            top_k: Số context items tối đa trong response
            return_structured_output: Trả về structured data
            timeout: Timeout cho request (seconds)
        
        Returns:
            Dictionary chứa {response, context, tokens_used, model}
            hoặc None nếu lỗi
        
        Example:
            client = RAGClient()
            result = client.query("What is the main topic of the document?")
            if result:
                print(f"LLM Response: {result['response']}")
                print(f"Context items: {len(result['context'])}")
        """
        
        if not query_text or not query_text.strip():
            self.logger.error("Query không được rỗng")
            return None
        
        try:
            payload = {
                "query": query_text,
                "top_k": top_k,
                "return_structured_output": return_structured_output
            }
            
            self.logger.info(f"Querying: {query_text[:100]}")
            
            response = requests.post(
                self.query_url,
                json=payload,
                timeout=timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            
            self.logger.info(
                f"Query processed: "
                f"model={result.get('model')} "
                f"context_items={len(result.get('context', []))}"
            )
            
            return result
        
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timeout sau {timeout}s")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def check_health(self) -> bool:
        """
        Kiểm tra xem RAG Service có đang chạy không
        
        Returns:
            True nếu service healthy, False nếu lỗi
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            is_healthy = response.status_code == 200
            
            if is_healthy:
                health_data = response.json()
                self.logger.info(
                    f"Service healthy: llm_provider={health_data.get('llm_provider')} "
                    f"embedding={health_data.get('embedding_model')}"
                )
            
            return is_healthy
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False
    
    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Lấy status hiện tại của RAG Service
        
        Returns:
            Dictionary chứa service status hoặc None
        """
        try:
            response = requests.get(self.status_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return None
    
    def get_models_info(self) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin về LLM models và embedding
        
        Returns:
            Dictionary chứa model info hoặc None
        """
        try:
            response = requests.get(self.models_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get models info: {e}")
            return None


# ── Ví dụ sử dụng ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Khởi tạo client
    client = RAGClient("http://localhost:8001")
    
    # Kiểm tra service
    if not client.check_health():
        print("RAG Service không khả dụng!")
        exit(1)
    
    # Lấy model info
    models_info = client.get_models_info()
    if models_info:
        print(f"LLM Model: {models_info['llm']['model']}")
        print(f"Embedding Model: {models_info['embedding']['model']}")
    
    # Upload document
    document_file = str(Path(__file__).parent.parent / "test" / "sample.txt")
    if Path(document_file).exists():
        result = client.upload_document(
            document_file,
            storage_type=StorageType.GRAPH,
            metadata={
                "source": "example",
                "language": "en"
            }
        )
        
        if result:
            print(f"\nDocument uploaded:")
            print(f"  ID: {result['document_id']}")
            print(f"  Chunks: {result['chunks_count']}")
            print(f"  Storage: {result['storage_type']}")
        
        # Query
        query_result = client.query("What is this document about?")
        if query_result:
            print(f"\nQuery result:")
            print(f"  Response: {query_result['response']}")
            print(f"  Context items: {len(query_result['context'])}")
            print(f"  Model: {query_result['model']}")
    else:
        print(f"Test file not found: {document_file}")

