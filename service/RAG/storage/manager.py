"""
Storage Utilities - Helper functions để làm việc với PostgreSQL, Qdrant, Redis
"""

import json
import redis
from typing import Optional, Dict, Any, List
from qdrant_client import QdrantClient
import psycopg2
from psycopg2.extras import RealDictCursor

from config import get_rag_config


class StorageManager:
    """
    Unified interface để quản lý 3 loại storage
    """
    
    def __init__(self):
        """Initialize connection tới tất cả storage backends"""
        self.rag_config = get_rag_config()
        self._pg_conn = None
        self._qdrant_client = None
        self._redis_client = None
    
    # ==================== PostgreSQL ====================
    
    @property
    def pg_connection(self):
        """Lazy-loaded PostgreSQL connection"""
        if self._pg_conn is None:
            self._pg_conn = psycopg2.connect(
                host=self.rag_config.postgres_host,
                port=self.rag_config.postgres_port,
                user=self.rag_config.postgres_user,
                password=self.rag_config.postgres_password,
                database=self.rag_config.postgres_db
            )
        return self._pg_conn
    
    def query_postgres(self, sql: str, params: tuple = None) -> List[Dict]:
        """
        Execute SELECT query trên PostgreSQL
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            List of results as dicts
        """
        with self.pg_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()
    
    def insert_postgres(self, sql: str, params: tuple = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE trên PostgreSQL
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            with self.pg_connection.cursor() as cur:
                cur.execute(sql, params or ())
                self.pg_connection.commit()
                return cur.rowcount
        except Exception as e:
            self.pg_connection.rollback()
            raise e
    
    def get_chunks(self, workspace: str, limit: int = 100) -> List[Dict]:
        """Get all chunks từ PostgreSQL"""
        return self.query_postgres(
            """
            SELECT * FROM chunks 
            WHERE workspace = %s
            LIMIT %s
            """,
            (workspace, limit)
        )
    
    def get_entities(self, workspace: str) -> List[Dict]:
        """Get all entities từ PostgreSQL"""
        return self.query_postgres(
            """
            SELECT * FROM entities 
            WHERE workspace = %s
            """,
            (workspace,)
        )
    
    def get_relations(self, workspace: str) -> List[Dict]:
        """Get all relations từ PostgreSQL"""
        return self.query_postgres(
            """
            SELECT * FROM relations 
            WHERE workspace = %s
            """,
            (workspace,)
        )
    
    def close_postgres(self):
        """Close PostgreSQL connection"""
        if self._pg_conn:
            self._pg_conn.close()
            self._pg_conn = None
    
    # ==================== Qdrant ====================
    
    @property
    def qdrant_client(self) -> QdrantClient:
        """Lazy-loaded Qdrant client"""
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=self.rag_config.qdrant_url,
                api_key=self.rag_config.qdrant_api_key
            )
        return self._qdrant_client
    
    def search_vectors(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Vector similarity search trên Qdrant
        
        Args:
            query_vector: Query embedding vector
            limit: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of similar items
        """
        results = self.qdrant_client.search(
            collection_name=self.rag_config.qdrant_collection,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload
            }
            for r in results
        ]
    
    def check_qdrant_health(self) -> bool:
        """Check nếu Qdrant server healthy"""
        try:
            info = self.qdrant_client.get_collections()
            return True
        except Exception as e:
            print(f"Qdrant health check failed: {e}")
            return False
    
    def get_qdrant_collection_info(self) -> Dict:
        """Get thông tin về Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(
                self.rag_config.qdrant_collection
            )
            return {
                "name": self.rag_config.qdrant_collection,
                "vectors_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size
            }
        except Exception as e:
            print(f"Failed to get collection info: {e}")
            return {}
    
    # ==================== Redis ====================
    
    @property
    def redis_client(self) -> redis.Redis:
        """Lazy-loaded Redis client"""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self.rag_config.redis_host,
                port=self.rag_config.redis_port,
                db=self.rag_config.redis_db,
                password=self.rag_config.redis_password,
                decode_responses=True
            )
        return self._redis_client
    
    def get_session(self, user_id: str) -> Optional[Dict]:
        """
        Get conversation session từ Redis
        
        Args:
            user_id: User ID
            
        Returns:
            Session data or None
        """
        session_data = self.redis_client.get(f"session:{user_id}")
        if session_data:
            return json.loads(session_data)
        return None
    
    def save_session(
        self,
        user_id: str,
        session_data: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Save conversation session tới Redis
        
        Args:
            user_id: User ID
            session_data: Session data dict
            ttl: Time-to-live in seconds (mặc định từ config)
            
        Returns:
            True nếu success
        """
        ttl = ttl or self.rag_config.redis_session_ttl
        return self.redis_client.setex(
            f"session:{user_id}",
            ttl,
            json.dumps(session_data)
        )
    
    def delete_session(self, user_id: str) -> bool:
        """Delete conversation session từ Redis"""
        return bool(self.redis_client.delete(f"session:{user_id}"))
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value từ Redis"""
        data = self.redis_client.get(f"cache:{key}")
        if data:
            return json.loads(data)
        return None
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set cached value tới Redis"""
        return self.redis_client.setex(
            f"cache:{key}",
            ttl,
            json.dumps(value)
        )
    
    def check_redis_health(self) -> bool:
        """Check nếu Redis server healthy"""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            print(f"Redis health check failed: {e}")
            return False
    
    # ==================== Health Checks ====================
    
    def check_all_health(self) -> Dict[str, bool]:
        """
        Check health của tất cả storage backends
        
        Returns:
            Dict of health statuses
        """
        return {
            "postgres": self._check_postgres_health(),
            "qdrant": self.check_qdrant_health(),
            "redis": self.check_redis_health()
        }
    
    def _check_postgres_health(self) -> bool:
        """Check nếu PostgreSQL healthy"""
        try:
            with self.pg_connection.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            print(f"PostgreSQL health check failed: {e}")
            return False
    
    # ==================== Cleanup ====================
    
    def close_all(self):
        """Close tất cả connections"""
        self.close_postgres()
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None


# Singleton instance
_storage_manager = None


def get_storage_manager() -> StorageManager:
    """Get StorageManager singleton"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager


# Example usage
if __name__ == "__main__":
    print("=== Storage Manager Health Check ===\n")
    
    manager = get_storage_manager()
    
    # Check health
    health = manager.check_all_health()
    print("Health Status:")
    for service, status in health.items():
        print(f"  {service}: {'✅' if status else '❌'}")
    
    print("\n=== Testing Functionality ===\n")
    
    # Test Redis
    if health['redis']:
        print("Redis:")
        manager.set_cache("test_key", {"message": "hello"})
        cached = manager.get_cache("test_key")
        print(f"  Cache test: {cached}")
    
    # Test Qdrant
    if health['qdrant']:
        print("\nQdrant:")
        qdrant_info = manager.get_qdrant_collection_info()
        print(f"  Collection info: {qdrant_info}")
    
    # Test PostgreSQL
    if health['postgres']:
        print("\nPostgreSQL:")
        chunks = manager.get_chunks("default", limit=3)
        print(f"  Sample chunks: {len(chunks)} found")
    
    print("\n✅ All checks complete!")
    
    # Cleanup
    manager.close_all()
