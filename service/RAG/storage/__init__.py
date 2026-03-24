"""
Storage Module - Unified interface cho PostgreSQL, Qdrant, Redis
"""

from .manager import StorageManager, get_storage_manager

__all__ = [
    'StorageManager',
    'get_storage_manager'
]
