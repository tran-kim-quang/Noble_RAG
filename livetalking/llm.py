"""
Backward-compatible module for legacy imports.
Use `rag_chat_client.py` for new code.
"""

from rag_chat_client import (  # noqa: F401
    get_noble_runtime_config,
    relay_rag_chat_to_avatar as stream_sales_chat_to_avatar,
)
