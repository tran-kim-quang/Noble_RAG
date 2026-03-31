from typing import Any, Dict, Optional

from stores.customer_identity_store import CustomerIdentityStore
from stores.purchase_history_store import PurchaseHistoryStore


class CustomerContextBuilder:
    """Build identity-scoped context for vision service only.

    Vision service should not own full chat/session business logic from RAG.
    """

    def __init__(
        self,
        identity_store: Optional[CustomerIdentityStore] = None,
        purchase_store: Optional[PurchaseHistoryStore] = None,
    ) -> None:
        self.identity_store = identity_store or CustomerIdentityStore()
        self.purchase_store = purchase_store or PurchaseHistoryStore()

    async def build(self, customer_id: str, fallback_session_id: Optional[str] = None) -> Dict[str, Any]:
        sessions = await self.identity_store.list_customer_sessions(customer_id, limit=5)
        purchase_history = await self.purchase_store.get_purchase_history(customer_id)
        recent_session_context: Dict[str, Any] = {
            "recent_sessions": sessions[:3],
        }
        if fallback_session_id:
            recent_session_context["requested_session_id"] = fallback_session_id

        return {
            "lead_profile": {},
            "recent_chat_history": [],
            "recent_session_context": recent_session_context,
            "purchase_history": purchase_history,
        }
