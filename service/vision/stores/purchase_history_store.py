import json
from pathlib import Path
from typing import Any, Dict, List

from core.config import get_settings


class PurchaseHistoryStore:
    def __init__(self) -> None:
        settings = get_settings()
        configured = Path(settings.vision_purchase_history_path)
        if configured.is_absolute():
            self.path = configured
        else:
            self.path = Path(__file__).resolve().parents[1] / configured

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_json({"customers": {}})

    def _read_json(self) -> Dict[str, Any]:
        self._ensure_file()
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, payload: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    async def get_purchase_history(self, customer_id: str) -> List[Dict[str, Any]]:
        payload = self._read_json()
        items = payload.get("customers", {}).get(customer_id, [])
        return items if isinstance(items, list) else []
