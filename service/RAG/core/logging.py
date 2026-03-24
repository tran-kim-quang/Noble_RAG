import logging as _stdlib_logging
from core.config import get_settings

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    _stdlib_logging.basicConfig(
        level=getattr(_stdlib_logging, settings.log_level.upper(), _stdlib_logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _configured = True


def get_logger(name: str) -> _stdlib_logging.Logger:
    setup_logging()
    return _stdlib_logging.getLogger(name)
