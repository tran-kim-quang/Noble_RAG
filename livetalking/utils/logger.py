import logging
import sys


class _UnicodeSafeStream:
    def __init__(self, stream):
        self._stream = stream

    def write(self, message):
        try:
            return self._stream.write(message)
        except UnicodeEncodeError:
            enc = getattr(self._stream, "encoding", None) or "utf-8"
            safe = str(message).encode(enc, errors="replace").decode(enc, errors="replace")
            return self._stream.write(safe)

    def flush(self):
        try:
            return self._stream.flush()
        except Exception:
            return None


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

_file_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_console_fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

if not logger.handlers:
    fhandler = logging.FileHandler("livetalking.log", encoding="utf-8", errors="replace")
    fhandler.setFormatter(_file_fmt)
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)

    shandler = logging.StreamHandler(_UnicodeSafeStream(sys.stdout))
    shandler.setFormatter(_console_fmt)
    shandler.setLevel(logging.DEBUG)
    logger.addHandler(shandler)
