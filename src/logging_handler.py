import asyncio
import collections
import logging
import threading
import time


logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)


class LogBufferHandler(logging.Handler):
    """Custom handler that stores log records in a deque for WebSocket streaming."""
    def __init__(self, maxlen=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.listeners: list[asyncio.Queue] = []
        self._lock = threading.Lock()

    def emit(self, record):
        entry = {
            "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "message": self.format(record),
        }
        with self._lock:
            self.buffer.append(entry)
            for q in self.listeners:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    pass

    def add_listener(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        with self._lock:
            self.listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        with self._lock:
            try:
                self.listeners.remove(q)
            except ValueError:
                pass

    def get_history(self) -> list[dict]:
        with self._lock:
            return list(self.buffer)


log_handler = LogBufferHandler()
log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(log_handler)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console_handler)
