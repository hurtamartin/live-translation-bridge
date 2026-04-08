import asyncio
import collections
import json as _json
import logging
import os
import threading
import time


logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production log aggregation (ELK, Datadog)."""
    def format(self, record):
        return _json.dumps({
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }, ensure_ascii=False)


class _AsyncLogQueue:
    """Thread-safe bridge from logging threads into an asyncio.Queue listener."""

    def __init__(self, maxsize=100):
        self._loop = asyncio.get_running_loop()
        self._q = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    def put_nowait(self, item):
        if self._closed:
            return

        def _put():
            if self._closed:
                return
            if self._q.full():
                try:
                    self._q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                self._q.put_nowait(item)
            except asyncio.QueueFull:
                pass

        try:
            self._loop.call_soon_threadsafe(_put)
        except RuntimeError:
            self._closed = True

    async def get(self):
        return await self._q.get()

    def close(self):
        self._closed = True


class LogBufferHandler(logging.Handler):
    """Custom handler that stores log records in a deque for WebSocket streaming."""
    def __init__(self, maxlen=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.listeners: list[_AsyncLogQueue] = []
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
                q.put_nowait(entry)

    def add_listener(self) -> "_AsyncLogQueue":
        q = _AsyncLogQueue(maxsize=100)
        with self._lock:
            self.listeners.append(q)
        return q

    def remove_listener(self, q: "_AsyncLogQueue"):
        q.close()
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

# Also log to console — use JSON format if LOG_FORMAT=json
console_handler = logging.StreamHandler()
if os.environ.get("LOG_FORMAT", "").lower() == "json":
    console_handler.setFormatter(JsonFormatter())
else:
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console_handler)
