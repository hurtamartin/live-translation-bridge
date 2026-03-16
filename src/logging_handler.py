import asyncio
import collections
import json as _json
import logging
import os
import queue as _queue
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


class _ThreadSafeLogQueue:
    """Wrapper that bridges thread-safe queue.Queue to async consumption."""
    def __init__(self, maxsize=100):
        self._q = _queue.Queue(maxsize=maxsize)

    def put_nowait(self, item):
        try:
            self._q.put_nowait(item)
        except _queue.Full:
            pass

    async def get(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                return await loop.run_in_executor(None, lambda: self._q.get(timeout=1.0))
            except _queue.Empty:
                continue


class LogBufferHandler(logging.Handler):
    """Custom handler that stores log records in a deque for WebSocket streaming."""
    def __init__(self, maxlen=500):
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.listeners: list[_ThreadSafeLogQueue] = []
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

    def add_listener(self) -> "_ThreadSafeLogQueue":
        q = _ThreadSafeLogQueue(maxsize=100)
        with self._lock:
            self.listeners.append(q)
        return q

    def remove_listener(self, q: "_ThreadSafeLogQueue"):
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
