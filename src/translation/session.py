import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket

from src.config import get_config_snapshot
from src.logging_handler import logger

MAX_CLIENTS = int(os.environ.get("MAX_CLIENTS", "200"))
MAX_CLIENTS_PER_IP = int(os.environ.get("MAX_CLIENTS_PER_IP", "50"))
CLIENT_QUEUE_SIZE = int(os.environ.get("CLIENT_QUEUE_SIZE", "20"))
CLIENT_SEND_TIMEOUT = float(os.environ.get("CLIENT_SEND_TIMEOUT", "2.0"))


@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = field(default_factory=lambda: get_config_snapshot()["default_target_lang"])
    connected_at: float = field(default_factory=time.time)
    client_ip: str = ""
    send_queue: asyncio.Queue[str] = field(default_factory=lambda: asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE))
    writer_task: asyncio.Task | None = None


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._lock = threading.Lock()
        self._stats = {
            "ws_messages_dropped": 0,
            "ws_send_failures": 0,
            "ws_slow_client_disconnects": 0,
        }

    async def _writer(self, session: ClientSession):
        """Single writer for a client WebSocket. Serializes all outbound messages."""
        should_close = False
        try:
            while True:
                payload = await session.send_queue.get()
                try:
                    await asyncio.wait_for(session.websocket.send_text(payload), timeout=CLIENT_SEND_TIMEOUT)
                except Exception as e:
                    logger.debug(f"Send failed for session {session.session_id[:8]}: {e}")
                    with self._lock:
                        self._stats["ws_send_failures"] += 1
                    should_close = True
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._remove_session(session.session_id)
            if should_close:
                try:
                    await asyncio.wait_for(
                        session.websocket.close(code=1011, reason="WebSocket send failed"),
                        timeout=CLIENT_SEND_TIMEOUT,
                    )
                except Exception:
                    pass

    async def connect(self, websocket: WebSocket) -> str | None:
        """Accept a new WebSocket client. Returns session_id, or None if limit reached."""
        session_id, _reason = await self.connect_with_reason(websocket)
        return session_id

    async def connect_with_reason(self, websocket: WebSocket) -> tuple[str | None, str]:
        """Accept a new WebSocket client. Returns (session_id, reject_reason)."""
        session_id = str(uuid.uuid4())
        client_ip = ""
        if websocket.client:
            client_ip = websocket.client.host
        session = ClientSession(session_id=session_id, websocket=websocket, client_ip=client_ip)
        with self._lock:
            if len(self._sessions) >= MAX_CLIENTS:
                return None, "Too many clients"
            if MAX_CLIENTS_PER_IP > 0 and client_ip:
                ip_count = sum(1 for s in self._sessions.values() if s.client_ip == client_ip)
                if ip_count >= MAX_CLIENTS_PER_IP:
                    return None, "Too many clients from this IP"
            self._sessions[session_id] = session
        try:
            await websocket.accept()
            session.writer_task = asyncio.create_task(self._writer(session))
            return session_id, ""
        except Exception:
            self.disconnect(session_id)
            raise

    def _remove_session(self, session_id: str) -> ClientSession | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def disconnect(self, session_id: str):
        session = self._remove_session(session_id)
        if not session:
            return
        task = session.writer_task
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if task and task is not current_task and not task.done():
            task.cancel()

    async def close_session(self, session_id: str, code: int = 1013, reason: str = "Client too slow"):
        session = self._remove_session(session_id)
        if not session:
            return
        task = session.writer_task
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if task and task is not current_task and not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(
                session.websocket.close(code=code, reason=reason),
                timeout=CLIENT_SEND_TIMEOUT,
            )
        except Exception:
            pass

    def set_language(self, session_id: str, lang: str):
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.target_lang = lang

    def get_unique_languages(self) -> set[str]:
        with self._lock:
            return {s.target_lang for s in self._sessions.values()}

    def get_sessions_for_language(self, lang: str) -> list[ClientSession]:
        with self._lock:
            return [s for s in self._sessions.values() if s.target_lang == lang]

    async def send_to_language(self, lang: str, message: str):
        sessions = self.get_sessions_for_language(lang)
        payload = json.dumps({"type": "subtitle", "text": message})
        slow_sessions: list[ClientSession] = []

        for session in sessions:
            try:
                session.send_queue.put_nowait(payload)
            except asyncio.QueueFull:
                slow_sessions.append(session)

        if slow_sessions:
            with self._lock:
                self._stats["ws_messages_dropped"] += len(slow_sessions)
                self._stats["ws_slow_client_disconnects"] += len(slow_sessions)
            for session in slow_sessions:
                logger.warning(f"Disconnecting slow client {session.session_id[:8]}: outbound queue full")
                await self.close_session(session.session_id, code=1013, reason="Client too slow")

    async def send_to_session(self, session_id: str, payload: str):
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            return
        try:
            session.send_queue.put_nowait(payload)
        except asyncio.QueueFull:
            with self._lock:
                self._stats["ws_messages_dropped"] += 1
                self._stats["ws_slow_client_disconnects"] += 1
            logger.warning(f"Disconnecting slow client {session.session_id[:8]}: outbound queue full")
            await self.close_session(session.session_id, code=1013, reason="Client too slow")

    def client_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get_sessions_info(self) -> list[dict]:
        with self._lock:
            now = time.time()
            return [{
                "id": s.session_id[:8],
                "lang": s.target_lang,
                "ip": s.client_ip,
                "connected_for": int(now - s.connected_at),
            } for s in self._sessions.values()]

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
