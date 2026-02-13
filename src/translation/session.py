import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket

from src.config import runtime_config


@dataclass
class ClientSession:
    session_id: str
    websocket: WebSocket
    target_lang: str = field(default_factory=lambda: runtime_config["default_target_lang"])
    connected_at: float = field(default_factory=time.time)
    client_ip: str = ""


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = str(uuid.uuid4())
        client_ip = ""
        if websocket.client:
            client_ip = websocket.client.host
        session = ClientSession(session_id=session_id, websocket=websocket, client_ip=client_ip)
        with self._lock:
            self._sessions[session_id] = session
        return session_id

    def disconnect(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

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

        async def _safe_send(session):
            try:
                await session.websocket.send_text(payload)
            except Exception:
                pass

        await asyncio.gather(*[_safe_send(s) for s in sessions])

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
