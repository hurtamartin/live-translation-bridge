import asyncio
import json
import types
import unittest

from src.translation.session import ClientSession, SessionManager


class FakeWebSocket:
    def __init__(self):
        self.client = types.SimpleNamespace(host="127.0.0.1")
        self.accepted = False
        self.sent: list[str] = []
        self.closed: list[tuple[int, str]] = []

    async def accept(self):
        self.accepted = True

    async def send_text(self, payload: str):
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed.append((code, reason))


class SessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_cleans_session_and_writer_task(self):
        manager = SessionManager()
        websocket = FakeWebSocket()

        session_id = await manager.connect(websocket)
        self.assertIsNotNone(session_id)
        self.assertTrue(websocket.accepted)

        manager.set_language(session_id, "eng")
        await manager.send_to_language("eng", "hello")

        for _ in range(10):
            if websocket.sent:
                break
            await asyncio.sleep(0.01)

        self.assertEqual(json.loads(websocket.sent[0]), {"type": "subtitle", "text": "hello"})

        task = manager._sessions[session_id].writer_task
        manager.disconnect(session_id)
        if task is not None:
            await task

        self.assertEqual(manager.client_count(), 0)

    async def test_full_client_queue_disconnects_slow_session(self):
        manager = SessionManager()
        websocket = FakeWebSocket()
        queue = asyncio.Queue(maxsize=1)
        await queue.put('{"type":"subtitle","text":"old"}')
        session = ClientSession(
            session_id="slow-session",
            websocket=websocket,
            target_lang="eng",
            client_ip="127.0.0.1",
            send_queue=queue,
        )
        manager._sessions[session.session_id] = session

        await manager.send_to_language("eng", "new")

        self.assertEqual(manager.client_count(), 0)
        self.assertEqual(websocket.closed, [(1013, "Client too slow")])
        stats = manager.get_stats()
        self.assertEqual(stats["ws_messages_dropped"], 1)
        self.assertEqual(stats["ws_slow_client_disconnects"], 1)


if __name__ == "__main__":
    unittest.main()
