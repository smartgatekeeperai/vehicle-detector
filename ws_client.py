# ws_client.py

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import websockets


WS_SERVER_URL = os.getenv("WS_SERVER_URL", "ws://127.0.0.1:3000/ws")
WS_QUEUE_MAXSIZE = int(os.getenv("WS_QUEUE_MAXSIZE", "2000"))
WS_CONNECT_TIMEOUT = float(os.getenv("WS_CONNECT_TIMEOUT", "3.0"))
WS_RECONNECT_SECONDS = float(os.getenv("WS_RECONNECT_SECONDS", "2.0"))


class RealtimeWsClient:
    def __init__(self) -> None:
        self.url = WS_SERVER_URL
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=WS_QUEUE_MAXSIZE)
        self.ws = None
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False

        self.connected = False
        self.last_error: Optional[str] = None
        self.last_connect_ts: Optional[int] = None
        self.last_send_ts: Optional[int] = None
        self.dropped_count = 0

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.worker_task = asyncio.create_task(self._worker(), name="realtime-ws-worker")

    async def stop(self) -> None:
        self.running = False

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self.worker_task = None

        await self._close_ws()

    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        payload = {
            "type": event_type,
            "data": data,
            "ts": int(time.time() * 1000),
        }

        try:
            self.queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            self.dropped_count += 1
            self.last_error = "websocket queue full"
            return False

    async def _close_ws(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None
        self.connected = False

    async def _connect(self) -> None:
        await self._close_ws()

        self.ws = await asyncio.wait_for(
            websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=2 * 1024 * 1024,
            ),
            timeout=WS_CONNECT_TIMEOUT,
        )

        await self.ws.send(
            json.dumps(
                {
                    "type": "hello",
                    "role": "fastapi",
                    "ts": int(time.time() * 1000),
                }
            )
        )

        self.connected = True
        self.last_error = None
        self.last_connect_ts = int(time.time() * 1000)
        print(f"[WS] connected to {self.url}")

    async def _worker(self) -> None:
        while self.running:
            try:
                payload = await self.queue.get()

                while self.running:
                    try:
                        if self.ws is None or getattr(self.ws, "closed", False):
                            await self._connect()

                        await self.ws.send(json.dumps(payload))
                        self.last_send_ts = int(time.time() * 1000)
                        break

                    except Exception as e:
                        self.connected = False
                        self.last_error = str(e)
                        print(f"[WS] send/connect failed: {e}")
                        await self._close_ws()
                        await asyncio.sleep(WS_RECONNECT_SECONDS)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = str(e)
                print(f"[WS] worker error: {e}")
                await asyncio.sleep(1.0)

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "websocket",
            "url": self.url,
            "connected": self.connected,
            "queue_size": self.queue.qsize(),
            "queue_maxsize": WS_QUEUE_MAXSIZE,
            "dropped_count": self.dropped_count,
            "last_error": self.last_error,
            "last_connect_ts": self.last_connect_ts,
            "last_send_ts": self.last_send_ts,
        }


realtime_ws = RealtimeWsClient()


async def start_realtime_ws() -> None:
    await realtime_ws.start()


async def stop_realtime_ws() -> None:
    await realtime_ws.stop()


async def publish_event(event_type: str, data: Dict[str, Any]) -> bool:
    return await realtime_ws.publish(event_type, data)


def ws_status() -> Dict[str, Any]:
    return realtime_ws.status()