"""Shared fixtures for the Whisper Cloud tests.

Transcription requests are exercised against a scripted local HTTP server,
so the whole request path (framing, headers, retries) runs for real
without touching the internet.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp import web

TEST_TRANSCRIPTION = "turn on the living room light"


class FakeTranscriptionServer:
    """A local server emulating OpenAI-compatible transcription endpoints.

    ``responses`` is a list of scripts consumed per request (the last one is
    reused). A script is either a ``(status, payload)`` tuple or a callable
    taking the request record and returning one:

    - ``(200, None)`` -> ``{"text": TEST_TRANSCRIPTION}``
    - ``(200, dict)`` -> the dict, as JSON
    - ``(422, "field_name")`` -> FastAPI style missing field body
    - any other status -> the payload as plain text
    """

    def __init__(
        self, responses: list[Any], models_responses: list[Any] | None = None
    ) -> None:
        """Initialize the scripted responses."""
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.models_responses = list(models_responses or [])
        self.models_requests: list[dict[str, Any]] = []
        self.delay = 0.0

    async def start(self) -> FakeTranscriptionServer:
        """Start the server and return self (with ``url`` set)."""
        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", self._transcriptions)
        app.router.add_post("/v1/chat/completions", self._chat_completions)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_get("/v1/models/{model}", self._handle_models)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.url = f"http://{host}:{port}"
        return self

    async def stop(self) -> None:
        """Stop the server."""
        await self._runner.cleanup()

    def _script(self, record: dict[str, Any]) -> tuple[Any, Any]:
        script = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if callable(script):
            return script(record)
        return script

    async def _record_multipart(self, request: web.Request) -> dict[str, Any]:
        record: dict[str, Any] = {
            "time": time.monotonic(),
            "kind": "transcriptions",
            "chunked": "chunked"
            in (request.headers.get("Transfer-Encoding") or "").lower(),
            "auth": request.headers.get("Authorization"),
            "fields": {},
            "file_field": None,
            "audio": b"",
            "audio_arrivals": [],
            "aborted": False,
        }
        try:
            reader = await request.multipart()
            part = await reader.next()
            while part is not None:
                if part.filename:
                    record["file_field"] = part.name
                    while True:
                        chunk = await part.read_chunk(2**16)
                        if not chunk:
                            break
                        record["audio"] += chunk
                        record["audio_arrivals"].append(time.monotonic())
                else:
                    record["fields"][part.name] = (await part.read()).decode()
                part = await reader.next()
        except (ConnectionResetError, asyncio.IncompleteReadError):
            record["aborted"] = True
        return record

    async def _respond(
        self, request: web.Request, record: dict[str, Any]
    ) -> web.Response:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.requests.append(record)
        status, payload = self._script(record)
        if status == "abort":
            # respond 200 with a body that is cut off mid-stream
            response = web.StreamResponse(
                status=200, headers={"Content-Length": "1000"}
            )
            await response.prepare(request)
            await response.write(b'{"text": "trunca')
            response.force_close()
            return response
        if status == 200 and payload is None:
            return web.json_response({"text": TEST_TRANSCRIPTION})
        if status == 200 and isinstance(payload, str):
            # Plain text response (e.g. whisper-asr-webservice /asr)
            return web.Response(text=payload)
        if status == 200:
            return web.json_response(payload)
        if status == 422:
            if isinstance(payload, str):
                return web.json_response(
                    {
                        "detail": [
                            {
                                "loc": ["body", payload],
                                "type": "value_error.missing",
                                "msg": "Field required",
                            }
                        ]
                    },
                    status=422,
                )
            # a dict payload is served as-is: a 422 without a learnable hint
            return web.json_response(payload, status=422)
        return web.Response(status=status, text=str(payload or "boom"))

    async def _transcriptions(self, request: web.Request) -> web.Response:
        if (script := self.responses[0] if len(self.responses) > 1 else None) == (
            "reset",
            None,
        ):
            # kill the connection in the middle of the client's upload
            self.responses.pop(0)
            self.requests.append({"reset": True, "time": time.monotonic()})
            request.transport.abort()
            return web.Response(status=500)  # pragma: no cover - never sent
        record = await self._record_multipart(request)
        return await self._respond(request, record)

    async def _chat_completions(self, request: web.Request) -> web.Response:
        import base64
        import json

        body = await request.json()
        content_part = body["messages"][0]["content"][0]
        data_url = content_part["input_audio"]["data"]
        prefix = "data:audio/wav;base64,"
        record: dict[str, Any] = {
            "time": time.monotonic(),
            "kind": "chat_completions",
            "chunked": "chunked"
            in (request.headers.get("Transfer-Encoding") or "").lower(),
            "auth": request.headers.get("Authorization"),
            "json": body,
            "audio": base64.b64decode(data_url[len(prefix) :]),
        }
        return await self._respond(request, record)

    async def _handle_models(self, request: web.Request) -> web.Response:
        """Serve scripted GET /v1/models responses."""
        self.models_requests.append(
            {
                "path": request.path,
                "auth": request.headers.get("Authorization"),
            }
        )
        if not self.models_responses:
            return web.json_response({"error": "not found"}, status=404)
        script = (
            self.models_responses.pop(0)
            if len(self.models_responses) > 1
            else self.models_responses[0]
        )
        status, payload = script(request) if callable(script) else script
        if status == 200:
            if isinstance(payload, str):
                # a plain text payload: an unparseable body for robustness tests
                return web.Response(text=payload)
            return web.json_response(payload)
        return web.json_response({"error": str(payload or "denied")}, status=status)


@pytest.fixture
async def transcription_server_factory() -> Callable[
    ..., Awaitable[FakeTranscriptionServer]
]:
    """Create scripted servers that are cleaned up after the test."""
    servers: list[FakeTranscriptionServer] = []

    async def make(
        responses: list[Any], models_responses: list[Any] | None = None
    ) -> FakeTranscriptionServer:
        server = await FakeTranscriptionServer(responses, models_responses).start()
        servers.append(server)
        return server

    yield make

    for server in servers:
        await server.stop()


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make transient-error retries instant."""
    monkeypatch.setattr(
        "custom_components.openai_whisper_cloud.client.RETRY_DELAY", 0.01
    )
