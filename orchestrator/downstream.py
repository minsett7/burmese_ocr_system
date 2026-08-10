from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class DownstreamError(Exception):
    service: str
    message: str
    status_code: int = 502
    upstream_status: int | None = None
    detail: Any = None

    def __str__(self) -> str:
        return f"{self.service}: {self.message}"


class DownstreamClient:
    def __init__(self, timeout: float, attempts: int, backoff: float, transport: httpx.AsyncBaseTransport | None = None):
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.transport = transport

    async def request(
        self,
        service: str,
        method: str,
        url: str,
        *,
        correlation_id: str,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        request_headers = {"X-Correlation-ID": correlation_id, **(headers or {})}
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                    response = await client.request(method, url, headers=request_headers, **kwargs)
                if response.status_code < 400:
                    return response
                detail: Any
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text[:1000]
                # Invalid requests and other permanent 4xx responses are never retried.
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise DownstreamError(service, "upstream rejected the request", 502, response.status_code, detail)
                last_error = DownstreamError(service, "temporary upstream failure", 503, response.status_code, detail)
            except DownstreamError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
            if attempt + 1 < self.attempts:
                await asyncio.sleep(self.backoff * (2**attempt))
        if isinstance(last_error, DownstreamError):
            raise last_error
        raise DownstreamError(service, "upstream is unavailable", 503, detail=str(last_error))
