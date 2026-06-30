import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 2))

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting (per IP) with upload exemption."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # EXEMPTION: Do not rate limit the upload endpoint to prevent 502s
        if request.url.path.endswith("/upload"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Update and clean old requests
        self._requests[client_ip] = [
            t
            for t in self._requests.get(client_ip, [])
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self.window_seconds,
                },
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
