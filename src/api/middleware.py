"""
API Middleware - Request logging, rate limiting, CORS, and error handling.
"""
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- client={request.client.host if request.client else 'unknown'}"
        )

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 2))

        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"-> {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting (per IP)."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old requests
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip]
                if now - t < self.window_seconds
            ]

        request_times = self._requests.get(client_ip, [])

        if len(request_times) >= self.max_requests:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": self.window_seconds},
            )

        request_times.append(now)
        self._requests[client_ip] = request_times
        return await call_next(request)
