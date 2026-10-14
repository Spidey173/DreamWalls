import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from app.core.config import settings

class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 5):
        self.requests_per_minute = requests_per_minute
        self.client_requests = defaultdict(list)

    def check(self, request: Request):
        # Identify client by IP
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - 60.0

        # Filter out timestamps older than 60 seconds
        timestamps = [t for t in self.client_requests[client_ip] if t > window_start]
        self.client_requests[client_ip] = timestamps

        if len(timestamps) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} uploads per minute allowed."
            )

        self.client_requests[client_ip].append(now)

rate_limiter = InMemoryRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)
