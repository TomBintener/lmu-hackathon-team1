import hashlib
import json
import time
from typing import Any, Protocol

import httpx

from app.config import Settings


class DiscogsError(Exception):
    """Base error for upstream catalog provider failures."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class DiscogsAuthError(DiscogsError):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message, status_code=status_code)


class DiscogsRateLimitError(DiscogsError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=429)


class DiscogsUpstreamError(DiscogsError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code)


class CatalogProvider(Protocol):
    def search_releases(
        self, params: dict[str, Any], bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]: ...

    def get_release(
        self, release_id: int, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]: ...


class ApiCacheStore(Protocol):
    def get_api_cache(self, cache_key: str) -> dict[str, Any] | None: ...

    def set_api_cache(
        self, cache_key: str, endpoint: str, status_code: int, payload: dict[str, Any]
    ) -> None: ...


class DiscogsClient:
    """Direct HTTP client for Discogs REST API v2 with PG api_cache & rate pacing."""

    def __init__(
        self,
        settings: Settings,
        cache_store: ApiCacheStore | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._cache = cache_store
        self._own_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=settings.discogs_base_url,
            timeout=settings.discogs_timeout_sec,
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    @staticmethod
    def _build_cache_key(endpoint: str, params: dict[str, Any] | None) -> str:
        raw = json.dumps(
            {"endpoint": endpoint, "params": params or {}},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _headers(self) -> dict[str, str]:
        token = (self._settings.discogs_token or "").strip()
        if not token or token == "PASTE_YOUR_DISCOGS_PERSONAL_ACCESS_TOKEN":
            raise DiscogsAuthError(
                "DISCOGS_TOKEN is not configured in environment/.env.",
                status_code=502,
            )
        return {
            "User-Agent": self._settings.discogs_user_agent,
            "Authorization": f"Discogs token={token}",
            "Accept": "application/json",
        }

    def _request_json(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        bypass_cache: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        cache_key = self._build_cache_key(endpoint, params)
        if not bypass_cache and self._cache is not None:
            cached = self._cache.get_api_cache(cache_key)
            if cached is not None:
                return cached["payload_json"], True

        headers = self._headers()
        max_retries = 2
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                response = self._client.get(endpoint, params=params, headers=headers)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise DiscogsUpstreamError(
                    f"Discogs network error on {endpoint}: {exc}",
                    status_code=502,
                ) from exc

            remaining_hdr = response.headers.get("X-Discogs-Ratelimit-Remaining")
            if remaining_hdr is not None:
                try:
                    remaining = int(remaining_hdr)
                    if remaining <= self._settings.discogs_min_remaining:
                        time.sleep(1.1)
                except ValueError:
                    pass

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise DiscogsUpstreamError(
                        f"Invalid JSON from Discogs on {endpoint}: {exc}",
                        status_code=502,
                    ) from exc
                if self._cache is not None:
                    self._cache.set_api_cache(cache_key, endpoint, 200, payload)
                return payload, False

            if response.status_code in (401, 403):
                raise DiscogsAuthError(
                    f"Discogs authentication/User-Agent error ({response.status_code}) on {endpoint}.",
                    status_code=502,
                )

            if response.status_code == 404:
                # Specific release ID not found
                return {}, False

            if response.status_code == 429:
                if attempt < max_retries:
                    retry_after = response.headers.get("Retry-After")
                    delay = 2.0 * (attempt + 1)
                    if retry_after:
                        try:
                            delay = min(float(retry_after), 10.0)
                        except ValueError:
                            pass
                    time.sleep(delay)
                    continue
                raise DiscogsRateLimitError(
                    f"Discogs rate limit (429) exceeded on {endpoint}."
                )

            if response.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                raise DiscogsUpstreamError(
                    f"Discogs server error ({response.status_code}) on {endpoint}.",
                    status_code=502,
                )

            raise DiscogsUpstreamError(
                f"Unexpected Discogs HTTP {response.status_code} on {endpoint}.",
                status_code=502,
            )

        raise DiscogsUpstreamError(f"Discogs request failed on {endpoint}: {last_exc}")

    def search_releases(
        self, params: dict[str, Any], bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        query_params = {"type": "release", "per_page": 10, **params}
        return self._request_json(
            "/database/search", params=query_params, bypass_cache=bypass_cache
        )

    def get_release(
        self, release_id: int, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        return self._request_json(
            f"/releases/{release_id}", params=None, bypass_cache=bypass_cache
        )

    def get_master_versions(
        self, master_id: int, per_page: int = 25, bypass_cache: bool = False
    ) -> tuple[dict[str, Any], bool]:
        return self._request_json(
            f"/masters/{master_id}/versions",
            params={"per_page": per_page},
            bypass_cache=bypass_cache,
        )
