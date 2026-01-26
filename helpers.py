import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def load_soql(path: str) -> str:
    """
    Load a SoQL query from disk and normalize whitespace.
    """
    text = Path(path).read_text(encoding="utf-8")
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    max_retries: int = 6,
    backoff_base: float = 1.0,
    backoff_cap: float = 60.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Response:
    """
    Robust HTTP request wrapper with retry/backoff.

    Retries on:
      - network errors / timeouts
      - HTTP 429 and common 5xx statuses (configurable)

    For 429, honors Retry-After header when present.

    Raises:
      - requests.RequestException for persistent network issues
      - RuntimeError for non-retriable HTTP errors or if retries exhausted
    """
    method = method.upper()

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.request(
                method,
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            # Success
            if resp.status_code < 400:
                return resp

            # Decide retry vs fail fast
            if resp.status_code in retry_statuses:
                # 429: rate-limited
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = float(ra)
                            time.sleep(min(sleep_s, backoff_cap))
                            continue
                        except ValueError:
                            # fall back to backoff below
                            pass

                # 5xx or 429 without valid Retry-After: backoff + retry
                if attempt < max_retries:
                    sleep_s = min(backoff_base * (2**attempt), backoff_cap)
                    sleep_s = sleep_s * (0.8 + 0.4 * random.random())  # jitter
                    time.sleep(sleep_s)
                    continue

                # Exhausted retries for retriable status
                raise RuntimeError(
                    f"HTTP {resp.status_code} after {max_retries} retries. Body: {resp.text[:2000]}"
                )

            # Non-retriable 4xx (e.g., 400, 401, 403, 404)
            raise RuntimeError(
                f"HTTP {resp.status_code} (non-retriable). Body: {resp.text[:2000]}"
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.RequestException,
        ) as e:
            last_exc = e
            if attempt < max_retries:
                sleep_s = min(backoff_base * (2**attempt), backoff_cap)
                sleep_s = sleep_s * (0.8 + 0.4 * random.random())  # jitter
                time.sleep(sleep_s)
                continue
            raise RuntimeError(
                f"Request failed after {max_retries} retries: {e}"
            ) from e

    # Should never reach here, but be defensive
    raise RuntimeError(f"Request failed: {last_exc}")
