import logging
import os
import time
from typing import Any

import requests


_RELOAD_ATTEMPTS = 3
_RELOAD_RETRY_DELAY_SECONDS = 2
_RELOAD_TIMEOUT_SECONDS = 10

logger = logging.getLogger(__name__)


def call_reload_model_endpoint() -> dict[str, Any]:
    """Notify the serving application to reload the production model.

    The endpoint URL and shared secret are read from environment variables so
    training artifacts and logs never need to contain the secret value.
    """
    reload_url = os.getenv("SERVICE_RELOAD_URL")
    reload_secret = os.getenv("SERVICE_RELOAD_SECRET")

    if not reload_url:
        message = "SERVICE_RELOAD_URL is not configured; skipping model reload."
        if os.getenv("CI"):
            raise RuntimeError(
                "SERVICE_RELOAD_URL is required in CI after successful training."
            )

        logger.warning(message)
        return {"status": "skipped", "reason": "SERVICE_RELOAD_URL is not configured"}

    headers = {"X-Service-Token": reload_secret or ""}
    last_error: str | None = None

    for attempt in range(1, _RELOAD_ATTEMPTS + 1):
        try:
            response = requests.post(
                reload_url,
                headers=headers,
                timeout=_RELOAD_TIMEOUT_SECONDS,
            )
            response.raise_for_status()

            try:
                return response.json()
            except ValueError:
                return {
                    "status": "success",
                    "http_status": response.status_code,
                }
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is None:
                last_error = type(exc).__name__
            else:
                last_error = f"{type(exc).__name__} with HTTP status {status_code}"

            if attempt < _RELOAD_ATTEMPTS:
                time.sleep(_RELOAD_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        "Failed to reload service model after "
        f"{_RELOAD_ATTEMPTS} attempts. Last error: {last_error}."
    )
