#!/usr/bin/env python3
"""Check whether the serving layer returns a consistent model version.

The script samples /model_state and /predict repeatedly and prints every
observed instance_id and loaded_version/model version. If multiple versions are
observed during the sample, the serving layer is inconsistent and may explain
an LMS failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests

DEFAULT_REQUESTS = 20
DEFAULT_PASSWORD = "password"


@dataclass(frozen=True)
class Observation:
    endpoint: str
    request_number: int
    instance_id: str | None
    loaded_version: str | None


def _build_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _empty_to_none(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _observe_model_state(
    base_url: str, request_number: int, timeout: float
) -> Observation:
    response = requests.get(_build_url(base_url, "/model_state"), timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()

    return Observation(
        endpoint="/model_state",
        request_number=request_number,
        instance_id=_empty_to_none(payload.get("instance_id")),
        loaded_version=_empty_to_none(payload.get("loaded_version")),
    )


def _observe_predict(
    base_url: str,
    request_number: int,
    timeout: float,
    password: str,
) -> Observation:
    response = requests.post(
        _build_url(base_url, "/predict"),
        json={"Password": [password]},
        timeout=timeout,
    )
    response.raise_for_status()

    return Observation(
        endpoint="/predict",
        request_number=request_number,
        instance_id=_empty_to_none(response.headers.get("X-Instance-ID")),
        loaded_version=_empty_to_none(response.headers.get("X-Model-Version")),
    )


def _format_value(value: str | None) -> str:
    return value if value is not None else "<none>"


def _print_observations(observations: list[Observation]) -> None:
    print("Observed serving metadata:")
    for observation in observations:
        print(
            f"{observation.request_number:02d} {observation.endpoint}: "
            f"instance_id={_format_value(observation.instance_id)} "
            f"loaded_version={_format_value(observation.loaded_version)}"
        )

    instance_ids = sorted({_format_value(item.instance_id) for item in observations})
    versions = sorted({_format_value(item.loaded_version) for item in observations})

    print(f"Observed instance_id values: {', '.join(instance_ids)}")
    print(f"Observed loaded_version values: {', '.join(versions)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call /model_state and /predict repeatedly to check whether "
            "different model versions are served."
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SERVICE_BASE_URL", "http://localhost:8000"),
        help="Service base URL. Defaults to SERVICE_BASE_URL or http://localhost:8000.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Number of iterations for each endpoint. Defaults to {DEFAULT_REQUESTS}.",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="Password value to send to /predict.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests < 1:
        raise SystemExit("--requests must be at least 1")

    observations: list[Observation] = []
    for request_number in range(1, args.requests + 1):
        observations.append(
            _observe_model_state(args.base_url, request_number, args.timeout)
        )
        observations.append(
            _observe_predict(
                args.base_url,
                request_number,
                args.timeout,
                args.password,
            )
        )

    _print_observations(observations)

    observed_versions = {item.loaded_version for item in observations}
    if len(observed_versions) > 1:
        print(
            "Inconsistent observed versions: serving layer inconsistency "
            "is a likely LMS failure cause.",
            file=sys.stderr,
        )
        return 1

    print("Observed versions are consistent across sampled serving responses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
