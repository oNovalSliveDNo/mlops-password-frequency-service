from pathlib import Path

import requests


def download_data(data_url: str, output_path: str) -> str:
    if not data_url or not data_url.strip():
        raise ValueError("DATA_URL is empty")

    try:
        response = requests.get(data_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to download data from {data_url}") from exc

    if not response.content:
        raise ValueError("Downloaded file is empty")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)

    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is empty: {path}")

    return str(path)
