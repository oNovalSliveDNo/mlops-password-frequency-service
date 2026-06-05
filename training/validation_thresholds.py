from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeGuard

DEFAULT_SCHEMA_THRESHOLDS: dict[str, float | int] = {
    "min_positive_target_share": 0.30,
    "max_positive_target_share": 0.70,
    "max_password_length": 128,
}

DEFAULT_MODEL_QUALITY_THRESHOLDS: dict[str, float] = {
    "max_rmse": 0.23,
    "max_mae": 0.18,
    "max_abs_mean_error": 0.09,
    "max_prediction_target_mean_gap": 0.10,
}

DEFAULT_VALIDATION_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "schema": DEFAULT_SCHEMA_THRESHOLDS,
    "model_quality": DEFAULT_MODEL_QUALITY_THRESHOLDS,
}

DEFAULT_VALIDATION_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[1] / "reference" / "validation_thresholds.json"
)


def _default_thresholds_copy() -> dict[str, dict[str, float | int]]:
    return deepcopy(DEFAULT_VALIDATION_THRESHOLDS)


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def load_validation_thresholds(
    path: str | Path = DEFAULT_VALIDATION_THRESHOLDS_PATH,
) -> dict[str, dict[str, float | int]]:
    """Load validation thresholds, falling back to defaults for missing values.

    The reference JSON is intentionally optional at runtime. If it is absent,
    malformed, or omits any known section/key, the code constants above provide
    the threshold values needed for validation to keep operating.
    """
    thresholds = _default_thresholds_copy()
    threshold_path = Path(path)

    try:
        loaded = json.loads(threshold_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return thresholds

    if not isinstance(loaded, dict):
        return thresholds

    for section, defaults in thresholds.items():
        loaded_section = loaded.get(section)
        if not isinstance(loaded_section, dict):
            continue

        for key, default_value in defaults.items():
            loaded_value = loaded_section.get(key)
            if _is_number(loaded_value):
                thresholds[section][key] = (
                    int(loaded_value)
                    if isinstance(default_value, int)
                    else float(loaded_value)
                )

    return thresholds
