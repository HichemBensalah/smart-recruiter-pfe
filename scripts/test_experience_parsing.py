from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.structuring.grounding_validator import infer_experience_level, parse_experience_value


def main() -> None:
    exact_cases = {
        "5+": 5.0,
        "2+": 2.0,
        "3-5": 4.0,
        "10 ans": 10.0,
        "5 years": 5.0,
        "2020 - 2022": 2.0,
        "present": None,
        "current": None,
        "ongoing": None,
        None: None,
        "": None,
    }

    for raw_value, expected in exact_cases.items():
        parsed = parse_experience_value(raw_value)
        if expected is None:
            assert parsed is None, f"{raw_value!r} should parse to None, got {parsed!r}"
        else:
            assert parsed is not None, f"{raw_value!r} should not parse to None"
            assert round(parsed, 1) == expected, f"{raw_value!r} expected {expected}, got {parsed}"

    for raw_value in ["07/2025 - 09/2025", None, ""]:
        parse_experience_value(raw_value)

    assert infer_experience_level("5+") == "mid"
    assert infer_experience_level("3-5") == "mid"
    assert infer_experience_level("10 ans") == "senior"
    assert infer_experience_level(None) is None
    assert infer_experience_level("") is None

    print("All experience parsing tests passed.")


if __name__ == "__main__":
    main()
