from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    source_path: Path
    ground_truth_path: Path
    doc_type: str
    language: str
    source_kind: str
    public_url: str | None = None


def load_manifest(manifest_path: str | Path) -> list[BenchmarkSample]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")

    samples: list[BenchmarkSample] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "source_path",
            "ground_truth_path",
            "doc_type",
            "language",
            "source_kind",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row_index, row in enumerate(reader, start=2):
            source_path = Path(row["source_path"])
            ground_truth_path = Path(row["ground_truth_path"])
            sample = BenchmarkSample(
                sample_id=row["sample_id"].strip(),
                source_path=source_path,
                ground_truth_path=ground_truth_path,
                doc_type=row["doc_type"].strip(),
                language=row["language"].strip(),
                source_kind=row["source_kind"].strip(),
                public_url=(row.get("public_url") or "").strip() or None,
            )
            _validate_sample(sample, row_index=row_index)
            samples.append(sample)
    return samples


def load_ground_truth(sample: BenchmarkSample) -> str:
    return sample.ground_truth_path.read_text(encoding="utf-8").strip()


def iter_limited(samples: Iterable[BenchmarkSample], limit: int | None) -> list[BenchmarkSample]:
    if limit is None:
        return list(samples)
    return list(samples)[:limit]


def _validate_sample(sample: BenchmarkSample, *, row_index: int) -> None:
    if not sample.sample_id:
        raise ValueError(f"Manifest row {row_index}: sample_id is empty.")
    if not sample.source_path.exists():
        raise FileNotFoundError(f"Manifest row {row_index}: source file not found: {sample.source_path}")
    if not sample.ground_truth_path.exists():
        raise FileNotFoundError(
            f"Manifest row {row_index}: ground truth file not found: {sample.ground_truth_path}"
        )

