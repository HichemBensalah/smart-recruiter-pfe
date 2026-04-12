from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "data" / "raw_cv"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
BENCHMARK_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "ocr" / "dataset"


@dataclass(slots=True)
class BootstrapEntry:
    sample_id: str
    source_path: Path
    ground_truth_source: Path
    doc_type: str
    language: str
    source_kind: str
    notes: str


DOCX_TO_SOURCES = {
    "0_anonyme": [RAW_ROOT / "pdf" / "0_anonyme.pdf"],
    "1_anonyme": [RAW_ROOT / "pdf" / "1_anonyme.pdf"],
    "2_anonyme": [RAW_ROOT / "pdf" / "2_anonyme.pdf"],
    "3_anonyme": [RAW_ROOT / "pdf" / "3_anonyme.pdf"],
    "Aziz_resumer": [RAW_ROOT / "pdf" / "Aziz_resume.pdf"],
    "Hichem_resume": [RAW_ROOT / "pdf" / "Hichem_resume.pdf", RAW_ROOT / "images" / "Hichem_image.jpg"],
    "resume_khairi": [RAW_ROOT / "pdf" / "khairi_resume.pdf", RAW_ROOT / "images" / "khairi_image.jpg"],
}


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest or BENCHMARK_ROOT / "manifests" / "benchmark_manifest.csv")
    raw_root = BENCHMARK_ROOT / "raw"
    gt_root = BENCHMARK_ROOT / "ground_truth"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    entries = build_entries()
    write_dataset(entries, raw_root=raw_root, gt_root=gt_root, manifest_path=manifest_path)
    print(f"Bootstrap complete. Manifest written to: {manifest_path}")
    print(f"Samples created: {len(entries)}")


def build_entries() -> list[BootstrapEntry]:
    entries: list[BootstrapEntry] = []
    for docx_stem, source_paths in DOCX_TO_SOURCES.items():
        gt_source = RAW_ROOT / "docx" / f"{docx_stem}.docx"
        if not gt_source.exists():
            continue
        for source_path in source_paths:
            if not source_path.exists():
                continue
            sample_id = f"{docx_stem}__{source_path.stem}"
            entries.append(
                BootstrapEntry(
                    sample_id=sample_id,
                    source_path=source_path,
                    ground_truth_source=gt_source,
                    doc_type="cv",
                    language="fr_en",
                    source_kind="pdf" if source_path.suffix.lower() == ".pdf" else "image",
                    notes="proxy_ground_truth_from_matching_docx_text",
                )
            )
    return entries


def write_dataset(
    entries: list[BootstrapEntry],
    *,
    raw_root: Path,
    gt_root: Path,
    manifest_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for entry in entries:
        dest_source = raw_root / f"{entry.sample_id}{entry.source_path.suffix.lower()}"
        dest_gt = gt_root / f"{entry.sample_id}.txt"
        shutil.copy2(entry.source_path, dest_source)
        dest_gt.write_text(extract_docx_plain_text(entry.ground_truth_source), encoding="utf-8")
        rows.append(
            {
                "sample_id": entry.sample_id,
                "source_path": str(dest_source.relative_to(PROJECT_ROOT)).replace("/", "\\"),
                "ground_truth_path": str(dest_gt.relative_to(PROJECT_ROOT)).replace("/", "\\"),
                "doc_type": entry.doc_type,
                "language": entry.language,
                "source_kind": entry.source_kind,
                "public_url": "",
                "notes": entry.notes,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "source_path",
                "ground_truth_path",
                "doc_type",
                "language",
                "source_kind",
                "public_url",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def extract_docx_plain_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", namespaces):
        texts = [node.text or "" for node in para.findall(".//w:t", namespaces)]
        paragraph = "".join(texts).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n".join(paragraphs).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap a small OCR benchmark dataset from existing repo assets."
    )
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
