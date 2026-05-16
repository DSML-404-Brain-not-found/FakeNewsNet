#!/usr/bin/env python3
"""
prepare_title_top_img_dataset.py

Purpose
-------
Build a lightweight FakeNewsNet dataset for CoCoCLIP-style multimodal experiments:

    title + top image + label

This script directly reads FakeNewsNet minimal CSV files such as:

    dataset/politifact_fake.csv
    dataset/politifact_real.csv

It does NOT require the full `main.py` crawling output and does NOT store article body text.
It uses newspaper3k only to extract each article's top image URL, downloads that image,
and writes a clean manifest for model training.

Bridge with prepare_fakenewsnet_dataset.py
------------------------------------------
To stay compatible with the existing `prepare_fakenewsnet_dataset.py` output style, this
script also exports:

    metadata/news_metadata.csv
    metadata/news_metadata.jsonl

with compatible columns:

    dataset,label,news_id,title,text,url,publish_date,source,top_img,image_urls,json_path,csv_path

Here, `text` is intentionally empty because this dataset uses title-only text.
`top_img` and `image_urls` both point to the top image URL when available.

Outputs
-------
prepared_fakenewsnet_politifact_title_top_img/
├── images/
│   └── politifact/
│       ├── fake/
│       └── real/
├── metadata/
│   ├── manifest.csv
│   ├── manifest.jsonl
│   ├── news_metadata.csv
│   └── news_metadata.jsonl
├── logs/
│   ├── top_img_success.csv
│   ├── top_img_failures.csv
│   └── run_summary.json
└── fakenewsnet_politifact_title_top_img.zip

Example
-------
python prepare_title_top_img_dataset.py \
  --input-root ./dataset \
  --output-root ./prepared_fakenewsnet_politifact_title_top_img \
  --datasets politifact \
  --labels fake real \
  --max-workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: requests. Install it with: pip install requests") from exc

try:
    from newspaper import Article
except ImportError as exc:
    raise SystemExit("Missing dependency: newspaper3k. Install it with: pip install newspaper3k") from exc


DEFAULT_DATASETS = ("gossipcop", "politifact")
DEFAULT_LABELS = ("fake", "real")
LABEL_ID = {"real": 0, "fake": 1}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class SourceRecord:
    dataset: str
    label: str
    label_id: int
    news_id: str
    title: str
    url: str
    csv_path: str


@dataclass
class ManifestRecord:
    id: str
    dataset: str
    label: str
    label_id: int
    title: str
    url: str
    top_img: str
    image_path: str


@dataclass
class NewsMetadataRecord:
    dataset: str
    label: str
    news_id: str
    title: str = ""
    text: str = ""
    url: str = ""
    publish_date: str = ""
    source: str = ""
    top_img: str = ""
    image_urls: str = ""
    json_path: str = ""
    csv_path: str = ""


@dataclass
class ProcessResult:
    status: str  # success / failed / skipped
    dataset: str
    label: str
    label_id: int
    news_id: str
    title: str
    url: str
    top_img: str = ""
    image_path: str = ""
    error_type: str = ""
    error_message: str = ""
    http_status: str = ""
    content_type: str = ""
    size_bytes: int = 0


def set_csv_field_limit() -> None:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare title + top_img-only FakeNewsNet dataset for CoCoCLIP-style experiments."
    )
    parser.add_argument("--input-root", type=Path, default=Path("./dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("./prepared_fakenewsnet_politifact_title_top_img"))
    parser.add_argument("--datasets", nargs="+", default=["politifact"], choices=list(DEFAULT_DATASETS))
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS), choices=list(DEFAULT_LABELS))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-download-images", action="store_true")
    parser.add_argument("--zip-name", default="fakenewsnet_politifact_title_top_img")
    return parser.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def stable_news_id(raw: str) -> str:
    raw = safe_str(raw).strip()
    if raw:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:160]
    return "unknown"


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def extension_from_response(url: str, content_type: str) -> str:
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg", ".ico"}:
        return ext
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            if guessed == ".jpe":
                return ".jpg"
            return guessed
    return ".jpg"


def read_source_records(input_root: Path, datasets: Iterable[str], labels: Iterable[str]) -> List[SourceRecord]:
    records: List[SourceRecord] = []
    for dataset in datasets:
        for label in labels:
            csv_path = input_root / f"{dataset}_{label}.csv"
            if not csv_path.exists():
                print(f"[WARN] Missing CSV: {csv_path}", file=sys.stderr)
                continue

            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    news_id = stable_news_id(row.get("id") or row.get("news_id") or f"{dataset}_{label}_{i}")
                    url = safe_str(row.get("news_url") or row.get("url")).strip()
                    title = safe_str(row.get("title")).strip()
                    records.append(
                        SourceRecord(
                            dataset=dataset,
                            label=label,
                            label_id=LABEL_ID[label],
                            news_id=news_id,
                            title=title,
                            url=url,
                            csv_path=str(csv_path.relative_to(input_root)),
                        )
                    )
    return records


def extract_top_img(url: str, timeout: int, sleep: float) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError("invalid_article_url")

    if sleep > 0:
        time.sleep(sleep)

    article = Article(url)
    article.download()
    article.parse()

    if not article.is_parsed:
        raise RuntimeError("article_parse_failed")

    top_img = safe_str(article.top_image).strip()
    if not top_img or not top_img.startswith(("http://", "https://")):
        raise RuntimeError("missing_top_img")

    return top_img


def download_image(url: str, output_dir: Path, news_id: str, timeout: int, retries: int, overwrite: bool) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}

    last_error_type = ""
    last_error_message = ""
    last_http_status = ""
    last_content_type = ""

    for _attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            last_http_status = str(response.status_code)
            last_content_type = response.headers.get("Content-Type", "")

            if response.status_code != 200:
                last_error_type = "http_error"
                last_error_message = f"HTTP status {response.status_code}"
                continue

            content_type = response.headers.get("Content-Type", "")
            if content_type and "image" not in content_type.lower():
                last_error_type = "non_image_content"
                last_error_message = f"Content-Type is {content_type}"
                continue

            ext = extension_from_response(url, content_type)
            file_path = output_dir / f"{news_id}_top_img_{short_hash(url)}{ext}"

            if file_path.exists() and not overwrite:
                return {
                    "image_path": str(file_path),
                    "http_status": last_http_status,
                    "content_type": content_type,
                    "size_bytes": file_path.stat().st_size,
                }

            size = 0
            with file_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)

            if size == 0:
                if file_path.exists():
                    file_path.unlink()
                last_error_type = "empty_file"
                last_error_message = "Downloaded file is empty"
                continue

            return {
                "image_path": str(file_path),
                "http_status": last_http_status,
                "content_type": content_type,
                "size_bytes": size,
            }

        except requests.exceptions.Timeout as exc:
            last_error_type = "timeout"
            last_error_message = str(exc)
        except requests.exceptions.RequestException as exc:
            last_error_type = "request_exception"
            last_error_message = str(exc)
        except Exception as exc:
            last_error_type = "unexpected_error"
            last_error_message = str(exc)

    raise RuntimeError(json.dumps({
        "error_type": last_error_type or "image_download_failed",
        "error_message": last_error_message,
        "http_status": last_http_status,
        "content_type": last_content_type,
    }))


def process_one(record: SourceRecord, args: argparse.Namespace, output_root: Path) -> ProcessResult:
    try:
        top_img = extract_top_img(record.url, timeout=args.timeout, sleep=args.sleep)
    except Exception as exc:
        return ProcessResult(
            status="failed",
            dataset=record.dataset,
            label=record.label,
            label_id=record.label_id,
            news_id=record.news_id,
            title=record.title,
            url=record.url,
            error_type="top_img_extraction_failed",
            error_message=str(exc),
        )

    if args.no_download_images:
        return ProcessResult(
            status="success",
            dataset=record.dataset,
            label=record.label,
            label_id=record.label_id,
            news_id=record.news_id,
            title=record.title,
            url=record.url,
            top_img=top_img,
        )

    try:
        output_dir = output_root / "images" / record.dataset / record.label
        image_info = download_image(
            top_img,
            output_dir=output_dir,
            news_id=record.news_id,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
        return ProcessResult(
            status="success",
            dataset=record.dataset,
            label=record.label,
            label_id=record.label_id,
            news_id=record.news_id,
            title=record.title,
            url=record.url,
            top_img=top_img,
            image_path=image_info["image_path"],
            http_status=image_info["http_status"],
            content_type=image_info["content_type"],
            size_bytes=image_info["size_bytes"],
        )
    except Exception as exc:
        error_type = "image_download_failed"
        error_message = str(exc)
        http_status = ""
        content_type = ""
        try:
            parsed = json.loads(str(exc))
            error_type = parsed.get("error_type", error_type)
            error_message = parsed.get("error_message", error_message)
            http_status = parsed.get("http_status", "")
            content_type = parsed.get("content_type", "")
        except Exception:
            pass

        return ProcessResult(
            status="failed",
            dataset=record.dataset,
            label=record.label,
            label_id=record.label_id,
            news_id=record.news_id,
            title=record.title,
            url=record.url,
            top_img=top_img,
            error_type=error_type,
            error_message=error_message,
            http_status=http_status,
            content_type=content_type,
        )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_outputs(results: List[ProcessResult], output_root: Path) -> None:
    metadata_dir = output_root / "metadata"
    logs_dir = output_root / "logs"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    success = [r for r in results if r.status == "success"]
    failures = [r for r in results if r.status != "success"]

    manifest_records = [
        ManifestRecord(
            id=r.news_id,
            dataset=r.dataset,
            label=r.label,
            label_id=r.label_id,
            title=r.title,
            url=r.url,
            top_img=r.top_img,
            image_path=r.image_path,
        )
        for r in success
    ]
    manifest_rows = [asdict(r) for r in manifest_records]

    news_metadata_rows = [
        asdict(
            NewsMetadataRecord(
                dataset=r.dataset,
                label=r.label,
                news_id=r.news_id,
                title=r.title,
                text="",
                url=r.url,
                publish_date="",
                source="",
                top_img=r.top_img,
                image_urls=r.top_img,
                json_path="",
                csv_path=f"{r.dataset}_{r.label}.csv",
            )
        )
        for r in success
    ]

    write_csv(metadata_dir / "manifest.csv", manifest_rows, list(asdict(ManifestRecord("", "", "", 0, "", "", "", "")).keys()))
    write_jsonl(metadata_dir / "manifest.jsonl", manifest_rows)

    write_csv(
        metadata_dir / "news_metadata.csv",
        news_metadata_rows,
        list(asdict(NewsMetadataRecord("", "", "")).keys()),
    )
    write_jsonl(metadata_dir / "news_metadata.jsonl", news_metadata_rows)

    result_fieldnames = list(asdict(ProcessResult("", "", "", 0, "", "", "")).keys())
    write_csv(logs_dir / "top_img_success.csv", [asdict(r) for r in success], result_fieldnames)
    write_csv(logs_dir / "top_img_failures.csv", [asdict(r) for r in failures], result_fieldnames)

    by_dataset_label: Dict[str, Dict[str, int]] = {}
    for r in results:
        key = f"{r.dataset}/{r.label}"
        by_dataset_label.setdefault(key, {"total": 0, "success": 0, "failed": 0})
        by_dataset_label[key]["total"] += 1
        if r.status == "success":
            by_dataset_label[key]["success"] += 1
        else:
            by_dataset_label[key]["failed"] += 1

    summary = {
        "total_records": len(results),
        "success": len(success),
        "failed": len(failures),
        "image_dir": str(output_root / "images"),
        "manifest_csv": str(metadata_dir / "manifest.csv"),
        "bridge_news_metadata_csv": str(metadata_dir / "news_metadata.csv"),
        "by_dataset_label": by_dataset_label,
    }

    with (logs_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def zip_outputs(output_root: Path, zip_name: str) -> Path:
    zip_path = output_root / f"{zip_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    include_dirs = [output_root / "images", output_root / "metadata", output_root / "logs"]
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for folder in include_dirs:
            if not folder.exists():
                continue
            for path in folder.rglob("*"):
                if path.is_file():
                    z.write(path, arcname=path.relative_to(output_root))
    return zip_path


def main() -> None:
    set_csv_field_limit()
    args = parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "images").mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)

    records = read_source_records(args.input_root, args.datasets, args.labels)
    print(f"[INFO] Source records loaded: {len(records)}")

    results: List[ProcessResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(process_one, record, args, args.output_root) for record in records]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if i % 50 == 0 or i == len(futures):
                success = sum(1 for r in results if r.status == "success")
                failed = sum(1 for r in results if r.status != "success")
                print(f"[PROCESS] {i}/{len(futures)} processed | success={success} failed={failed}")

    build_outputs(results, args.output_root)
    zip_path = zip_outputs(args.output_root, args.zip_name)

    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status != "success")
    print(f"[DONE] Success: {success}")
    print(f"[DONE] Failed: {failed}")
    print(f"[DONE] Output root: {args.output_root}")
    print(f"[DONE] Zip: {zip_path}")


if __name__ == "__main__":
    main()
