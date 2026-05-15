#!/usr/bin/env python3
"""
prepare_fakenewsnet_dataset.py

Purpose
-------
After cloning/downloading FakeNewsNet and collecting news content, this script organizes
text metadata and downloads two image versions:

1) Full image version: downloads `top_img` plus all URLs in `images`.
2) Strict top_img-only version: downloads only the URL stored in `top_img`.

Failed image URLs are recorded in logs. Both image versions are compressed into separate
zip files.

Expected input structure
------------------------
This script supports both common FakeNewsNet layouts:

1) Full collected layout, e.g.
   dataset/gossipcop/fake/<news_id>/news content.json
   dataset/gossipcop/real/<news_id>/news content.json
   dataset/politifact/fake/<news_id>/news content.json
   dataset/politifact/real/<news_id>/news content.json

2) Minimal CSV layout, e.g.
   dataset/gossipcop_fake.csv
   dataset/gossipcop_real.csv
   dataset/politifact_fake.csv
   dataset/politifact_real.csv

For image downloading, full collected layout is preferred because it usually contains
`news content.json` with `top_img` and/or `images` fields. The minimal CSV layout is
still exported as text metadata, but may not contain usable image URLs.

Example
-------
python prepare_fakenewsnet_dataset.py \
  --input-root ./dataset \
  --output-root ./prepared_fakenewsnet \
  --datasets gossipcop politifact \
  --labels fake real \
  --max-workers 8

Outputs
-------
prepared_fakenewsnet/
├── metadata/
│   ├── news_metadata.jsonl
│   └── news_metadata.csv
├── images/                         # full version: top_img + images
├── images_top_img/                 # strict top_img-only version
├── logs/
│   ├── image_download_success.csv
│   ├── image_download_failures.csv
│   ├── top_img_download_success.csv
│   ├── top_img_download_failures.csv
│   └── run_summary.json
├── fakenewsnet_images.zip
└── fakenewsnet_top_img_only.zip
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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: requests\n"
        "Install it with: pip install requests"
    ) from exc


DEFAULT_DATASETS = ("gossipcop", "politifact")
DEFAULT_LABELS = ("fake", "real")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class NewsRecord:
    dataset: str
    label: str
    news_id: str
    title: str = ""
    text: str = ""
    url: str = ""
    publish_date: str = ""
    source: str = ""
    top_img: str = ""
    image_urls: str = ""  # pipe-separated string for CSV compatibility; top_img + images
    json_path: str = ""
    csv_path: str = ""


@dataclass
class ImageJob:
    dataset: str
    label: str
    news_id: str
    image_index: int
    url: str
    output_dir: str
    image_version: str = "full"  # full / top_img_only


@dataclass
class DownloadResult:
    status: str  # success / failed / skipped
    dataset: str
    label: str
    news_id: str
    image_index: int
    url: str
    image_version: str = "full"
    file_path: str = ""
    error_type: str = ""
    error_message: str = ""
    http_status: str = ""
    content_type: str = ""
    size_bytes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Organize FakeNewsNet text metadata, download full images and strict top_img-only images, "
            "log failures, and zip both versions."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./dataset"),
        help="Path to FakeNewsNet dataset folder. Default: ./dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./prepared_fakenewsnet"),
        help="Output folder. Default: ./prepared_fakenewsnet",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=list(DEFAULT_DATASETS),
        help="Datasets to process. Default: gossipcop politifact",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        choices=list(DEFAULT_LABELS),
        help="Labels to process. Default: fake real",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel image download workers. Default: 8",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout seconds per image. Default: 20",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count per failed image. Default: 2",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds before each download request. Useful for polite crawling. Default: 0",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded image files.",
    )
    parser.add_argument(
        "--no-download-images",
        action="store_true",
        help="Only export text metadata; do not download or zip images.",
    )
    parser.add_argument(
        "--skip-full-images",
        action="store_true",
        help="Skip full image download and zip; only process strict top_img-only images.",
    )
    parser.add_argument(
        "--skip-top-img-only",
        action="store_true",
        help="Skip strict top_img-only image download and zip; only process full images.",
    )
    parser.add_argument(
        "--zip-name",
        default="fakenewsnet_images",
        help="Full image zip filename without .zip. Default: fakenewsnet_images",
    )
    parser.add_argument(
        "--top-img-zip-name",
        default="fakenewsnet_top_img_only",
        help="Strict top_img-only zip filename without .zip. Default: fakenewsnet_top_img_only",
    )
    return parser.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def normalize_url_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        urls = []
        for item in value:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict):
                for key in ("url", "src", "image", "image_url"):
                    if item.get(key):
                        urls.append(str(item[key]).strip())
                        break
        return urls
    return []


def stable_news_id(raw: str) -> str:
    raw = raw.strip()
    if raw:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:160]
    return "unknown"


def valid_http_urls(urls: Iterable[str]) -> List[str]:
    return list(dict.fromkeys([u for u in urls if u.startswith(("http://", "https://"))]))


def iter_news_json_paths(input_root: Path, datasets: Iterable[str], labels: Iterable[str]) -> Iterable[Tuple[str, str, Path]]:
    for dataset in datasets:
        for label in labels:
            base = input_root / dataset / label
            if not base.exists():
                continue
            for json_path in base.rglob("*.json"):
                if json_path.name.lower() == "news content.json":
                    yield dataset, label, json_path


def load_json_record(dataset: str, label: str, json_path: Path, input_root: Path) -> NewsRecord:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    news_id = stable_news_id(json_path.parent.name)
    top_img_urls = valid_http_urls(normalize_url_list(data.get("top_img")))
    top_img = top_img_urls[0] if top_img_urls else ""

    urls: List[str] = []
    urls.extend(top_img_urls)
    urls.extend(normalize_url_list(data.get("images")))
    urls = valid_http_urls(urls)

    return NewsRecord(
        dataset=dataset,
        label=label,
        news_id=news_id,
        title=safe_str(data.get("title", "")),
        text=safe_str(data.get("text", "")),
        url=safe_str(data.get("url", "")),
        publish_date=safe_str(data.get("publish_date", data.get("date", ""))),
        source=safe_str(data.get("source", "")),
        top_img=top_img,
        image_urls="|".join(urls),
        json_path=str(json_path.relative_to(input_root)),
    )


def iter_minimal_csv_paths(input_root: Path, datasets: Iterable[str], labels: Iterable[str]) -> Iterable[Tuple[str, str, Path]]:
    for dataset in datasets:
        for label in labels:
            csv_path = input_root / f"{dataset}_{label}.csv"
            if csv_path.exists():
                yield dataset, label, csv_path


def load_csv_records(dataset: str, label: str, csv_path: Path, input_root: Path) -> List[NewsRecord]:
    records: List[NewsRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            news_id = stable_news_id(
                row.get("id") or row.get("news_id") or row.get("article_id") or f"{dataset}_{label}_{i}"
            )
            top_img_urls = valid_http_urls(
                normalize_url_list(row.get("top_img") or row.get("image_url") or row.get("img_url") or row.get("image"))
            )
            top_img = top_img_urls[0] if top_img_urls else ""

            url_candidates: List[str] = []
            url_candidates.extend(top_img_urls)
            for key in ("images", "image_urls", "image", "image_url", "img_url"):
                url_candidates.extend(normalize_url_list(row.get(key)))
            image_urls = valid_http_urls(url_candidates)

            records.append(
                NewsRecord(
                    dataset=dataset,
                    label=label,
                    news_id=news_id,
                    title=safe_str(row.get("title", "")),
                    text=safe_str(row.get("text", row.get("news_text", ""))),
                    url=safe_str(row.get("news_url", row.get("url", ""))),
                    publish_date=safe_str(row.get("publish_date", row.get("date", ""))),
                    source=safe_str(row.get("source", "")),
                    top_img=top_img,
                    image_urls="|".join(image_urls),
                    csv_path=str(csv_path.relative_to(input_root)),
                )
            )
    return records


def collect_records(input_root: Path, datasets: Iterable[str], labels: Iterable[str]) -> List[NewsRecord]:
    records: List[NewsRecord] = []
    seen_keys = set()

    for dataset, label, json_path in iter_news_json_paths(input_root, datasets, labels):
        try:
            record = load_json_record(dataset, label, json_path, input_root)
            key = (record.dataset, record.label, record.news_id)
            seen_keys.add(key)
            records.append(record)
        except Exception as exc:
            print(f"[WARN] Failed to read JSON: {json_path} ({exc})", file=sys.stderr)

    for dataset, label, csv_path in iter_minimal_csv_paths(input_root, datasets, labels):
        try:
            for record in load_csv_records(dataset, label, csv_path, input_root):
                key = (record.dataset, record.label, record.news_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    records.append(record)
        except Exception as exc:
            print(f"[WARN] Failed to read CSV: {csv_path} ({exc})", file=sys.stderr)

    return records


def write_metadata(records: List[NewsRecord], output_root: Path) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = metadata_dir / "news_metadata.jsonl"
    csv_path = metadata_dir / "news_metadata.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    fieldnames = list(asdict(records[0]).keys()) if records else list(NewsRecord("", "", "").__dict__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


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


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def download_one(job: ImageJob, timeout: int, retries: int, sleep: float, overwrite: bool) -> DownloadResult:
    headers = {"User-Agent": USER_AGENT}
    out_dir = Path(job.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not job.url.startswith(("http://", "https://")):
        return DownloadResult(
            status="skipped",
            dataset=job.dataset,
            label=job.label,
            news_id=job.news_id,
            image_index=job.image_index,
            url=job.url,
            image_version=job.image_version,
            error_type="invalid_url",
            error_message="URL does not start with http:// or https://",
        )

    last_error = ""
    last_error_type = ""
    last_status = ""
    last_content_type = ""

    for _attempt in range(retries + 1):
        try:
            if sleep > 0:
                time.sleep(sleep)
            response = requests.get(job.url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            last_status = str(response.status_code)
            last_content_type = response.headers.get("Content-Type", "")

            if response.status_code != 200:
                last_error_type = "http_error"
                last_error = f"HTTP status {response.status_code}"
                continue

            content_type = response.headers.get("Content-Type", "")
            if content_type and "image" not in content_type.lower():
                last_error_type = "non_image_content"
                last_error = f"Content-Type is {content_type}"
                continue

            ext = extension_from_response(job.url, content_type)
            filename = f"{job.news_id}_{job.image_index:02d}_{short_hash(job.url)}{ext}"
            file_path = out_dir / filename

            if file_path.exists() and not overwrite:
                return DownloadResult(
                    status="success",
                    dataset=job.dataset,
                    label=job.label,
                    news_id=job.news_id,
                    image_index=job.image_index,
                    url=job.url,
                    image_version=job.image_version,
                    file_path=str(file_path),
                    http_status=last_status,
                    content_type=content_type,
                    size_bytes=file_path.stat().st_size,
                )

            size = 0
            with file_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        size += len(chunk)

            if size == 0:
                try:
                    file_path.unlink(missing_ok=True)
                except TypeError:
                    if file_path.exists():
                        file_path.unlink()
                last_error_type = "empty_file"
                last_error = "Downloaded file is empty"
                continue

            return DownloadResult(
                status="success",
                dataset=job.dataset,
                label=job.label,
                news_id=job.news_id,
                image_index=job.image_index,
                url=job.url,
                image_version=job.image_version,
                file_path=str(file_path),
                http_status=last_status,
                content_type=content_type,
                size_bytes=size,
            )

        except requests.exceptions.Timeout as exc:
            last_error_type = "timeout"
            last_error = str(exc)
        except requests.exceptions.RequestException as exc:
            last_error_type = "request_exception"
            last_error = str(exc)
        except Exception as exc:
            last_error_type = "unexpected_error"
            last_error = str(exc)

    return DownloadResult(
        status="failed",
        dataset=job.dataset,
        label=job.label,
        news_id=job.news_id,
        image_index=job.image_index,
        url=job.url,
        image_version=job.image_version,
        error_type=last_error_type,
        error_message=last_error,
        http_status=last_status,
        content_type=last_content_type,
    )


def build_full_image_jobs(records: List[NewsRecord], output_root: Path) -> List[ImageJob]:
    jobs: List[ImageJob] = []
    for record in records:
        urls = [u for u in record.image_urls.split("|") if u.strip()]
        for idx, url in enumerate(urls):
            jobs.append(
                ImageJob(
                    dataset=record.dataset,
                    label=record.label,
                    news_id=record.news_id,
                    image_index=idx,
                    url=url.strip(),
                    output_dir=str(output_root / "images" / record.dataset / record.label),
                    image_version="full",
                )
            )
    return jobs


def build_top_img_jobs(records: List[NewsRecord], output_root: Path) -> List[ImageJob]:
    jobs: List[ImageJob] = []
    for record in records:
        top_img = record.top_img.strip()
        if not top_img:
            continue
        jobs.append(
            ImageJob(
                dataset=record.dataset,
                label=record.label,
                news_id=record.news_id,
                image_index=0,
                url=top_img,
                output_dir=str(output_root / "images_top_img" / record.dataset / record.label),
                image_version="top_img_only",
            )
        )
    return jobs


def write_download_logs(results: List[DownloadResult], output_root: Path, success_name: str, failure_name: str) -> None:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    success_path = logs_dir / success_name
    failures_path = logs_dir / failure_name
    fieldnames = list(asdict(DownloadResult("", "", "", "", 0, "")).keys())

    with success_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result.status == "success":
                writer.writerow(asdict(result))

    with failures_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result.status != "success":
                writer.writerow(asdict(result))


def download_images(jobs: List[ImageJob], args: argparse.Namespace, label: str) -> List[DownloadResult]:
    if not jobs:
        return []

    results: List[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(download_one, job, args.timeout, args.retries, args.sleep, args.overwrite)
            for job in jobs
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if i % 50 == 0 or i == len(futures):
                success = sum(1 for r in results if r.status == "success")
                failed = sum(1 for r in results if r.status != "success")
                print(f"[DOWNLOAD:{label}] {i}/{len(futures)} processed | success={success} failed/skipped={failed}")
    return results


def zip_directory(output_root: Path, source_dir_name: str, zip_name: str) -> Optional[Path]:
    source_dir = output_root / source_dir_name
    if not source_dir.exists():
        return None
    zip_base = output_root / zip_name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=source_dir)
    return Path(zip_path)


def summarize_jobs_results(jobs: List[ImageJob], results: List[DownloadResult]) -> dict:
    return {
        "total_image_urls": len(jobs),
        "download_success": sum(1 for r in results if r.status == "success"),
        "download_failed_or_skipped": sum(1 for r in results if r.status != "success"),
    }


def write_summary(
    records: List[NewsRecord],
    full_jobs: List[ImageJob],
    full_results: List[DownloadResult],
    top_img_jobs: List[ImageJob],
    top_img_results: List[DownloadResult],
    output_root: Path,
    full_zip_path: Optional[Path],
    top_img_zip_path: Optional[Path],
) -> None:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_records": len(records),
        "records_with_image_urls": sum(1 for r in records if r.image_urls),
        "records_with_top_img": sum(1 for r in records if r.top_img),
        "full_images": {
            **summarize_jobs_results(full_jobs, full_results),
            "zip_path": str(full_zip_path) if full_zip_path else "",
            "image_dir": str(output_root / "images"),
        },
        "top_img_only": {
            **summarize_jobs_results(top_img_jobs, top_img_results),
            "zip_path": str(top_img_zip_path) if top_img_zip_path else "",
            "image_dir": str(output_root / "images_top_img"),
        },
        "by_dataset_label": {},
    }

    for record in records:
        key = f"{record.dataset}/{record.label}"
        summary["by_dataset_label"].setdefault(
            key,
            {"records": 0, "records_with_image_urls": 0, "records_with_top_img": 0},
        )
        summary["by_dataset_label"][key]["records"] += 1
        if record.image_urls:
            summary["by_dataset_label"][key]["records_with_image_urls"] += 1
        if record.top_img:
            summary["by_dataset_label"][key]["records_with_top_img"] += 1

    with (logs_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_root: Path = args.input_root
    output_root: Path = args.output_root

    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "images_top_img").mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input root: {input_root.resolve()}")
    print(f"[INFO] Output root: {output_root.resolve()}")

    records = collect_records(input_root, args.datasets, args.labels)
    print(f"[INFO] Collected news records: {len(records)}")
    write_metadata(records, output_root)
    print(f"[INFO] Metadata exported to: {output_root / 'metadata'}")

    full_jobs: List[ImageJob] = []
    top_img_jobs: List[ImageJob] = []
    full_results: List[DownloadResult] = []
    top_img_results: List[DownloadResult] = []
    full_zip_path: Optional[Path] = None
    top_img_zip_path: Optional[Path] = None

    if args.no_download_images:
        print("[INFO] --no-download-images enabled; image download and zip skipped.")
    else:
        if not args.skip_full_images:
            full_jobs = build_full_image_jobs(records, output_root)
            print(f"[INFO] Full image URLs found: {len(full_jobs)}")
            full_results = download_images(full_jobs, args, "full")
            write_download_logs(
                full_results,
                output_root,
                success_name="image_download_success.csv",
                failure_name="image_download_failures.csv",
            )
            full_zip_path = zip_directory(output_root, "images", args.zip_name)
            if full_zip_path:
                print(f"[INFO] Full images zipped to: {full_zip_path}")
        else:
            print("[INFO] --skip-full-images enabled; full image version skipped.")

        if not args.skip_top_img_only:
            top_img_jobs = build_top_img_jobs(records, output_root)
            print(f"[INFO] Strict top_img URLs found: {len(top_img_jobs)}")
            top_img_results = download_images(top_img_jobs, args, "top_img_only")
            write_download_logs(
                top_img_results,
                output_root,
                success_name="top_img_download_success.csv",
                failure_name="top_img_download_failures.csv",
            )
            top_img_zip_path = zip_directory(output_root, "images_top_img", args.top_img_zip_name)
            if top_img_zip_path:
                print(f"[INFO] Strict top_img-only images zipped to: {top_img_zip_path}")
        else:
            print("[INFO] --skip-top-img-only enabled; top_img-only version skipped.")

    write_summary(
        records,
        full_jobs,
        full_results,
        top_img_jobs,
        top_img_results,
        output_root,
        full_zip_path,
        top_img_zip_path,
    )
    print(f"[DONE] Summary written to: {output_root / 'logs' / 'run_summary.json'}")


if __name__ == "__main__":
    main()
