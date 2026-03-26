"""Upload JSON records to Pinecone integrated-embedding index.

Expected input record shape (text-focused):
{
    "text": "the quick brown fox jumped over the lazy dog"
}

The uploader auto-generates sequential ids and upserts records as:
{
    "id": "tourism_001",
    "text": "..."
}

Defaults target your index details:
- index: major-project
- host: https://major-project-br8qn31.svc.aped-4627-b74a.pinecone.io
- model: llama-text-embed-v2 (configured on Pinecone index)
- embed field map: text

Run from project root:
    ./venv/bin/python vectordb_uploader/main.py --file vectordb_uploader/dataset/sample_records.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pinecone import Pinecone  # type: ignore[import]

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

DEFAULT_INDEX_NAME = "major-project"
DEFAULT_HOST = "https://major-project-br8qn31.svc.aped-4627-b74a.pinecone.io"
DEFAULT_NAMESPACE = "tourism"
DEFAULT_BATCH_SIZE = 96
PINECONE_MAX_BATCH_SIZE = 96


def _load_env() -> None:
    """Load .env files if python-dotenv is available."""
    if load_dotenv is None:
        return
    load_dotenv(".env", override=False)
    load_dotenv("backend/.env", override=False)


def _pick_input_file(dataset_dir: Path, explicit_file: str | None) -> Path:
    if explicit_file:
        path = Path(explicit_file)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path

    candidates = sorted(dataset_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No JSON file found in dataset directory: {dataset_dir}. "
            "Add a JSON file or pass --file."
        )
    return candidates[0]


def _load_records(input_file: Path, id_prefix: str, start_id: int) -> list[dict[str, Any]]:
    with input_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            records = payload["records"]
        else:
            raise ValueError("JSON object input must include a list under key 'records'.")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Input JSON must be either a list of records or an object with a 'records' list.")

    normalized: list[dict[str, Any]] = []
    for i, record in enumerate(records, start=1):
        text: str | None = None

        if isinstance(record, dict):
            text_value = record.get("text")
            if isinstance(text_value, str) and text_value.strip():
                text = text_value.strip()
        elif isinstance(record, str) and record.strip():
            text = record.strip()
        else:
            raise ValueError(f"Record #{i} must be an object with 'text' or a non-empty string.")

        if not text:
            raise ValueError(f"Record #{i} has invalid or missing 'text'.")

        generated_id = f"{id_prefix}{start_id + i - 1:03d}"
        normalized.append({"id": generated_id, "text": text})

    return normalized


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _require_api_key() -> str:
    api_key = os.getenv("VECTORDB_PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Pinecone API key. Set VECTORDB_PINECONE_API_KEY (or PINECONE_API_KEY) in environment/.env."
        )
    return api_key


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upsert JSON records into Pinecone integrated-embedding index.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to input JSON file. If omitted, first *.json in vectordb_uploader/dataset is used.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="vectordb_uploader/dataset",
        help="Directory to scan for JSON if --file is not provided.",
    )
    parser.add_argument("--index", type=str, default=DEFAULT_INDEX_NAME, help="Pinecone index name.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Pinecone index host URL.")
    parser.add_argument("--namespace", type=str, default=DEFAULT_NAMESPACE, help="Namespace for records.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Upsert batch size.")
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="tourism_",
        help="Prefix for generated sequential record ids.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Starting number for generated sequential record ids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print what would be uploaded without writing to Pinecone.",
    )

    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        print("ERROR: --batch-size must be > 0")
        return 2
    if args.start_id <= 0:
        print("ERROR: --start-id must be > 0")
        return 2

    # Pinecone integrated embedding endpoint rejects batch sizes above 96.
    effective_batch_size = min(args.batch_size, PINECONE_MAX_BATCH_SIZE)
    if effective_batch_size != args.batch_size:
        print(
            f"WARNING: Requested batch size {args.batch_size} exceeds Pinecone max {PINECONE_MAX_BATCH_SIZE}. "
            f"Using {effective_batch_size}."
        )

    _load_env()

    try:
        input_file = _pick_input_file(Path(args.dataset_dir), args.file)
        records = _load_records(input_file, id_prefix=args.id_prefix, start_id=args.start_id)
    except Exception as exc:
        print(f"ERROR: Failed to load input records: {exc}")
        return 1

    print(f"Input file: {input_file}")
    print(f"Records loaded: {len(records)}")
    if records:
        print(f"Generated id range: {records[0]['id']} .. {records[-1]['id']}")
    print(f"Index: {args.index}")
    print(f"Host: {args.host}")
    print(f"Namespace: {args.namespace}")
    print("Integrated embedding field map: text (metadata dropped by design)")

    if args.dry_run:
        print("Dry run enabled. No data was uploaded.")
        return 0

    try:
        api_key = _require_api_key()
        pc = Pinecone(api_key=api_key)
        index = pc.Index(name=args.index, host=args.host)

        # Contactability check before upsert.
        stats = index.describe_index_stats()
        print("Connected to Pinecone index successfully.")
        print(f"Current index stats: {stats}")

        total_upserted = 0
        batches = _chunk(records, effective_batch_size)
        for batch_no, batch in enumerate(batches, start=1):
            if hasattr(index, "upsert_records"):
                response = index.upsert_records(namespace=args.namespace, records=batch)
            else:
                raise RuntimeError(
                    "This pinecone SDK/index client does not expose upsert_records for integrated embedding indexes. "
                    "Upgrade pinecone package to a version that supports upsert_records."
                )

            # Response shape may vary by SDK version.
            upserted_count = 0
            if isinstance(response, dict):
                upserted_count = int(response.get("upserted_count", 0))
            else:
                upserted_count = len(batch)
            total_upserted += upserted_count
            print(f"Batch {batch_no}/{len(batches)} uploaded. Batch size={len(batch)} upserted={upserted_count}")

        print(f"Upload complete. Total records upserted: {total_upserted}")
        return 0

    except Exception as exc:
        print(f"ERROR: Upload failed: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
