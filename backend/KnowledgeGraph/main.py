"""Upload JSON records to Neo4j Knowledge Graph.

Expected input record shape:
{
    "id": "tourism_196",
    "name": "Malpe Dolphin Boat Tour",
    "category": "Activity",
    "city": "Malpe",
    "state": "Karnataka",
    "tags": ["dolphin", "tour"]
}

The uploader:
- Reads from KnowledgeGraph/dataset/ folder (or specified file)
- Ingests attributes: id, name, category, city, state, tags
- Skips 'text' attribute (that's for vector DB only)
- Creates Place nodes with all attributes as properties
- Creates City nodes and LOCATED_IN relationships

Credentials loaded from backend/.env:
- KG_NEO4J_URI
- KG_NEO4J_USERNAME
- KG_NEO4J_PASSWORD

Run from project root:
    python backend/KnowledgeGraph/main.py --file backend/KnowledgeGraph/dataset/sample_records.json
    
Or to auto-detect from dataset folder:
    python backend/KnowledgeGraph/main.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_env() -> None:
    """Load .env files if python-dotenv is available."""
    if load_dotenv is None:
        return
    load_dotenv(".env", override=False)
    load_dotenv("backend/.env", override=False)


def _pick_input_file(dataset_dir: Path, explicit_file: str | None) -> Path:
    """Pick input file: explicit > first JSON in dataset dir > error."""
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


def _load_records(input_file: Path) -> list[dict[str, Any]]:
    """Load records from JSON file. Expects list or dict with 'records' key."""
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

    # Validate each record has required fields
    normalized: list[dict[str, Any]] = []
    for i, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record #{i} must be an object, got {type(record).__name__}")
        
        required_fields = {"id", "name", "category", "city", "state"}
        missing = required_fields - set(record.keys())
        if missing:
            raise ValueError(
                f"Record #{i} missing required fields: {missing}. "
                f"Required: {required_fields}. Got: {set(record.keys())}"
            )

        # Extract attributes (skip 'text', handle tags as array)
        normalized.append({
            "id": record["id"],
            "name": record["name"],
            "category": record["category"],
            "city": record["city"],
            "state": record["state"],
            "tags": record.get("tags", []),  # Default to empty array if not provided
        })

    return normalized


def _require_credentials() -> tuple[str, str, str]:
    """Load Neo4j credentials from environment."""
    uri = os.getenv("KG_NEO4J_URI")
    username = os.getenv("KG_NEO4J_USERNAME")
    password = os.getenv("KG_NEO4J_PASSWORD")

    if not uri or not username or not password:
        raise RuntimeError(
            "Missing Neo4j credentials. Set KG_NEO4J_URI, KG_NEO4J_USERNAME, and "
            "KG_NEO4J_PASSWORD in backend/.env or environment."
        )
    return uri, username, password


def _insert_place(tx, record: dict[str, Any]) -> str:
    """Insert or update a Place node with associated City."""
    result = tx.run(
        """
        MERGE (p:Place {id: $id})
        SET p.name = $name,
            p.category = $category,
            p.city = $city,
            p.state = $state,
            p.tags = $tags

        MERGE (c:City {name: $city})
        SET c.state = $state

        MERGE (p)-[:LOCATED_IN]->(c)

        RETURN p.id AS id
        """,
        id=record["id"],
        name=record["name"],
        category=record["category"],
        city=record["city"],
        state=record["state"],
        tags=record["tags"],
    )
    single_result = result.single()
    return single_result.get("id") if single_result else None


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upsert JSON records into Neo4j Knowledge Graph."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to input JSON file. If omitted, first *.json in backend/KnowledgeGraph/dataset is used.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="backend/KnowledgeGraph/dataset",
        help="Directory to scan for JSON if --file is not provided.",
    )

    args = parser.parse_args(argv)

    try:
        # Load environment
        _load_env()

        # Pick input file
        dataset_dir = Path(args.dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        input_file = _pick_input_file(dataset_dir, args.file)
        logger.info(f"Reading from: {input_file}")

        # Load records
        records = _load_records(input_file)
        logger.info(f"Loaded {len(records)} place records")

        # Get credentials
        uri, username, password = _require_credentials()

        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))
        try:
            driver.verify_connectivity()
            logger.info("✅ Connected to Neo4j Aura")
        except ServiceUnavailable as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return 1

        # Upload records
        inserted_count = 0
        try:
            with driver.session() as session:
                for i, record in enumerate(records, start=1):
                    try:
                        place_id = session.execute_write(_insert_place, record)
                        inserted_count += 1
                        if i % 50 == 0 or i == len(records):
                            logger.info(f"  [{i}/{len(records)}] Processed: {record['name']}")
                    except Exception as e:
                        logger.error(f"  Failed to insert record #{i} ({record.get('id')}): {e}")
                        raise

            logger.info(f"✅ {inserted_count}/{len(records)} records uploaded successfully!")
            return 0

        finally:
            driver.close()

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return 1
    except ValueError as e:
        logger.error(f"❌ Invalid input: {e}")
        return 1
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
