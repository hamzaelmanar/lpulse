"""
ingestion/ingest.py
────────────────────
CLI orchestrator: fetch raw events via HyperSync, then decode.

Usage — single pool:
    python -m ingestion.ingest --chain celo --pool 0xF55791...

Usage — all pools in registry:
    python -m ingestion.ingest --all

Usage — specific pool from registry by address (no need to type chain):
    python -m ingestion.ingest --pool 0xF55791...

Optional flags:
    --data-dir  path/to/data   (default: data/)
    --skip-fetch               only run decode step (raw already fetched)
    --skip-decode              only run fetch step

After a successful run, decoded Parquet files are ready for the feature pipeline:
    python -m features.pipeline --chain ... --pool ... --merkl-url ... --source parquet
"""

import argparse
import asyncio
from pathlib import Path

import yaml

from features._env import setup
from ingestion.hypersync_client import fetch
from ingestion.decode_events import decode

setup()

REGISTRY_PATH = Path(__file__).parent / "pools_registry.yaml"


def _load_registry():
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)["pools"]


def _run_pool(chain: str, pool: str, data_dir: Path, skip_fetch: bool, skip_decode: bool) -> None:
    print(f"\n[{chain}] {pool}")
    if not skip_fetch:
        print("  Step 1/2 — fetch raw events from HyperSync")
        result = asyncio.run(fetch(chain=chain, pool_address=pool, data_dir=data_dir))
        print(f"  Fetched: {result}")
    else:
        print("  Step 1/2 — fetch skipped (--skip-fetch)")

    if not skip_decode:
        print("  Step 2/2 — decode events to Parquet")
        counts = decode(chain=chain, pool_address=pool, data_dir=data_dir)
        total = sum(counts.values())
        print(f"  Decoded: {total} events across {len([k for k,v in counts.items() if v > 0])} event types")
    else:
        print("  Step 2/2 — decode skipped (--skip-decode)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LPulse ingestion — fetch + decode pool events")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Ingest all pools in pools_registry.yaml")
    group.add_argument("--pool", help="Pool contract address (single pool)")
    parser.add_argument("--chain", help="Chain slug, required when --pool is given without --all")
    parser.add_argument("--data-dir", default="data", help="Root data directory (default: data/)")
    parser.add_argument("--skip-fetch",  action="store_true", help="Skip HyperSync fetch step")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode step")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.all:
        pools = _load_registry()
        print(f"Ingesting {len(pools)} pool(s) from registry ...")
        for entry in pools:
            _run_pool(
                chain=entry["chain"],
                pool=entry["pool_address"],
                data_dir=data_dir,
                skip_fetch=args.skip_fetch,
                skip_decode=args.skip_decode,
            )
        print("\nDone.")
        return

    if args.pool:
        chain = args.chain
        if not chain:
            # Look up chain from registry
            pools = _load_registry()
            match = [p for p in pools if p["pool_address"].lower() == args.pool.lower()]
            if not match:
                parser.error("--chain is required when --pool is not in pools_registry.yaml")
            chain = match[0]["chain"]
        _run_pool(
            chain=chain,
            pool=args.pool,
            data_dir=data_dir,
            skip_fetch=args.skip_fetch,
            skip_decode=args.skip_decode,
        )
        print("\nDone.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
