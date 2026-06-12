"""
ingestion/hypersync_client.py
──────────────────────────────
Fetches raw Mint/Burn/Swap/Collect/Initialize logs + block headers
for a single V3 pool via the HyperSync API and writes Parquet to:

    data/raw/{chain}/{pool_address}/logs.parquet
    data/raw/{chain}/{pool_address}/blocks.parquet

Re-runs are incremental: watermark tracked in
    data/raw/{chain}/{pool_address}/.watermark.json

Output convention matches FDP bronze layer:
  - All columns stored as str (TEXT equivalent) — avoids uint256 overflow.
  - Timestamps remain 0x-prefixed hex (same as FDP; metrics.py handles them).
  - Append semantics: watermark prevents duplicate block ranges.

On GCP this function's body stays the same; only the write target changes
(GCS bucket path instead of local data/raw/).
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import hypersync
from hypersync import BlockField, HexOutput, LogField, LogSelection

from features._env import setup

setup()


def _append_parquet(df: pd.DataFrame, path: Path) -> None:
    """Append df to an existing Parquet file, or create it if absent."""
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(path, index=False)


async def fetch(chain: str, pool_address: str, data_dir: Path) -> dict:
    """
    Fetch raw logs + blocks for one pool.

    Parameters
    ----------
    chain         : HyperSync chain slug  (e.g. "celo", "hyperevm")
    pool_address  : Pool contract address (checksum or lowercase)
    data_dir      : Root data directory   (e.g. Path("data"))

    Returns
    -------
    dict with keys "logs" and "blocks" (row counts written this run).
    Returns {"logs": 0, "blocks": 0} when already up to date.
    """
    pool_lower = pool_address.lower()
    raw_dir = data_dir / "raw" / chain / pool_lower
    raw_dir.mkdir(parents=True, exist_ok=True)

    watermark_path = raw_dir / ".watermark.json"
    from_block = 0
    if watermark_path.exists():
        with open(watermark_path) as f:
            from_block = json.load(f).get("block", 0) + 1

    client = hypersync.HypersyncClient(
        hypersync.ClientConfig(
            url=f"https://{chain}.hypersync.xyz",
            bearer_token=os.getenv("HYPERSYNC_BEARER_TOKEN"),
        )
    )

    height = await client.get_height()
    if from_block > height:
        print(f"  Already at block {height}. Nothing to fetch.")
        return {"logs": 0, "blocks": 0}

    print(f"  Fetching blocks {from_block} -> {height} ...")

    field_selection = hypersync.FieldSelection(
        block=[BlockField.NUMBER, BlockField.TIMESTAMP],
        log=[
            LogField.BLOCK_NUMBER,
            LogField.TRANSACTION_HASH,
            LogField.TRANSACTION_INDEX,
            LogField.LOG_INDEX,
            LogField.DATA,
            LogField.ADDRESS,
            LogField.TOPIC0,
            LogField.TOPIC1,
            LogField.TOPIC2,
            LogField.TOPIC3,
        ],
    )

    query = hypersync.Query(
        from_block=from_block,
        to_block=height,
        field_selection=field_selection,
        logs=[LogSelection(address=[pool_address], topics=[])],
    )
    config = hypersync.StreamConfig(hex_output=HexOutput.PREFIXED)

    with tempfile.TemporaryDirectory() as tmp:
        await client.collect_parquet(tmp, query, config)

        logs_df = pq.read_table(f"{tmp}/logs.parquet").to_pandas()
        blocks_df = pq.read_table(f"{tmp}/blocks.parquet").to_pandas()

    # Normalise column names
    logs_df.columns = [c.lower() for c in logs_df.columns]
    blocks_df.columns = [c.lower() for c in blocks_df.columns]

    # Tag with chain + pool (mirrors FDP bronze convention)
    logs_df["chain_name"] = chain
    logs_df["pool_address"] = pool_lower
    blocks_df["chain_name"] = chain

    # Cast everything to str — avoids uint256/int256 overflow, matches FDP TEXT
    logs_df = logs_df.astype(str)
    blocks_df = blocks_df.astype(str)

    _append_parquet(logs_df, raw_dir / "logs.parquet")
    _append_parquet(blocks_df, raw_dir / "blocks.parquet")

    with open(watermark_path, "w") as f:
        json.dump({"block": height}, f)

    print(f"  Wrote {len(logs_df)} log rows, {len(blocks_df)} block rows. Watermark -> {height}.")
    return {"logs": len(logs_df), "blocks": len(blocks_df)}
