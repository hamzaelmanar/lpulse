"""
features/pipeline.py
─────────────────────
Orchestrates the full feature engineering run for one pool.

Reads from either:
  - PostgreSQL (default, --source postgres) — populated by FDP ingestion
  - Local Parquet  (--source parquet)       — populated by ingestion/ingest.py

Calls features/metrics.py functions in dependency order and writes three
Parquet files to data/ for Hugo's Day-1 training jobs.

Output files:
    data/lp_features.parquet         one row per position_id (at-entry features + labels)
    data/lp_survival_labels.parquet  position_id, duration_seconds, status, exit_type
    data/lp_event_sequences.parquet  long table keyed by position_id + seq_num

Usage (Postgres source — existing FDP data):
    python -m features.pipeline \
        --chain celo --pool 0xF557... --merkl-url "..."

Usage (Parquet source — after running ingestion/ingest.py):
    python -m features.pipeline \
        --chain celo --pool 0xF557... --merkl-url "..." --source parquet

Will be refactored into a PySpark job (Dataproc) for multi-pool fan-out.
The pandas→Spark translation is 1-to-1: each function in metrics.py is
pool-independent and stateless.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from features._env import setup
from features.merkl import fetch_campaign_windows
from features.metrics import event_sequence, exit_type, verify_lp_exit

setup()

DATA_DIR = Path(__file__).parent.parent / "data"


def _get_engine():
    url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url, future=True)


def _load_table(engine, table: str, chain: str, pool: str) -> pd.DataFrame:
    q = text(f"SELECT * FROM raw.{table} WHERE chain_name = :chain AND pool_address = :pool")
    try:
        with engine.connect() as conn:
            return pd.read_sql(q, conn, params={"chain": chain, "pool": pool.lower()})
    except Exception as exc:
        if "UndefinedTable" in type(exc).__name__ or "does not exist" in str(exc) or "n'existe pas" in str(exc):
            print(f"  Warning: raw.{table} not found -- returning empty DataFrame.")
            return pd.DataFrame()
        raise


def _load_table_from_parquet(data_dir: Path, chain: str, pool: str, table: str) -> pd.DataFrame:
    """Read a decoded event Parquet written by ingestion/decode_events.py.

    Returns an empty DataFrame (not an error) when the file doesn't exist —
    same contract as _load_table() for missing tables (e.g. lp_collect_events).
    """
    path = data_dir / "decoded" / chain / pool.lower() / f"{table}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    print(f"  Warning: {path} not found -- returning empty DataFrame.")
    return pd.DataFrame()


def run(chain: str, pool: str, merkl_url: str, source: str = "postgres") -> None:
    pool = pool.lower()

    print(f"Fetching campaign windows from Merkl API...")
    campaigns_df = fetch_campaign_windows(merkl_url)
    print(f"  {len(campaigns_df)} campaign(s) found.")

    print(f"Loading raw events for {pool} on {chain} (source={source})...")
    if source == "parquet":
        mint_df    = _load_table_from_parquet(DATA_DIR, chain, pool, "lp_mint_events")
        burn_df    = _load_table_from_parquet(DATA_DIR, chain, pool, "lp_burn_events")
        swap_df    = _load_table_from_parquet(DATA_DIR, chain, pool, "lp_swap_events")
        collect_df = _load_table_from_parquet(DATA_DIR, chain, pool, "lp_collect_events")
    else:
        engine = _get_engine()
        mint_df     = _load_table(engine, "lp_mint_events",    chain, pool)
        burn_df     = _load_table(engine, "lp_burn_events",    chain, pool)
        swap_df     = _load_table(engine, "lp_swap_events",    chain, pool)
        collect_df  = _load_table(engine, "lp_collect_events", chain, pool)
    print(
        f"  Mints: {len(mint_df)}  Burns: {len(burn_df)}  "
        f"Swaps: {len(swap_df)}  Collects: {len(collect_df)}"
    )

    print("Building lp_summary (verify_lp_exit)...")
    lp_summary = verify_lp_exit(mint_df, burn_df, campaigns_df)
    print(f"  {len(lp_summary)} positions reconstructed.")

    print("Classifying exit types...")
    lp_summary = exit_type(lp_summary, swap_df)

    print("Building event sequences...")
    sequences = event_sequence(mint_df, burn_df, collect_df, swap_df, lp_summary)

    DATA_DIR.mkdir(exist_ok=True)

    at_entry_cols = [
        "position_id", "owner", "pool_address", "chain_name",
        "tick_lower", "tick_upper", "tick_range_width",
        "first_mint_timestamp", "first_mint_tx_hash",
        "duration_seconds", "status", "exit_type", "lp_cohort",
        "event_count",
    ]
    lp_features = lp_summary[at_entry_cols]
    lp_features.to_parquet(DATA_DIR / "lp_features.parquet", index=False)

    survival_cols = ["position_id", "duration_seconds", "status", "exit_type"]
    lp_summary[survival_cols].to_parquet(DATA_DIR / "lp_survival_labels.parquet", index=False)

    sequences.to_parquet(DATA_DIR / "lp_event_sequences.parquet", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_exited   = int((lp_summary["status"] == 1).sum())
    n_censored = int((lp_summary["status"] == 0).sum())
    exit_dist  = lp_summary["exit_type"].value_counts().to_dict()
    cohort_dist = lp_summary["lp_cohort"].value_counts().to_dict()

    print("\n" + "-" * 60)
    print(f"  pool             {pool}")
    print(f"  chain            {chain}")
    print(f"  positions        {len(lp_summary):,}  ({n_exited:,} exited, {n_censored:,} censored)")
    print(f"  exit_type        {exit_dist}")
    print(f"  lp_cohort        {cohort_dist}")
    print(f"  sequence rows    {len(sequences):,}")
    print("-" * 60)
    print(f"  -> data/lp_features.parquet         ({len(lp_features):,} rows)")
    print(f"  -> data/lp_survival_labels.parquet  ({len(lp_summary):,} rows)")
    print(f"  -> data/lp_event_sequences.parquet  ({len(sequences):,} rows)")
    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="LPulse feature pipeline")
    parser.add_argument("--pool",      required=True, help="Pool contract address")
    parser.add_argument("--chain",     required=True, help="Chain name (e.g. celo)")
    parser.add_argument("--merkl-url", required=True,
                        help="Merkl opportunity URL for campaign window lookup")
    parser.add_argument("--source", choices=["postgres", "parquet"], default="postgres",
                        help="Event data source: postgres (default, FDP) or parquet (ingestion/ingest.py output)")
    args = parser.parse_args()
    run(chain=args.chain, pool=args.pool, merkl_url=args.merkl_url, source=args.source)


if __name__ == "__main__":
    main()
