"""
features/pipeline.py
─────────────────────
Orchestrates the full feature engineering run for one pool.

Reads from PostgreSQL (raw schema — already populated by financial-data-platform
ingestion), calls features/metrics.py functions in dependency order, and writes
three Parquet files to data/ for Hugo's Day-1 training jobs.

Output files:
    data/lp_features.parquet         one row per position_id (at-entry features + labels)
    data/lp_survival_labels.parquet  position_id, duration_seconds, status, exit_type
    data/lp_event_sequences.parquet  long table keyed by position_id + seq_num

Usage:
    python -m features.pipeline --pool 0xF55791... --chain celo

Will be refactored into a PySpark job (Dataproc) for multi-pool fan-out.
The pandas→Spark translation is 1-to-1: each function in metrics.py is
pool-independent and stateless.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from features.metrics import event_sequence, exit_type, verify_lp_exit

load_dotenv()

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
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"chain": chain, "pool": pool.lower()})


def run(chain: str, pool: str) -> None:
    engine = _get_engine()
    pool = pool.lower()

    print(f"Loading raw events for {pool} on {chain}…")
    mint_df = _load_table(engine, "lp_mint_events", chain, pool)
    burn_df = _load_table(engine, "lp_burn_events", chain, pool)
    swap_df = _load_table(engine, "lp_swap_events", chain, pool)
    collect_df = _load_table(engine, "lp_collect_events", chain, pool)

    campaigns_q = text(
        "SELECT * FROM raw.merkl_campaigns WHERE chain_name = :chain AND pool_address = :pool"
    )
    with engine.connect() as conn:
        campaigns_df = pd.read_sql(campaigns_q, conn, params={"chain": chain, "pool": pool})

    print("Building lp_summary…")
    lp_summary = verify_lp_exit(mint_df, burn_df, campaigns_df)

    print("Classifying exit types…")
    lp_summary = exit_type(lp_summary, swap_df)

    print("Building event sequences…")
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
    print(f"  → data/lp_features.parquet ({len(lp_features)} rows)")

    survival_cols = ["position_id", "duration_seconds", "status", "exit_type"]
    lp_summary[survival_cols].to_parquet(DATA_DIR / "lp_survival_labels.parquet", index=False)
    print(f"  → data/lp_survival_labels.parquet")

    sequences.to_parquet(DATA_DIR / "lp_event_sequences.parquet", index=False)
    print(f"  → data/lp_event_sequences.parquet ({len(sequences)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LPulse feature pipeline")
    parser.add_argument("--pool", required=True, help="Pool contract address")
    parser.add_argument("--chain", required=True, help="Chain name (e.g. celo)")
    args = parser.parse_args()
    run(chain=args.chain, pool=args.pool)


if __name__ == "__main__":
    main()
