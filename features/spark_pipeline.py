"""
features/spark_pipeline.py
───────────────────────────
PySpark port of features/pipeline.py.

Reads decoded event Parquet from data/decoded/{chain}/{pool}/
(written by ingestion/ingest.py) and writes three output Parquet files to
data/{chain}/{pool}/ — same paths as the pandas pipeline.

Key design decisions vs pandas pipeline:
  - verify_lp_exit(), exit_type(), event_sequence() logic is re-implemented
    using native Spark DataFrame operations (no pandas UDFs — avoids
    serialization overhead and is Dataproc-portable).
  - merge_asof (find last swap tick before each event) is implemented via
    a Spark window function: last non-null value ordered by timestamp.
  - hex timestamp conversion handled by a Spark UDF wrapping _hex_or_dec_to_int.
  - Output: single-file Parquet per table (coalesce(1)) for local use.
    On Dataproc, remove coalesce() to let Spark write partitioned output.

Local performance note:
  Spark adds ~15-20s JVM startup overhead vs pandas. The per-pool compute
  time is similar or slightly slower locally. The value is pool-level
  parallelism on Dataproc (N pools as N Spark tasks on a cluster).

Usage:
    python -m features.spark_pipeline \\
        --chain celo \\
        --pool 0xF55791... \\
        --merkl-url "https://app.merkl.xyz/..." \\
        [--master local[*]]

Requires: pyspark, Java 8/11/17 on PATH or JAVA_HOME set.
"""

import argparse
import hashlib
import os
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType

from features._env import setup
from features.merkl import fetch_campaign_windows

setup()

BASE_DATA_DIR = Path(__file__).parent.parent / "data"

# ── Spark session ──────────────────────────────────────────────────────────────

def _get_spark(master: str = "local[*]") -> SparkSession:
    return (
        SparkSession.builder
        .master(master)
        .appName("lpulse-feature-pipeline")
        .config("spark.sql.shuffle.partitions", "8")   # small for local
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


# ── UDFs (mirrors metrics.py helpers) ─────────────────────────────────────────

@F.udf(returnType=LongType())
def _udf_hex_to_int(x):
    if x is None:
        return None
    x = str(x).strip()
    if x.startswith("0x") or x.startswith("0X"):
        return int(x, 16)
    try:
        return int(x)
    except ValueError:
        return None


@F.udf(returnType=IntegerType())
def _udf_int_col(x):
    """Safe int cast for tick / amount columns stored as decimal strings."""
    if x is None:
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


@F.udf(returnType=StringType())
def _udf_position_id(chain, pool, owner, tick_lower, tick_upper, first_mint_tx):
    if any(v is None for v in [chain, pool, owner, tick_lower, tick_upper, first_mint_tx]):
        return None
    key = f"{chain}|{pool}|{owner}|{tick_lower}|{tick_upper}|{first_mint_tx}"
    return hashlib.md5(key.encode()).hexdigest()


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _read_events(spark: SparkSession, chain: str, pool: str, table: str):
    path = BASE_DATA_DIR / "decoded" / chain / pool / f"{table}.parquet"
    if not path.exists():
        print(f"  Warning: {path} not found -- returning empty DataFrame.")
        return spark.createDataFrame([], schema="block_number STRING")
    return spark.read.parquet(str(path))


# ── verify_lp_exit (Spark) ─────────────────────────────────────────────────────

def _verify_lp_exit_spark(spark, mint_df, burn_df, campaigns_df, chain: str, pool: str):
    """
    Reconstruct LP position cycles. Logic mirrors metrics.verify_lp_exit().

    Returns a Spark DataFrame with one row per position cycle.
    """
    # Cast timestamp columns
    mint = mint_df.withColumn("ts",     _udf_hex_to_int(F.col("timestamp"))) \
                  .withColumn("tick_lower", _udf_int_col(F.col("tick_lower"))) \
                  .withColumn("tick_upper", _udf_int_col(F.col("tick_upper"))) \
                  .withColumn("amount",     F.col("amount").cast("decimal(38,0)")) \
                  .withColumn("owner",      F.lower(F.col("owner"))) \
                  .withColumn("pool_address", F.lower(F.col("pool_address")))

    burn = burn_df.withColumn("ts",     _udf_hex_to_int(F.col("timestamp"))) \
                  .withColumn("tick_lower", _udf_int_col(F.col("tick_lower"))) \
                  .withColumn("tick_upper", _udf_int_col(F.col("tick_upper"))) \
                  .withColumn("amount",     F.col("amount").cast("decimal(38,0)")) \
                  .withColumn("owner",      F.lower(F.col("owner"))) \
                  .withColumn("pool_address", F.lower(F.col("pool_address")))

    # Combine mint (+amount) and burn (-amount) to compute cumulative liquidity
    mint_tagged = mint.withColumn("liq_delta", F.col("amount"))
    burn_tagged = burn.withColumn("liq_delta", F.col("amount").cast("decimal(38,0)") * -1)

    events = mint_tagged.unionByName(burn_tagged, allowMissingColumns=True) \
                        .select("chain_name", "pool_address", "owner",
                                "tick_lower", "tick_upper", "ts",
                                "transaction_hash", "liq_delta")

    # Cumulative liquidity per identity key
    id_keys  = ["chain_name", "pool_address", "owner", "tick_lower", "tick_upper"]
    ord_win  = Window.partitionBy(*id_keys).orderBy("ts", "transaction_hash")
    cum_win  = Window.partitionBy(*id_keys).orderBy("ts", "transaction_hash") \
                     .rowsBetween(Window.unboundedPreceding, 0)

    events = events.withColumn("cum_liq", F.sum("liq_delta").over(cum_win))

    # Detect cycle boundaries: cum_liq drops to 0 → position closed
    events = events.withColumn("prev_cum", F.lag("cum_liq").over(ord_win))
    events = events.withColumn("is_open",
        (F.col("prev_cum").isNull()) |
        (F.col("prev_cum") <= 0)
    )
    events = events.withColumn("cycle_id",
        F.sum(F.col("is_open").cast("int")).over(ord_win)
    )

    cycle_keys = id_keys + ["cycle_id"]

    # First mint in each cycle
    first_mint = mint_tagged.withColumn("ts", _udf_hex_to_int(F.col("timestamp"))) \
                            .withColumn("tick_lower", _udf_int_col(F.col("tick_lower"))) \
                            .withColumn("tick_upper", _udf_int_col(F.col("tick_upper"))) \
                            .withColumn("owner", F.lower(F.col("owner"))) \
                            .withColumn("pool_address", F.lower(F.col("pool_address")))

    cycle_events = events.join(
        first_mint.select(*id_keys, "ts", "transaction_hash"),
        on=id_keys, how="left"
    )

    # Aggregate per cycle
    agg = events.groupBy(*cycle_keys).agg(
        F.min("ts").alias("first_mint_timestamp"),
        F.max(F.when(F.col("liq_delta") < 0, F.col("ts"))).alias("exit_timestamp"),
        F.count("*").alias("event_count"),
    )

    agg = agg.withColumn("status",
        F.when(F.col("exit_timestamp").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )
    agg = agg.withColumn("duration_seconds",
        F.when(
            F.col("status") == 1,
            F.col("exit_timestamp") - F.col("first_mint_timestamp")
        ).otherwise(F.lit(None).cast(LongType()))
    )
    agg = agg.withColumn("tick_range_width",
        F.col("tick_upper") - F.col("tick_lower")
    )

    # Campaign cohort assignment
    if not campaigns_df.empty:
        global_start = int(campaigns_df["start_timestamp"].min())
        global_end   = int(campaigns_df["end_timestamp"].max())
    else:
        global_start = global_end = 0

    agg = agg.withColumn("lp_cohort",
        F.when(
            (F.col("first_mint_timestamp") >= global_start) &
            (F.col("first_mint_timestamp") <= global_end),
            F.lit("during_campaign")
        ).when(F.col("first_mint_timestamp") < global_start, F.lit("pre_campaign"))
         .when(F.col("first_mint_timestamp") > global_end,   F.lit("post_campaign"))
         .otherwise(F.lit("unknown"))
    )

    # position_id: use first mint tx hash via join
    first_tx = events.filter(F.col("is_open")).groupBy(*cycle_keys) \
                     .agg(F.first("transaction_hash").alias("first_mint_tx_hash"))

    agg = agg.join(first_tx, on=cycle_keys, how="left")
    agg = agg.withColumn("position_id",
        _udf_position_id(
            F.col("chain_name"), F.col("pool_address"), F.col("owner"),
            F.col("tick_lower").cast("string"), F.col("tick_upper").cast("string"),
            F.col("first_mint_tx_hash"),
        )
    )

    return agg


# ── exit_type (Spark) ──────────────────────────────────────────────────────────

def _exit_type_spark(lp_summary, swap_df):
    """Classify exits as voluntary_exit, range_exit, or censored."""
    swap = swap_df.withColumn("swap_ts", _udf_hex_to_int(F.col("timestamp"))) \
                  .withColumn("tick_int", _udf_int_col(F.col("tick"))) \
                  .withColumn("pool_address", F.lower(F.col("pool_address")))

    exited = lp_summary.filter(F.col("status") == 1)

    # Last swap tick before exit: window ordered by swap_ts, join on pool
    swap_latest = swap.select("pool_address", "swap_ts", "tick_int")

    joined = exited.join(swap_latest, on="pool_address", how="left") \
                   .filter(F.col("swap_ts") <= F.col("exit_timestamp"))

    w = Window.partitionBy("position_id").orderBy(F.desc("swap_ts"))
    last_tick = joined.withColumn("rn", F.row_number().over(w)) \
                      .filter(F.col("rn") == 1) \
                      .select("position_id", "tick_int")

    exited = exited.join(last_tick, on="position_id", how="left")
    exited = exited.withColumn("exit_type",
        F.when(F.col("tick_int").isNull(), F.lit("voluntary_exit"))
         .when(
             (F.col("tick_int") >= F.col("tick_lower")) &
             (F.col("tick_int") <= F.col("tick_upper")),
             F.lit("voluntary_exit")
         ).otherwise(F.lit("range_exit"))
    ).drop("tick_int")

    censored = lp_summary.filter(F.col("status") == 0) \
                         .withColumn("exit_type", F.lit("censored"))

    return exited.unionByName(censored)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(chain: str, pool: str, merkl_url: str, master: str = "local[*]") -> Path:
    spark = _get_spark(master)
    pool  = pool.lower()
    out_dir = BASE_DATA_DIR / chain / pool
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching campaign windows from Merkl API...")
    campaigns_df = fetch_campaign_windows(merkl_url)
    print(f"  {len(campaigns_df)} campaign(s) found.")

    print(f"Loading decoded Parquet for {pool} on {chain}...")
    mint_df    = _read_events(spark, chain, pool, "lp_mint_events")
    burn_df    = _read_events(spark, chain, pool, "lp_burn_events")
    swap_df    = _read_events(spark, chain, pool, "lp_swap_events")
    collect_df = _read_events(spark, chain, pool, "lp_collect_events")

    print("Building lp_summary (verify_lp_exit)...")
    lp_summary = _verify_lp_exit_spark(spark, mint_df, burn_df, campaigns_df, chain, pool)

    print("Classifying exit types...")
    lp_summary = _exit_type_spark(lp_summary, swap_df)
    lp_summary.cache()

    n_total    = lp_summary.count()
    n_exited   = lp_summary.filter(F.col("status") == 1).count()
    n_censored = lp_summary.filter(F.col("status") == 0).count()

    print(f"  {n_total:,} positions ({n_exited:,} exited, {n_censored:,} censored).")

    # ── Write outputs ──────────────────────────────────────────────────────────
    at_entry_cols = [
        "position_id", "owner", "pool_address", "chain_name",
        "tick_lower", "tick_upper", "tick_range_width",
        "first_mint_timestamp", "first_mint_tx_hash",
        "duration_seconds", "status", "exit_type", "lp_cohort", "event_count",
    ]
    # Write via pandas for single-file output (avoids Spark part-00000 naming)
    lp_features = lp_summary.select(
        *[c for c in at_entry_cols if c in lp_summary.columns]
    ).toPandas()
    lp_features.to_parquet(out_dir / "lp_features.parquet", index=False)

    survival_cols = ["position_id", "duration_seconds", "status", "exit_type"]
    lp_summary.select(
        *[c for c in survival_cols if c in lp_summary.columns]
    ).toPandas().to_parquet(out_dir / "lp_survival_labels.parquet", index=False)

    print("-" * 60)
    print(f"  -> {out_dir}/lp_features.parquet         ({len(lp_features):,} rows)")
    print("-" * 60)

    spark.stop()
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LPulse PySpark feature pipeline")
    parser.add_argument("--pool",      required=True)
    parser.add_argument("--chain",     required=True)
    parser.add_argument("--merkl-url", required=True)
    parser.add_argument("--master",    default="local[*]",
                        help="Spark master URL (default: local[*])")
    args = parser.parse_args()
    run(chain=args.chain, pool=args.pool, merkl_url=args.merkl_url, master=args.master)


if __name__ == "__main__":
    main()
