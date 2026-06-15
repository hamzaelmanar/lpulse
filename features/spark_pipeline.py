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
from typing import Optional

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType

from features._env import setup
from features.merkl import fetch_campaign_windows

setup()

BASE_DATA_DIR = Path(__file__).parent.parent / "data"

# ── Spark session ──────────────────────────────────────────────────────────────

def _get_spark(master: str = "local[*]") -> SparkSession:
    # On Windows, Spark workers call bare `python` which hits the MS Store stub.
    # Point PYSPARK_PYTHON to the same interpreter running this process.
    import sys
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder
        .master(master)
        .appName("lpulse-feature-pipeline")
        .config("spark.sql.shuffle.partitions", "32")  # 4× more partitions → less per-partition spill
        .config("spark.driver.memory", "8g")           # Fix 2: prevent JVM heap exhaustion
        .config("spark.memory.fraction", "0.8")        # more execution memory, less reserved
        .config("spark.memory.storageFraction", "0.3") # balance storage vs execution
        .getOrCreate()
    )


# ── Native column helpers (no Python UDFs — avoids worker subprocess on Windows) ──

def _hex_to_int_col(col_expr):
    """Convert a hex-or-decimal string column to long. No Python UDF."""
    return (
        F.when(F.upper(F.substring(col_expr, 1, 2)) == F.lit("0X"),
               F.conv(F.regexp_replace(col_expr, "^0[xX]", ""), 16, 10).cast(LongType()))
        .otherwise(col_expr.cast(LongType()))
    )


def _int_col(col_expr):
    """Safe int cast for tick columns stored as decimal strings."""
    return col_expr.cast(IntegerType())


def _position_id_col(chain, pool, owner, tick_lower, tick_upper, first_mint_tx):
    """MD5 surrogate key — identical output to hashlib.md5(key).hexdigest()."""
    return F.md5(F.concat_ws("|", chain, pool, owner,
                              tick_lower.cast("string"),
                              tick_upper.cast("string"),
                              first_mint_tx))



# ── I/O helpers ────────────────────────────────────────────────────────────────

def _read_events(spark: SparkSession, chain: str, pool: str, table: str):
    path = BASE_DATA_DIR / "decoded" / chain / pool / f"{table}.parquet"
    if not path.exists():
        print(f"  Warning: {path} not found -- returning empty DataFrame.")
        # Schema covers all _BASE_COLS so downstream withColumn calls on any base column don't crash.
        return spark.createDataFrame([], schema=(
            "block_number STRING, transaction_hash STRING, transaction_index STRING, "
            "log_index STRING, timestamp STRING, chain_name STRING, pool_address STRING"
        ))
    return spark.read.parquet(str(path))


# ── verify_lp_exit (Spark) ─────────────────────────────────────────────────────

def _verify_lp_exit_spark(spark, mint_df, burn_df, campaigns_df, chain: str, pool: str,
                          skew_threshold: int = 1000):
    """
    Reconstruct LP position cycles. Logic mirrors metrics.verify_lp_exit().

    Hot/normal split: id_keys combinations with more events than skew_threshold
    are routed through a pandas fallback on the driver to avoid window spill.
    Normal keys use the full Spark window path.

    skew_threshold: max events per id_keys combination allowed on the Spark
    window path. Determined empirically via the hot-key diagnostic.
    TODO: make adaptive — compute as a percentile of the per-key count
    distribution rather than a fixed constant (see spark_postmortem.md).

    Returns a Spark DataFrame with one row per position cycle.
    """
    import pandas as pd
    import hashlib

    id_keys   = ["chain_name", "pool_address", "owner", "tick_lower", "tick_upper"]
    cycle_keys = id_keys + ["cycle_id"]
    _blk_ord  = [
        F.col("block_number").cast(LongType()),
        F.col("transaction_index").cast(IntegerType()),
        F.col("log_index").cast(IntegerType()),
    ]

    if not campaigns_df.empty:
        global_start = int(campaigns_df["start_timestamp"].min())
        global_end   = int(campaigns_df["end_timestamp"].max())
    else:
        global_start = global_end = 0

    # ── shared cast helper ────────────────────────────────────────────────────
    def _cast(df):
        return (
            df.withColumn("ts",          _hex_to_int_col(F.col("timestamp")))
              .withColumn("tick_lower",   _int_col(F.col("tick_lower")))
              .withColumn("tick_upper",   _int_col(F.col("tick_upper")))
              .withColumn("amount",       F.col("amount").cast("decimal(38,0)"))
              .withColumn("owner",        F.lower(F.col("owner")))
              .withColumn("pool_address", F.lower(F.col("pool_address")))
        )

    mint_tagged = _cast(mint_df).withColumn("liq_delta", F.col("amount"))
    burn_tagged = _cast(burn_df).withColumn(
        "liq_delta", F.col("amount").cast("decimal(38,0)") * -1
    )

    events = mint_tagged.unionByName(burn_tagged, allowMissingColumns=True) \
                        .select("chain_name", "pool_address", "owner",
                                "tick_lower", "tick_upper", "ts",
                                "block_number", "transaction_index", "log_index",
                                "transaction_hash", "liq_delta")

    # ── hot/normal split ──────────────────────────────────────────────────────
    key_counts    = events.groupBy(*id_keys).count()
    hot_keys      = key_counts.filter(F.col("count") >= skew_threshold).drop("count")
    normal_events = events.join(hot_keys, on=id_keys, how="left_anti")
    hot_events    = events.join(hot_keys, on=id_keys, how="inner")

    n_hot = hot_keys.count()
    print(f"  Skew split: {n_hot} hot key(s) >= {skew_threshold} events → pandas path.")

    # ── normal path: Spark window ─────────────────────────────────────────────
    def _spark_window_agg(ev):
        ord_win = Window.partitionBy(*id_keys).orderBy(*_blk_ord)
        cum_win = Window.partitionBy(*id_keys).orderBy(*_blk_ord) \
                        .rowsBetween(Window.unboundedPreceding, 0)

        ev = ev.withColumn("cum_liq",  F.sum("liq_delta").over(cum_win))
        ev = ev.withColumn("prev_cum", F.lag("cum_liq").over(ord_win))
        ev = ev.withColumn("is_open",
            F.col("prev_cum").isNull() | (F.col("prev_cum") <= 0)
        )
        ev = ev.withColumn("cycle_id",
            F.sum(F.col("is_open").cast("int")).over(ord_win)
        )

        first_tx = ev.filter(F.col("is_open")).groupBy(*cycle_keys) \
                     .agg(F.first("transaction_hash").alias("first_mint_tx_hash"))

        agg = ev.groupBy(*cycle_keys).agg(
            F.min("ts").alias("first_mint_timestamp"),
            F.max(F.when(F.col("liq_delta") < 0, F.col("ts"))).alias("exit_timestamp"),
            F.count("*").alias("event_count"),
        )
        agg = agg.join(first_tx, on=cycle_keys, how="left")
        return agg

    normal_agg = _spark_window_agg(normal_events)

    # ── hot path: pandas inline stub ──────────────────────────────────────────
    def _pandas_agg(ev_spark) -> Optional["pd.DataFrame"]:
        """Inline pandas implementation for hot (skewed) id_keys partitions."""
        pdf = ev_spark.toPandas()
        if pdf.empty:
            return None

        pdf["ts"]                = pd.to_numeric(pdf["ts"], errors="coerce")
        pdf["block_number"]      = pd.to_numeric(pdf["block_number"], errors="coerce")
        pdf["transaction_index"] = pd.to_numeric(pdf["transaction_index"], errors="coerce")
        pdf["log_index"]         = pd.to_numeric(pdf["log_index"], errors="coerce")
        pdf["liq_delta"]         = pdf["liq_delta"].apply(lambda x: int(x))

        pdf = pdf.sort_values(
            ["chain_name", "pool_address", "owner", "tick_lower", "tick_upper",
             "block_number", "transaction_index", "log_index"]
        ).reset_index(drop=True)

        pdf["cum_liq"]  = pdf.groupby(id_keys)["liq_delta"].transform("cumsum")
        pdf["prev_cum"] = pdf.groupby(id_keys)["cum_liq"].transform(lambda s: s.shift(1))
        pdf["is_open"]  = pdf["prev_cum"].isna() | (pdf["prev_cum"] <= 0)
        pdf["cycle_id"] = pdf.groupby(id_keys)["is_open"].transform("cumsum")

        rows = []
        for keys, grp in pdf.groupby(cycle_keys):
            key_dict = dict(zip(cycle_keys, keys))
            first_open = grp[grp["is_open"]]
            first_mint_tx = first_open.iloc[0]["transaction_hash"] if not first_open.empty else None
            burns = grp[grp["liq_delta"] < 0]
            exit_ts = int(burns["ts"].max()) if not burns.empty else None
            row = {
                **key_dict,
                "first_mint_timestamp": int(grp["ts"].min()),
                "exit_timestamp":       exit_ts,
                "event_count":          len(grp),
                "first_mint_tx_hash":   first_mint_tx,
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    hot_pdf = _pandas_agg(hot_events)

    # ── combine results ───────────────────────────────────────────────────────
    def _enrich(agg):
        """Add derived columns shared by both paths."""
        agg = agg.withColumn("status",
            F.when(F.col("exit_timestamp").isNotNull(), F.lit(1)).otherwise(F.lit(0))
        )
        agg = agg.withColumn("duration_seconds",
            F.when(
                F.col("status") == 1,
                F.col("exit_timestamp").cast(LongType()) - F.col("first_mint_timestamp").cast(LongType())
            ).otherwise(F.lit(None).cast(LongType()))
        )
        agg = agg.withColumn("tick_range_width",
            F.col("tick_upper") - F.col("tick_lower")
        )
        agg = agg.withColumn("lp_cohort",
            F.when(
                (F.col("first_mint_timestamp") >= global_start) &
                (F.col("first_mint_timestamp") <= global_end),
                F.lit("during_campaign")
            ).when(F.col("first_mint_timestamp") < global_start, F.lit("pre_campaign"))
             .when(F.col("first_mint_timestamp") > global_end,   F.lit("post_campaign"))
             .otherwise(F.lit("unknown"))
        )
        agg = agg.withColumn("position_id",
            _position_id_col(
                F.col("chain_name"), F.col("pool_address"), F.col("owner"),
                F.col("tick_lower"), F.col("tick_upper"),
                F.col("first_mint_tx_hash"),
            )
        )
        return agg

    normal_agg = _enrich(normal_agg)

    if hot_pdf is not None and not hot_pdf.empty:
        # Write hot pandas result to a temp parquet file and read back as a
        # native Spark parquet DataFrame.  spark.createDataFrame(pandas_df)
        # uses a Python-backed RDD: when Spark re-evaluates the lazy plan it
        # spawns a Python worker subprocess which crashes on Windows.
        # Reading from a real parquet file is pure JVM — no Python worker.
        import tempfile
        _tmp_dir  = Path(tempfile.mkdtemp())
        _tmp_path = _tmp_dir / "hot_agg.parquet"
        hot_pdf.to_parquet(str(_tmp_path), index=False)
        hot_spark = spark.read.parquet(str(_tmp_path))
        hot_spark = _enrich(hot_spark)
        result = normal_agg.unionByName(hot_spark, allowMissingColumns=True)
    else:
        result = normal_agg

    return result


# ── exit_type (Spark) ──────────────────────────────────────────────────────────

def _exit_type_spark(lp_summary, swap_df):
    """Classify exits as voluntary_exit, range_exit, or censored.

    Local-mode implementation: collects both DataFrames to pandas and delegates
    to metrics.exit_type() which uses pd.merge_asof (O(N+M) ASOF join).

    This avoids the cartesian cross-join that Spark would generate for the
    non-equi predicate  swap_ts <= exit_timestamp  on a single pool_address —
    that pattern materialises O(N_positions × N_swaps) intermediate rows,
    exhausting Tungsten execution memory in a single-JVM local deployment.

    Dataproc note: at multi-pool scale, replace this with a native Spark
    range-join approach (bucketed sort-merge or Spark 3.3+ range-join
    optimisation).  The pandas path is intentionally local-only.
    """
    import tempfile
    from features.metrics import exit_type as pd_exit_type

    print("  Collecting lp_summary and swap_df to pandas for ASOF join...")
    lp_pdf   = lp_summary.toPandas()
    swap_pdf = swap_df.toPandas()
    print(f"  lp_summary: {len(lp_pdf):,} rows  |  swap_df: {len(swap_pdf):,} rows")

    # Spark nullable LongType → float64 in pandas when the column contains any NULLs
    # (exit_timestamp is NULL for censored positions, first_mint_timestamp never null
    # but may still be inferred as float64 in the same batch).
    # pd.merge_asof requires exact dtype equality on the `on` key.
    # Censored rows are filtered inside metrics.exit_type before the merge, so
    # setting their NULL exit_timestamp to 0 is harmless.
    for _col in ("exit_timestamp", "first_mint_timestamp"):
        if _col in lp_pdf.columns:
            lp_pdf[_col] = lp_pdf[_col].fillna(0).astype("int64")

    result_pdf = pd_exit_type(lp_pdf, swap_pdf)

    # Back to Spark via parquet — spark.createDataFrame(pandas_df) spawns a
    # Python worker subprocess which crashes on Windows; reading a real parquet
    # file is pure JVM.
    _tmp = Path(tempfile.mkdtemp()) / "exit_type.parquet"
    result_pdf.to_parquet(str(_tmp), index=False)
    spark = lp_summary.sparkSession
    return spark.read.parquet(str(_tmp))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(chain: str, pool: str, merkl_url: str, master: str = "local[*]",
        skew_threshold: int = 1000) -> Path:
    import time
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
    _t0 = time.perf_counter()
    lp_summary = _verify_lp_exit_spark(spark, mint_df, burn_df, campaigns_df, chain, pool,
                                        skew_threshold=skew_threshold)
    print(f"  verify_lp_exit: {time.perf_counter() - _t0:.1f}s")

    print("Classifying exit types...")
    lp_summary = _exit_type_spark(lp_summary, swap_df)

    # Fix 5 (revised): single toPandas() — no cache(), no shuffle-based count().
    #
    # Root cause of all previous deadlocks: any Spark action (count, agg, toPandas)
    # that reads the lp_summary partitions goes through a shuffle stage. Those stages
    # allocate Tungsten memory pages. Partitions 2 and 6 consistently fail to acquire
    # pages because earlier tasks in the same stage hold the Tungsten allocator lock
    # (unified memory pool is exhausted). This is a local-mode constraint: a single
    # JVM, one shared Tungsten pool, no inter-node memory isolation.
    #
    # toPandas() bypasses Tungsten entirely: rows are serialized as Java objects over
    # the py4j socket to Python. Memory pressure moves to the Python heap (normal
    # CPython allocator), which is independent of Tungsten. No lock contention possible.
    #
    # Trade-off: the full DAG runs once (no cache), counts are done in pandas (free).
    # Two file writes from one in-memory DataFrame — no second DAG execution.

    at_entry_cols = [
        "position_id", "owner", "pool_address", "chain_name",
        "tick_lower", "tick_upper", "tick_range_width",
        "first_mint_timestamp", "first_mint_tx_hash",
        "duration_seconds", "status", "exit_type", "lp_cohort", "event_count",
    ]
    survival_cols = ["position_id", "duration_seconds", "status", "exit_type"]

    # Collect union of all needed columns in one Spark action
    all_out_cols = list(dict.fromkeys(at_entry_cols + survival_cols))
    print("Collecting results (toPandas)...")
    lp_pdf = lp_summary.select(
        *[c for c in all_out_cols if c in lp_summary.columns]
    ).toPandas()

    n_total    = len(lp_pdf)
    n_exited   = int((lp_pdf["status"] == 1).sum())
    n_censored = int((lp_pdf["status"] == 0).sum())
    print(f"  {n_total:,} positions ({n_exited:,} exited, {n_censored:,} censored).")

    # ── Write outputs ──────────────────────────────────────────────────────────
    feat_cols = [c for c in at_entry_cols if c in lp_pdf.columns]
    lp_pdf[feat_cols].to_parquet(out_dir / "lp_features.parquet", index=False)

    surv_cols = [c for c in survival_cols if c in lp_pdf.columns]
    lp_pdf[surv_cols].to_parquet(out_dir / "lp_survival_labels.parquet", index=False)

    print("-" * 60)
    print(f"  -> {out_dir}/lp_features.parquet         ({n_total:,} rows)")
    print("-" * 60)

    try:
        spark.stop()
    except Exception:
        # py4j socket closes before Python sees the JVM shutdown on Windows;
        # the data is already written so this is safe to swallow.
        pass
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="LPulse PySpark feature pipeline")
    parser.add_argument("--pool",           required=True)
    parser.add_argument("--chain",          required=True)
    parser.add_argument("--merkl-url",      required=True)
    parser.add_argument("--master",         default="local[*]",
                        help="Spark master URL (default: local[*])")
    parser.add_argument("--skew-threshold", type=int, default=1000,
                        help="Max events per id_key allowed on Spark window path (default: 1000). "
                             "Set to 999999 to disable hot/normal split for benchmarking.")
    args = parser.parse_args()
    run(chain=args.chain, pool=args.pool, merkl_url=args.merkl_url, master=args.master,
        skew_threshold=args.skew_threshold)


if __name__ == "__main__":
    main()
