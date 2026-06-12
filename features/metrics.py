"""
features/metrics.py
────────────────────
Core feature engineering functions for LPulse.

Written in pandas. Will be ported to PySpark (Dataproc) once the pipeline
is validated locally. Each function is pool-independent and stateless —
the PySpark port will be a direct translation using Spark DataFrame APIs.

Function inventory (ordered by dependency):
    verify_lp_exit()    → lp_summary: one row per position_id
                          Backlog item 1 — unblocks Hugo's Day-1 DeepSurv slice.
    exit_type()         → adds exit_type label to lp_summary
                          Backlog item 4 — unblocks DeepHit competing risks.
    event_sequence()    → long table per position_id
                          Backlog item 5 — unblocks DRSA/LSTM.

Deferred (not in this module):
    collected_fees()    — backlog item 2, blocked on per-LP fee accounting
    tvl()               — backlog item 3, blocked on Chainlink oracle ingestion
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np
import pandas as pd


# ── verify_lp_exit ────────────────────────────────────────────────────────────

def verify_lp_exit(
    mint_df: pd.DataFrame,
    burn_df: pd.DataFrame,
    campaigns_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Reconstruct individual LP positions from raw Mint/Burn events and compute
    survival labels.

    The key fix over FDP's position_key: when an LP opens, fully closes, and
    reopens the same tick range, those are distinct positions — not one. We
    detect cycle boundaries (cumulative liquidity hits 0) and stamp each new
    cycle with the tx hash of its opening Mint → ``position_id``.

    Parameters
    ----------
    mint_df : DataFrame
        Raw decoded Mint events. Required columns:
        block_number, transaction_hash, transaction_index, log_index,
        timestamp, chain_name, pool_address,
        owner, tick_lower, tick_upper, amount (liquidity, decimal TEXT or numeric).
    burn_df : DataFrame
        Raw decoded Burn events. Same columns as mint_df.
    campaigns_df : DataFrame, optional
        Merkl campaign windows from fetch_campaign_windows(). Required columns:
        start_timestamp (Unix s), end_timestamp (Unix s).
        lp_cohort boundaries: global_start = min(start_timestamp),
        global_end = max(end_timestamp), mirroring fct_lp_positions.sql logic.
        If None, lp_cohort is set to 'unknown' for all rows.

    Returns
    -------
    DataFrame — one row per position_id with columns:
        position_id         MD5(chain+pool+owner+tick_lower+tick_upper+first_mint_tx)
        owner               LP wallet address (lowercase)
        pool_address
        chain_name
        tick_lower
        tick_upper
        tick_range_width    tick_upper - tick_lower  (at-entry feature for DeepSurv)
        first_mint_tx_hash  opening transaction of this position cycle
        first_mint_timestamp  Unix seconds
        exit_timestamp      last Burn timestamp if exited, else pool max timestamp
        duration_seconds    exit_timestamp - first_mint_timestamp
        status              1 = fully exited, 0 = censored (still active)
        lp_cohort           'pre_campaign' | 'during_campaign' | 'post_campaign'
                            | 'unknown' (if no campaigns_df provided)
        event_count         total Mint+Burn events for this position
    """
    _SORT_COLS = [
        "chain_name", "pool_address", "owner", "tick_lower", "tick_upper",
        "block_number", "transaction_index", "log_index",
    ]
    _GROUP_KEY = ["chain_name", "pool_address", "owner", "tick_lower", "tick_upper"]
    _CYCLE_KEY = _GROUP_KEY + ["cycle_id"]
    _EVENT_COLS = [
        "block_number", "transaction_hash", "transaction_index",
        "log_index", "block_timestamp", "chain_name", "pool_address",
        "owner", "tick_lower", "tick_upper",
    ]

    # ── 1. Normalise and combine Mint + Burn ──────────────────────────────────

    def _prep(df: pd.DataFrame, event_type: str, sign: int) -> pd.DataFrame:
        # Raw tables name the timestamp column "timestamp"; rename to block_timestamp.
        out = df[
            ["block_number", "transaction_hash", "transaction_index", "log_index",
             "timestamp", "chain_name", "pool_address", "owner",
             "tick_lower", "tick_upper", "amount"]
        ].copy()
        out = out.rename(columns={"timestamp": "block_timestamp"})
        out["event_type"] = event_type
        # Python int preserves uint128 precision (avoids int64 overflow)
        out["liquidity_delta"] = out["amount"].apply(lambda x: sign * int(x))
        return out[_EVENT_COLS + ["event_type", "liquidity_delta"]]

    mints = _prep(mint_df, "Mint", +1)
    burns = _prep(burn_df, "Burn", -1)
    # Drop zero-amount burns (dust / fee collection artefacts)
    burns = burns[burns["liquidity_delta"] != 0].copy()

    events = pd.concat([mints, burns], ignore_index=True)

    # Cast sort columns to int for correct numeric ordering
    for col in ["block_number", "transaction_index", "log_index", "block_timestamp",
                "tick_lower", "tick_upper"]:
        events[col] = events[col].apply(_hex_or_dec_to_int)
    events["owner"] = events["owner"].str.lower()
    events["pool_address"] = events["pool_address"].str.lower()

    events = events.sort_values(_SORT_COLS).reset_index(drop=True)

    # ── 2. Cumulative liquidity per position_key ──────────────────────────────
    # Python-int cumsum via transform preserves arbitrary precision.
    events["cumulative_liq"] = (
        events.groupby(_GROUP_KEY)["liquidity_delta"]
        .transform(lambda s: s.cumsum())
    )

    # State BEFORE adding this event's liquidity
    events["cumulative_liq_before"] = (
        events.groupby(_GROUP_KEY)["cumulative_liq"]
        .transform(lambda s: s.shift(1).fillna(0))
    )

    # ── 3. Cycle detection ────────────────────────────────────────────────────
    # A new cycle starts when: a Mint arrives and cumulative_liq_before == 0.
    # (First Mint on a tick range always has cumulative_liq_before == 0 because
    #  fillna(0) fills the first row; re-opened ranges also satisfy this after
    #  cumulative_liq hit 0 on a prior Burn.)
    events["is_cycle_start"] = (
        (events["event_type"] == "Mint") & (events["cumulative_liq_before"] == 0)
    )

    # Orphan Burns (Burn with no preceding Mint) are data anomalies — warn and drop.
    orphans = events[(events["event_type"] == "Burn") & (events["cumulative_liq_before"] == 0)]
    if not orphans.empty:
        print(
            f"  Warning: {len(orphans)} orphan Burn event(s) with cumulative_liq_before == 0 "
            "(no preceding Mint on same tick range). These rows will be dropped."
        )

    events["cycle_id"] = (
        events.groupby(_GROUP_KEY)["is_cycle_start"].transform("cumsum")
    )

    # Drop orphans (cycle_id == 0 means no opening Mint was ever seen)
    events = events[events["cycle_id"] > 0].copy()

    # ── 4. Per-cycle aggregation ──────────────────────────────────────────────

    # Opening Mint tx hash (the event that started the cycle)
    cycle_opens = (
        events[events["is_cycle_start"]][_CYCLE_KEY + ["transaction_hash"]]
        .rename(columns={"transaction_hash": "first_mint_tx_hash"})
    )

    # Final cumulative liquidity (last row in each cycle, already sorted)
    final_liq = (
        events.groupby(_CYCLE_KEY)["cumulative_liq"]
        .last()
        .reset_index()
        .rename(columns={"cumulative_liq": "final_cumulative_liq"})
    )

    # First Mint timestamp
    first_mint_ts = (
        events[events["event_type"] == "Mint"]
        .groupby(_CYCLE_KEY)["block_timestamp"]
        .min()
        .reset_index()
        .rename(columns={"block_timestamp": "first_mint_timestamp"})
    )

    # Last event timestamp per cycle (used as exit_timestamp for fully exited positions)
    last_event_ts = (
        events.groupby(_CYCLE_KEY)["block_timestamp"]
        .max()
        .reset_index()
        .rename(columns={"block_timestamp": "last_event_timestamp"})
    )

    event_counts = (
        events.groupby(_CYCLE_KEY).size().reset_index(name="event_count")
    )

    cycle_df = (
        cycle_opens
        .merge(final_liq, on=_CYCLE_KEY)
        .merge(first_mint_ts, on=_CYCLE_KEY)
        .merge(last_event_ts, on=_CYCLE_KEY)
        .merge(event_counts, on=_CYCLE_KEY)
    )

    # ── 5. Pool-level max timestamp (right-censoring time) ────────────────────
    pool_max_ts = (
        events.groupby(["chain_name", "pool_address"])["block_timestamp"]
        .max()
        .reset_index()
        .rename(columns={"block_timestamp": "pool_max_timestamp"})
    )
    cycle_df = cycle_df.merge(pool_max_ts, on=["chain_name", "pool_address"])

    # ── 6. Survival labels ────────────────────────────────────────────────────
    cycle_df["status"] = (cycle_df["final_cumulative_liq"] == 0).astype(int)
    cycle_df["exit_timestamp"] = np.where(
        cycle_df["status"] == 1,
        cycle_df["last_event_timestamp"],
        cycle_df["pool_max_timestamp"],
    )
    cycle_df["duration_seconds"] = (
        cycle_df["exit_timestamp"] - cycle_df["first_mint_timestamp"]
    )

    # ── 7. At-entry features ──────────────────────────────────────────────────
    cycle_df["tick_range_width"] = cycle_df["tick_upper"] - cycle_df["tick_lower"]

    # ── 8. Stable position_id (includes first_mint_tx_hash) ──────────────────
    cycle_df["position_id"] = cycle_df.apply(
        lambda r: _make_position_id(
            r["chain_name"], r["pool_address"], r["owner"],
            int(r["tick_lower"]), int(r["tick_upper"]), r["first_mint_tx_hash"],
        ),
        axis=1,
    )

    # ── 9. lp_cohort — mirrors fct_lp_positions.sql cross join campaign_window ─
    if campaigns_df is not None and not campaigns_df.empty:
        global_start = int(campaigns_df["start_timestamp"].min())
        global_end   = int(campaigns_df["end_timestamp"].max())

        def _cohort(ts: int) -> str:
            if ts < global_start:
                return "pre_campaign"
            elif ts <= global_end:
                return "during_campaign"
            else:
                return "post_campaign"

        cycle_df["lp_cohort"] = cycle_df["first_mint_timestamp"].apply(_cohort)
    else:
        cycle_df["lp_cohort"] = "unknown"

    # ── 10. Final output ──────────────────────────────────────────────────────
    out_cols = [
        "position_id", "owner", "pool_address", "chain_name",
        "tick_lower", "tick_upper", "tick_range_width",
        "first_mint_tx_hash", "first_mint_timestamp",
        "exit_timestamp", "duration_seconds", "status", "lp_cohort",
        "event_count",
    ]
    return cycle_df[out_cols].reset_index(drop=True)


# ── exit_type ─────────────────────────────────────────────────────────────────

def exit_type(
    lp_summary: pd.DataFrame,
    swap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each exited LP position as voluntary_exit or range_exit.

    Logic:
        For each position where status == 1 (fully exited):
          1. Find the last Swap event at or before exit_timestamp via merge_asof
             on block_timestamp. The Swap row carries a decoded ``tick`` column
             (current pool tick after the swap) — no sqrtPriceX96 computation needed.
          2. If tick_lower <= tick_at_exit <= tick_upper → 'voluntary_exit'
             (price was in-range when LP burned; they chose to leave)
             Else → 'range_exit'
             (price was outside the LP's range; position had become dead capital)
          3. No prior swap found → 'range_exit' (conservative fallback, logged)
        Censored positions (status == 0) → 'censored'

    The mercenary hypothesis predicts: campaign LPs cluster in voluntary_exit
    around campaign end dates. Non-campaign LPs should show more range_exit
    (passive, don't rebalance). DeepHit models these as competing risks.

    Parameters
    ----------
    lp_summary : DataFrame
        Output of verify_lp_exit(). Required columns:
        position_id, pool_address, chain_name, tick_lower, tick_upper,
        exit_timestamp (int, Unix seconds), status.
    swap_df : DataFrame
        Raw decoded Swap events. Required columns:
        timestamp (hex or decimal string, Unix seconds), chain_name,
        pool_address, tick (decimal string, current pool tick after swap).

    Returns
    -------
    lp_summary with an added ``exit_type`` column:
        'voluntary_exit' | 'range_exit' | 'censored'
    """
    result = lp_summary.copy()
    result["exit_type"] = "censored"

    exited_mask = result["status"] == 1
    if not exited_mask.any():
        return result

    # ── Prepare swaps ──────────────────────────────────────────────────────
    swaps = swap_df[["timestamp", "chain_name", "pool_address", "tick"]].copy()
    swaps = swaps.rename(columns={"timestamp": "swap_ts"})
    swaps["swap_ts"] = swaps["swap_ts"].apply(_hex_or_dec_to_int)
    swaps["tick_int"] = swaps["tick"].apply(lambda x: int(x))
    swaps = swaps.sort_values("swap_ts").reset_index(drop=True)

    # ── Prepare exited positions ───────────────────────────────────────────
    exited = result.loc[exited_mask, [
        "position_id", "pool_address", "chain_name",
        "tick_lower", "tick_upper", "exit_timestamp",
    ]].copy()
    exited = exited.sort_values("exit_timestamp").reset_index(drop=True)

    # ── merge_asof: last swap at or before exit_timestamp, per pool ────────
    # by=['pool_address', 'chain_name'] requires both DFs to have matching vals
    merged = pd.merge_asof(
        exited.rename(columns={"exit_timestamp": "swap_ts"}),
        swaps[["pool_address", "chain_name", "swap_ts", "tick_int"]],
        on="swap_ts",
        by=["pool_address", "chain_name"],
        direction="backward",
    )

    no_swap = merged["tick_int"].isna().sum()
    if no_swap:
        print(
            f"  Warning: {no_swap} exited position(s) had no prior Swap event — "
            "classified as range_exit (conservative)."
        )

    # ── Classify ───────────────────────────────────────────────────────────
    def _classify(row) -> str:
        if pd.isna(row["tick_int"]):
            return "range_exit"
        tick = int(row["tick_int"])
        if int(row["tick_lower"]) <= tick <= int(row["tick_upper"]):
            return "voluntary_exit"
        return "range_exit"

    merged["exit_type_val"] = merged.apply(_classify, axis=1)
    exit_map = merged.set_index("position_id")["exit_type_val"].to_dict()

    result.loc[exited_mask, "exit_type"] = (
        result.loc[exited_mask, "position_id"].map(exit_map)
    )

    return result


# ── event_sequence ────────────────────────────────────────────────────────────

def event_sequence(
    mint_df: pd.DataFrame,
    burn_df: pd.DataFrame,
    collect_df: pd.DataFrame,
    swap_df: pd.DataFrame,
    lp_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the event sequence table: one row per event per position_id.

    Used by the DRSA/LSTM model to learn event trajectories leading to exit.
    The sequence [Mint → Swap×N → Collect → Burn] encodes the behavioural
    signature of an LP's lifecycle. Fees stop accumulating when price leaves
    range — this pattern is a leading indicator of exit.

    Parameters
    ----------
    mint_df, burn_df, collect_df : DataFrame
        Decoded event tables. Required columns: block_number, block_timestamp,
        transaction_hash, log_index, chain_name, pool_address,
        owner, tick_lower, tick_upper, amount (or amount0/amount1 for collect).

        Note: collect_df uses raw.lp_collect_events (decoded via HyperSync;
        topic0 = 0x70935338…). Ensure this table is populated before calling.
    swap_df : DataFrame
        Swap events. Required columns: block_number, block_timestamp,
        chain_name, pool_address, sqrt_price_x96.
    lp_summary : DataFrame
        Output of verify_lp_exit(). Used to resolve position_id per event.
        Required columns: position_id, owner, pool_address, chain_name,
        tick_lower, tick_upper, first_mint_timestamp.

    Returns
    -------
    DataFrame — long table with columns:
        position_id
        seq_num         0-indexed event order within the position lifecycle
        event_type      'Mint' | 'Burn' | 'Collect'
        block_number
        block_timestamp
        transaction_hash
        log_index
        liquidity_delta signed liquidity change (0 for Collect)
        amount0_raw     token0 amount (numeric)
        amount1_raw     token1 amount (numeric)
        price_at_event  spot price derived from last Swap before this event
                        via merge_asof on block_number (NaN if no prior Swap)
    """
    # TODO: implement
    raise NotImplementedError("event_sequence() — Phase 4")


# ── Internal helpers (shared by multiple functions) ───────────────────────────

def _hex_or_dec_to_int(x) -> int:
    """Convert a value that may be a 0x-prefixed hex string or a decimal string to int."""
    s = str(x).strip()
    return int(s, 16) if s.startswith(("0x", "0X")) else int(s)


def _make_position_id(
    chain_name: str,
    pool_address: str,
    owner: str,
    tick_lower: int,
    tick_upper: int,
    first_mint_tx_hash: str,
) -> str:
    """
    Stable 32-char surrogate key for one LP position cycle.

    Differs from FDP's position_key by including first_mint_tx_hash —
    this disambiguates re-opened positions on the same tick range.
    """
    raw = f"{chain_name}|{pool_address.lower()}|{owner.lower()}|{tick_lower}|{tick_upper}|{first_mint_tx_hash.lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _sqrt_price_to_tick(sqrt_price_x96: int) -> int:
    """
    Convert Uniswap V3 sqrtPriceX96 to the corresponding tick.

        tick = floor(log(sqrtPriceX96² / 2¹⁹²) / log(1.0001))
             = floor(2 * log(sqrtPriceX96 / 2⁹⁶) / log(1.0001))

    Uses natural log to avoid precision loss from log2 on large ints.
    Safe for sqrtPriceX96 in the valid Uniswap V3 range.
    """
    if sqrt_price_x96 <= 0:
        raise ValueError(f"sqrtPriceX96 must be positive, got {sqrt_price_x96}")
    log_ratio = 2 * math.log(sqrt_price_x96 / (2**96))
    return math.floor(log_ratio / math.log(1.0001))
