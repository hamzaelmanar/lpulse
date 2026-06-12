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
        block_timestamp, chain_name, pool_address,
        owner, tick_lower, tick_upper, amount (liquidity, decimal TEXT or numeric).
    burn_df : DataFrame
        Raw decoded Burn events. Same columns except amount is liquidity removed.
    campaigns_df : DataFrame, optional
        Merkl campaign windows. Required columns:
        pool_address, chain_name, campaign_start (Unix s), campaign_end (Unix s).
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
    # TODO: implement
    raise NotImplementedError("verify_lp_exit() — Phase 2")


# ── exit_type ─────────────────────────────────────────────────────────────────

def exit_type(
    lp_summary: pd.DataFrame,
    swap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify each exited LP position as voluntary_exit or range_exit.

    Logic:
        For each position where status == 1 (fully exited):
          1. Find the last Swap event at or before the Burn block via merge_asof.
          2. Compute tick_at_burn from sqrtPriceX96:
                tick = floor(log2(sqrtPriceX96² / 2¹⁹²) / log2(1.0001))
             Equivalently:
                tick = floor(2 * log(sqrtPriceX96 / 2⁹⁶) / log(1.0001))
          3. If tick_lower <= tick_at_burn <= tick_upper → 'voluntary_exit'
             Else → 'range_exit'
        Censored positions (status == 0) → 'censored'

    The mercenary hypothesis predicts: campaign LPs cluster in voluntary_exit
    around campaign end dates. Non-campaign LPs should show more range_exit
    (passive, don't rebalance). DeepHit models these as competing risks.

    Parameters
    ----------
    lp_summary : DataFrame
        Output of verify_lp_exit(). Required columns:
        position_id, pool_address, chain_name, tick_lower, tick_upper,
        exit_timestamp, status.
    swap_df : DataFrame
        Raw decoded Swap events. Required columns:
        block_number, block_timestamp, chain_name, pool_address, sqrt_price_x96.

    Returns
    -------
    lp_summary with an added ``exit_type`` column:
        'voluntary_exit' | 'range_exit' | 'censored'
    """
    # TODO: implement
    raise NotImplementedError("exit_type() — Phase 3")


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
