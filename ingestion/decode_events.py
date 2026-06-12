"""
ingestion/decode_events.py
───────────────────────────
ABI-decodes raw Uniswap V3 log events from local Parquet (written by
hypersync_client.py) and writes one decoded Parquet per event type to:

    data/decoded/{chain}/{pool_address}/lp_mint_events.parquet
    data/decoded/{chain}/{pool_address}/lp_burn_events.parquet
    data/decoded/{chain}/{pool_address}/lp_swap_events.parquet
    data/decoded/{chain}/{pool_address}/lp_initialize_events.parquet

Column conventions:
  - All decoded numeric columns are str (TEXT) — same as FDP bronze.
    features/metrics.py handles type coercions via _hex_or_dec_to_int().
  - Timestamps remain 0x-prefixed hex (from HyperSync PREFIXED output).

Re-runs are incremental: watermark tracked in
    data/decoded/{chain}/{pool_address}/.watermark.json

Pure transformation — no network calls, no Postgres dependency.
On GCP, swap read/write targets to GCS; decoder functions are unchanged.
"""

import json
from pathlib import Path

import pandas as pd

# ── Uniswap V3 topic0 hashes (keccak256 of ABI event signature) ──────────────
TOPIC0_EVENT_MAP = {
    "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde": "Mint",
    "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c": "Burn",
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67": "Swap",
    "0x98636036cb66a21942a7841a974033d1c7cda7e036c4f67c88c2bdfc2240d609": "Initialize",
    "0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0": "Collect",
}

_EVENT_TABLES = {
    "Mint":       "lp_mint_events",
    "Burn":       "lp_burn_events",
    "Swap":       "lp_swap_events",
    "Initialize": "lp_initialize_events",
    "Collect":    "lp_collect_events",
}

_BASE_COLS = [
    "block_number",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "timestamp",
    "chain_name",
    "pool_address",
]


# ── Hex converters (arbitrary-precision, identical to FDP) ────────────────────

def _to_uint(hexstr: str) -> int:
    return int(hexstr, 16) if hexstr else 0


def _to_int256(hexstr: str) -> int:
    x = _to_uint(hexstr)
    return x - 2**256 if x >= 2**255 else x


def _to_int24(hexstr: str) -> int:
    return _to_int256(hexstr)


def _hex_chunk(data_no_prefix: str, word_index: int) -> str:
    start = word_index * 64
    return data_no_prefix[start: start + 64]


def _address_from_topic(topic: str):
    return "0x" + topic[-40:] if topic else None


# ── Per-event decoders (verbatim from FDP ingestion/utils/decode_events.py) ───

def _decode_mint(row: pd.Series) -> dict:
    d = row["data"][2:]
    return {
        "sender":     _address_from_topic(_hex_chunk(d, 0)),
        "owner":      _address_from_topic(row["topic1"]),
        "tick_lower": str(_to_int24(row["topic2"])),
        "tick_upper": str(_to_int24(row["topic3"])),
        "amount":     str(_to_uint(_hex_chunk(d, 1))),
        "amount0":    str(_to_uint(_hex_chunk(d, 2))),
        "amount1":    str(_to_uint(_hex_chunk(d, 3))),
    }


def _decode_burn(row: pd.Series) -> dict:
    d = row["data"][2:]
    return {
        "owner":      _address_from_topic(row["topic1"]),
        "tick_lower": str(_to_int24(row["topic2"])),
        "tick_upper": str(_to_int24(row["topic3"])),
        "amount":     str(_to_uint(_hex_chunk(d, 0))),
        "amount0":    str(_to_uint(_hex_chunk(d, 1))),
        "amount1":    str(_to_uint(_hex_chunk(d, 2))),
    }


def _decode_swap(row: pd.Series) -> dict:
    d = row["data"][2:]
    return {
        "sender":          _address_from_topic(row["topic1"]),
        "recipient":       _address_from_topic(row["topic2"]),
        "amount0":         str(_to_int256(_hex_chunk(d, 0))),
        "amount1":         str(_to_int256(_hex_chunk(d, 1))),
        "sqrt_price_x96":  str(_to_uint(_hex_chunk(d, 2))),
        "liquidity":       str(_to_uint(_hex_chunk(d, 3))),
        "tick":            str(_to_int256(_hex_chunk(d, 4))),
    }


def _decode_initialize(row: pd.Series) -> dict:
    d = row["data"][2:]
    return {
        "sqrt_price_x96": str(_to_uint(_hex_chunk(d, 0))),
        "tick":           str(_to_int256(_hex_chunk(d, 1))),
    }


def _decode_collect(row: pd.Series) -> dict:
    """
    Collect(address indexed owner, address recipient,
            int24 indexed tickLower, int24 indexed tickUpper,
            uint128 amount0, uint128 amount1)
    topics: [topic0=sig, topic1=owner, topic2=tickLower, topic3=tickUpper]
    data:   [recipient(32), amount0(32), amount1(32)]
    """
    d = row["data"][2:]
    return {
        "owner":      _address_from_topic(row["topic1"]),
        "recipient":  _address_from_topic(_hex_chunk(d, 0)),
        "tick_lower": str(_to_int24(row["topic2"])),
        "tick_upper": str(_to_int24(row["topic3"])),
        "amount0":    str(_to_uint(_hex_chunk(d, 1))),
        "amount1":    str(_to_uint(_hex_chunk(d, 2))),
    }


_EVENT_DECODERS = {
    "Mint":       _decode_mint,
    "Burn":       _decode_burn,
    "Swap":       _decode_swap,
    "Initialize": _decode_initialize,
    "Collect":    _decode_collect,
}


# ── Main decode function ──────────────────────────────────────────────────────

def decode(chain: str, pool_address: str, data_dir: Path) -> dict:
    """
    Read raw logs + blocks from data/raw/{chain}/{pool}/, decode all
    recognised event types, and write one Parquet per event type to
    data/decoded/{chain}/{pool}/.

    Parameters
    ----------
    chain        : Chain slug (e.g. "celo", "hyperevm")
    pool_address : Pool contract address
    data_dir     : Root data directory

    Returns
    -------
    dict mapping event_name -> rows_written (0 if no new data).
    """
    pool_lower = pool_address.lower()
    raw_dir     = data_dir / "raw"     / chain / pool_lower
    decoded_dir = data_dir / "decoded" / chain / pool_lower
    decoded_dir.mkdir(parents=True, exist_ok=True)

    logs_path   = raw_dir / "logs.parquet"
    blocks_path = raw_dir / "blocks.parquet"

    if not logs_path.exists():
        print(f"  No raw logs found at {logs_path}. Run fetch first.")
        return {}

    watermark_path = decoded_dir / ".watermark.json"
    last_decoded = 0
    if watermark_path.exists():
        with open(watermark_path) as f:
            last_decoded = json.load(f).get("block", 0)

    logs_df   = pd.read_parquet(logs_path)
    blocks_df = pd.read_parquet(blocks_path)

    # Filter to new blocks only
    logs_df = logs_df[logs_df["block_number"].astype("int64", errors="ignore") > last_decoded].copy()
    # Cast block_number safely: some rows may still be hex strings from raw
    logs_df["_block_int"] = logs_df["block_number"].apply(
        lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)
    )
    logs_df = logs_df[logs_df["_block_int"] > last_decoded].copy()

    if logs_df.empty:
        print(f"  No new log rows to decode (watermark: {last_decoded}).")
        return {}

    print(f"  Decoding {len(logs_df)} log rows (blocks > {last_decoded}) ...")

    # Join timestamp from blocks
    blocks_df["_block_int"] = blocks_df["number"].apply(
        lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else int(x)
    )
    block_ts = blocks_df[["_block_int", "timestamp", "chain_name"]].drop_duplicates("_block_int")

    logs_df = logs_df.merge(block_ts[["_block_int", "timestamp"]], on="_block_int", how="left")

    # Map topic0 -> event name
    logs_df["event_name"] = logs_df["topic0"].map(TOPIC0_EVENT_MAP)

    counts = {}
    for event_name, decoder in _EVENT_DECODERS.items():
        subset = logs_df[logs_df["event_name"] == event_name].copy()
        if subset.empty:
            print(f"  No {event_name} events — skipping.")
            counts[event_name] = 0
            continue

        decoded_fields = subset.apply(decoder, axis=1, result_type="expand")
        base = subset[
            [c for c in _BASE_COLS if c in subset.columns]
        ].reset_index(drop=True)
        out = pd.concat([base, decoded_fields.reset_index(drop=True)], axis=1)
        out = out.astype(str).replace("None", None)

        table = _EVENT_TABLES[event_name]
        out_path = decoded_dir / f"{table}.parquet"

        # Append to existing decoded file or create
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            out = pd.concat([existing, out], ignore_index=True)

        out.to_parquet(out_path, index=False)
        print(f"  {event_name}: {len(subset)} rows -> {out_path.name}")
        counts[event_name] = len(subset)

    # Update watermark to max decoded block
    max_block = int(logs_df["_block_int"].max())
    with open(watermark_path, "w") as f:
        json.dump({"block": max_block}, f)

    print(f"  Decode watermark -> {max_block}.")
    return counts
