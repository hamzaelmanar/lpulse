"""
tests/test_e2e.py
──────────────────
End-to-end test for the LPulse feature pipeline.

Scope: Phase 2 — verify_lp_exit() with real data.
       Phases 3–5 (exit_type, event_sequence, full pipeline.run()) are
       marked xfail until implemented.

Requires:
    - POSTGRES_* environment variables pointing at the financial-data-platform
      Postgres instance (raw.lp_mint_events and raw.lp_burn_events populated)
    - Network access to api.merkl.xyz (live campaign window fetch)

Run:
    pytest tests/test_e2e.py -v

Skip if Postgres unavailable:
    pytest tests/test_e2e.py -v -m "not e2e"
"""

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from features._env import setup
setup()

# ── Constants ─────────────────────────────────────────────────────────────────

CELO_POOL    = "0xF55791AfBB35aD42984f18D6Fe3e1fF73D81900c"
CELO_CHAIN   = "celo"
CELO_MERKL   = "https://app.merkl.xyz/opportunities/celo/CLAMM/0xF55791AfBB35aD42984f18D6Fe3e1fF73D81900c"

VALID_COHORTS    = {"pre_campaign", "during_campaign", "post_campaign", "unknown"}
VALID_EXIT_TYPES = {"voluntary_exit", "range_exit", "censored"}
VALID_STATUS     = {0, 1}


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _pg_available() -> bool:
    try:
        from sqlalchemy import create_engine, text
        url = (
            f"postgresql+psycopg2://"
            f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
            f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}"
            f"/{os.getenv('POSTGRES_DB')}"
        )
        with create_engine(url, future=True).connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="Postgres not reachable — set POSTGRES_* env vars"
)


@pytest.fixture(scope="module")
def engine():
    url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url, future=True)


def _load(engine, table: str) -> pd.DataFrame:
    q = text(
        f"SELECT * FROM raw.{table} "
        "WHERE chain_name = :chain AND pool_address = :pool"
    )
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={
            "chain": CELO_CHAIN,
            "pool": CELO_POOL.lower(),
        })


@pytest.fixture(scope="module")
def raw_events(engine):
    def _load_safe(engine, table):
        try:
            return _load(engine, table)
        except Exception as exc:
            if "UndefinedTable" in type(exc).__name__ or "does not exist" in str(exc) or "n'existe pas" in str(exc):
                print(f"  Warning: raw.{table} not found — returning empty DataFrame.")
                return pd.DataFrame()
            raise
    return {
        "mint":    _load(engine, "lp_mint_events"),
        "burn":    _load(engine, "lp_burn_events"),
        "swap":    _load(engine, "lp_swap_events"),
        "collect": _load_safe(engine, "lp_collect_events"),
    }


@pytest.fixture(scope="module")
def campaigns():
    from features.merkl import fetch_campaign_windows
    return fetch_campaign_windows(CELO_MERKL)


@pytest.fixture(scope="module")
def lp_summary(raw_events, campaigns):
    from features.metrics import verify_lp_exit
    return verify_lp_exit(raw_events["mint"], raw_events["burn"], campaigns)


# ── Merkl API tests (no Postgres needed) ─────────────────────────────────────

class TestMerklClient:
    def test_fetches_campaigns(self):
        from features.merkl import fetch_campaign_windows
        df = fetch_campaign_windows(CELO_MERKL)
        assert not df.empty, "Expected at least one campaign"
        assert "start_timestamp" in df.columns
        assert "end_timestamp" in df.columns

    def test_timestamps_are_integers(self):
        from features.merkl import fetch_campaign_windows
        df = fetch_campaign_windows(CELO_MERKL)
        assert df["start_timestamp"].dtype in ("int64", "int32", object)
        for _, row in df.iterrows():
            assert int(row["start_timestamp"]) < int(row["end_timestamp"]), \
                f"Campaign {row['campaign_id']}: start >= end"

    def test_invalid_url_raises(self):
        from features.merkl import fetch_campaign_windows
        with pytest.raises(Exception):
            fetch_campaign_windows("https://app.merkl.xyz/opportunities/celo/CLAMM/0xDEAD")


# ── verify_lp_exit tests ──────────────────────────────────────────────────────

@requires_pg
class TestVerifyLpExit:
    def test_returns_dataframe(self, lp_summary):
        assert isinstance(lp_summary, pd.DataFrame)

    def test_non_empty(self, lp_summary, raw_events):
        assert len(lp_summary) > 0, (
            f"Expected positions from {len(raw_events['mint'])} Mints "
            f"and {len(raw_events['burn'])} Burns"
        )

    def test_no_duplicate_position_ids(self, lp_summary):
        dupes = lp_summary[lp_summary["position_id"].duplicated()]
        assert dupes.empty, (
            f"{len(dupes)} duplicate position_id(s) found:\n{dupes}"
        )

    def test_position_id_is_32_chars(self, lp_summary):
        assert (lp_summary["position_id"].str.len() == 32).all(), \
            "position_id should be a 32-char MD5 hex string"

    def test_status_values(self, lp_summary):
        bad = set(lp_summary["status"].unique()) - VALID_STATUS
        assert not bad, f"Unexpected status value(s): {bad}"

    def test_duration_non_negative(self, lp_summary):
        neg = lp_summary[lp_summary["duration_seconds"] < 0]
        assert neg.empty, (
            f"{len(neg)} position(s) with negative duration_seconds:\n{neg}"
        )

    def test_lp_cohort_values(self, lp_summary):
        bad = set(lp_summary["lp_cohort"].unique()) - VALID_COHORTS
        assert not bad, f"Unexpected lp_cohort value(s): {bad}"

    def test_no_null_position_ids(self, lp_summary):
        assert lp_summary["position_id"].notna().all()

    def test_tick_range_width_positive(self, lp_summary):
        bad = lp_summary[lp_summary["tick_range_width"] <= 0]
        assert bad.empty, f"{len(bad)} position(s) with tick_range_width <= 0"

    def test_event_count_at_least_one(self, lp_summary):
        bad = lp_summary[lp_summary["event_count"] < 1]
        assert bad.empty, f"{len(bad)} position(s) with event_count < 1"


# ── exit_type tests ───────────────────────────────────────────────────────────

@requires_pg
class TestExitType:
    @pytest.fixture(scope="class")
    def lp_with_exit_type(self, lp_summary, raw_events):
        from features.metrics import exit_type
        return exit_type(lp_summary, raw_events["swap"])

    def test_column_present(self, lp_with_exit_type):
        assert "exit_type" in lp_with_exit_type.columns

    def test_valid_values(self, lp_with_exit_type):
        bad = set(lp_with_exit_type["exit_type"].unique()) - VALID_EXIT_TYPES
        assert not bad, f"Unexpected exit_type value(s): {bad}"

    def test_censored_positions_are_censored(self, lp_with_exit_type):
        censored = lp_with_exit_type[lp_with_exit_type["status"] == 0]
        wrong = censored[censored["exit_type"] != "censored"]
        assert wrong.empty, f"{len(wrong)} censored position(s) with non-censored exit_type"

    def test_exited_positions_not_censored(self, lp_with_exit_type):
        exited = lp_with_exit_type[lp_with_exit_type["status"] == 1]
        wrong = exited[exited["exit_type"] == "censored"]
        assert wrong.empty, f"{len(wrong)} exited position(s) labelled 'censored'"

    def test_no_nulls(self, lp_with_exit_type):
        nulls = lp_with_exit_type["exit_type"].isna().sum()
        assert nulls == 0, f"{nulls} null exit_type value(s)"


# ── event_sequence tests ──────────────────────────────────────────────────────

@requires_pg
class TestEventSequence:
    @pytest.fixture(scope="class")
    def sequences(self, raw_events, lp_summary):
        from features.metrics import event_sequence
        return event_sequence(
            raw_events["mint"], raw_events["burn"],
            raw_events["collect"], raw_events["swap"],
            lp_summary,
        )

    def test_non_empty(self, sequences):
        assert len(sequences) > 0

    def test_required_columns(self, sequences):
        required = {"position_id", "seq_num", "event_type", "block_number",
                    "block_timestamp", "transaction_hash", "log_index",
                    "liquidity_delta", "amount0_raw", "amount1_raw", "price_at_event"}
        missing = required - set(sequences.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_event_type_values(self, sequences):
        bad = set(sequences["event_type"].unique()) - {"Mint", "Burn", "Collect"}
        assert not bad, f"Unexpected event_type values: {bad}"

    def test_seq_num_starts_at_zero(self, sequences):
        first_seqs = sequences.groupby("position_id")["seq_num"].min()
        bad = first_seqs[first_seqs != 0]
        assert bad.empty, f"{len(bad)} position(s) where seq_num doesn't start at 0"

    def test_all_position_ids_in_summary(self, sequences, lp_summary):
        seq_ids  = set(sequences["position_id"].unique())
        summ_ids = set(lp_summary["position_id"].unique())
        orphans  = seq_ids - summ_ids
        assert not orphans, f"{len(orphans)} position_id(s) in sequences not in lp_summary"


# ── pipeline output smoke test (reads existing Parquet — no re-run) ──────────────────────────────────────────

class TestPipelineOutput:
    """
    Reads existing Parquet output (written by the last pipeline run) and checks
    structural invariants. Does NOT re-run the pipeline — fast, no Postgres.
    Skips if output files don't exist yet.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _require_output(self):
        from features.pipeline import BASE_DATA_DIR
        out = BASE_DATA_DIR / CELO_CHAIN / CELO_POOL.lower() / "lp_features.parquet"
        if not out.exists():
            pytest.skip("No pipeline output found — run features.pipeline first")

    def _out(self, filename):
        from features.pipeline import BASE_DATA_DIR
        return BASE_DATA_DIR / CELO_CHAIN / CELO_POOL.lower() / filename

    def test_no_duplicate_position_ids(self):
        df = pd.read_parquet(self._out("lp_features.parquet"))
        dupes = df[df["position_id"].duplicated()]
        assert dupes.empty, f"{len(dupes)} duplicate position_id(s) in lp_features"

    def test_no_null_exit_type(self):
        df = pd.read_parquet(self._out("lp_features.parquet"))
        nulls = df["exit_type"].isna().sum()
        assert nulls == 0, f"{nulls} null exit_type in lp_features"


# ── ingestion/decode_events unit tests (no network, no Postgres) ──────────────

class TestDecodeEvents:
    """
    Unit tests for ingestion/decode_events.py decoder functions.
    All fixtures are synthetic — no HyperSync or Postgres required.
    """

    def _make_logs_row(self, event_name: str) -> dict:
        """Return a minimal synthetic raw log row for the given event type."""
        from ingestion.decode_events import TOPIC0_EVENT_MAP
        topic0 = next(k for k, v in TOPIC0_EVENT_MAP.items() if v == event_name)
        zeros64 = "0" * 64
        # Synthetic addresses encoded as 32-byte padded topics
        owner_topic  = "0x" + "0" * 24 + "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        sender_topic = "0x" + "0" * 24 + "1234567812345678123456781234567812345678"
        # tick_lower = -100, tick_upper = 100 (sign-extended as int256 in topic)
        tick_lower_hex = "0x" + hex(-100 % (2**256))[2:].zfill(64)
        tick_upper_hex = "0x" + hex(100)[2:].zfill(64)

        base = {
            "block_number": "12345",
            "transaction_hash": "0xabc",
            "transaction_index": "0",
            "log_index": "0",
            "timestamp": "0x680fa8d8",
            "chain_name": "celo",
            "pool_address": "0xf55791",
            "topic0": topic0,
            "topic1": owner_topic,
            "topic2": tick_lower_hex,
            "topic3": tick_upper_hex,
            "data": "0x" + zeros64 * 4,  # 4 zero words — valid for Mint/Burn
        }
        return base

    def test_mint_decoder_returns_required_fields(self):
        from ingestion.decode_events import _decode_mint
        row = pd.Series(self._make_logs_row("Mint"))
        result = _decode_mint(row)
        assert set(result.keys()) == {"sender", "owner", "tick_lower", "tick_upper",
                                      "amount", "amount0", "amount1"}

    def test_burn_decoder_returns_required_fields(self):
        from ingestion.decode_events import _decode_burn
        row = pd.Series(self._make_logs_row("Burn"))
        row["data"] = "0x" + "0" * 192  # 3 words for Burn
        result = _decode_burn(row)
        assert set(result.keys()) == {"owner", "tick_lower", "tick_upper",
                                      "amount", "amount0", "amount1"}

    def test_swap_decoder_returns_required_fields(self):
        from ingestion.decode_events import _decode_swap
        row = pd.Series(self._make_logs_row("Swap"))
        row["data"] = "0x" + "0" * 320  # 5 words for Swap
        result = _decode_swap(row)
        assert set(result.keys()) == {"sender", "recipient", "amount0", "amount1",
                                      "sqrt_price_x96", "liquidity", "tick"}

    def test_tick_sign_extension(self):
        """_to_int24 must handle negative ticks correctly (two's complement)."""
        from ingestion.decode_events import _to_int24
        # -887272 is the minimum V3 tick, stored as sign-extended int256 in topic
        raw = hex(-887272 % (2**256))[2:].zfill(64)
        assert _to_int24(raw) == -887272

    def test_topic0_map_has_five_events(self):
        from ingestion.decode_events import TOPIC0_EVENT_MAP
        expected = {"Mint", "Burn", "Swap", "Initialize", "Collect"}
        assert set(TOPIC0_EVENT_MAP.values()) == expected

    def test_decode_writes_parquet(self, tmp_path):
        """decode() on synthetic raw Parquet writes decoded files correctly."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from ingestion.decode_events import TOPIC0_EVENT_MAP, decode

        # Build minimal raw logs parquet
        zeros64 = "0" * 64
        owner_topic  = "0x" + "0" * 24 + "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        tick_low_hex = "0x" + hex(-100 % (2**256))[2:].zfill(64)
        tick_hi_hex  = "0x" + hex(100)[2:].zfill(64)
        topic0_mint  = next(k for k, v in TOPIC0_EVENT_MAP.items() if v == "Mint")

        logs_rows = [
            {
                "block_number": "100",
                "transaction_hash": "0xabc",
                "transaction_index": "0",
                "log_index": "0",
                "chain_name": "celo",
                "pool_address": "0xtest",
                "topic0": topic0_mint,
                "topic1": owner_topic,
                "topic2": tick_low_hex,
                "topic3": tick_hi_hex,
                "data": "0x" + zeros64 * 4,
                "address": "0xtest",
            }
        ]
        blocks_rows = [
            {"number": "100", "timestamp": "0x680fa8d8", "chain_name": "celo"}
        ]

        chain, pool = "celo", "0xtest"
        raw_dir = tmp_path / "raw" / chain / pool
        raw_dir.mkdir(parents=True)
        pd.DataFrame(logs_rows).to_parquet(raw_dir / "logs.parquet", index=False)
        pd.DataFrame(blocks_rows).to_parquet(raw_dir / "blocks.parquet", index=False)

        counts = decode(chain=chain, pool_address=pool, data_dir=tmp_path)

        assert counts.get("Mint", 0) == 1
        decoded_path = tmp_path / "decoded" / chain / pool / "lp_mint_events.parquet"
        assert decoded_path.exists()
        df = pd.read_parquet(decoded_path)
        assert "owner" in df.columns
        assert "tick_lower" in df.columns



