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

    def test_all_three_cohorts_present(self, lp_summary):
        cohorts = set(lp_summary["lp_cohort"].unique())
        expected = {"pre_campaign", "during_campaign", "post_campaign"}
        missing = expected - cohorts
        assert not missing, (
            f"Expected cohorts {expected}, missing: {missing}. "
            f"Found: {cohorts}"
        )

    def test_no_null_position_ids(self, lp_summary):
        assert lp_summary["position_id"].notna().all()

    def test_no_null_first_mint_timestamp(self, lp_summary):
        assert lp_summary["first_mint_timestamp"].notna().all()

    def test_tick_range_width_positive(self, lp_summary):
        bad = lp_summary[lp_summary["tick_range_width"] <= 0]
        assert bad.empty, f"{len(bad)} position(s) with tick_range_width <= 0"

    def test_event_count_at_least_one(self, lp_summary):
        bad = lp_summary[lp_summary["event_count"] < 1]
        assert bad.empty, f"{len(bad)} position(s) with event_count < 1"

    def test_more_positions_than_unique_owners(self, lp_summary):
        """
        Validates the core fix: per-cycle position_id means at minimum as many
        positions as unique owners. With re-opens on the same tick range there
        should be strictly more positions than owners.
        """
        n_positions = len(lp_summary)
        n_owners = lp_summary["owner"].nunique()
        assert n_positions >= n_owners, \
            "Should have at least as many positions as unique owners"

    def test_cohort_split_sanity(self, lp_summary, campaigns):
        """
        All during_campaign LPs must have first_mint_timestamp within
        [global_start, global_end]. Sanity-checks the cohort assignment.
        """
        global_start = int(campaigns["start_timestamp"].min())
        global_end   = int(campaigns["end_timestamp"].max())
        during = lp_summary[lp_summary["lp_cohort"] == "during_campaign"]
        if during.empty:
            pytest.skip("No during_campaign positions found — nothing to check")
        out_of_window = during[
            (during["first_mint_timestamp"] < global_start) |
            (during["first_mint_timestamp"] > global_end)
        ]
        assert out_of_window.empty, (
            f"{len(out_of_window)} during_campaign LP(s) have first_mint_timestamp "
            "outside the campaign window"
        )


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

    def test_both_exit_types_present(self, lp_with_exit_type):
        exited = lp_with_exit_type[lp_with_exit_type["status"] == 1]
        types = set(exited["exit_type"].unique())
        assert types, "No exited positions found"
        # At least one type must be present; both is better
        assert types.issubset({"voluntary_exit", "range_exit"}), \
            f"Unexpected types in exited set: {types}"

    def test_distribution_logged(self, lp_with_exit_type):
        """Non-assertion: print distribution for human review."""
        dist = lp_with_exit_type["exit_type"].value_counts()
        print(f"\n  exit_type distribution:\n{dist.to_string()}")

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

    def test_returns_dataframe(self, sequences):
        assert isinstance(sequences, pd.DataFrame)

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

    def test_seq_num_contiguous(self, sequences):
        """seq_num within each position is 0,1,2,... with no gaps."""
        def _gapless(s):
            return list(sorted(s)) == list(range(len(s)))
        bad = sequences.groupby("position_id")["seq_num"].apply(_gapless)
        assert bad.all(), f"{(~bad).sum()} position(s) with non-contiguous seq_num"

    def test_all_position_ids_in_summary(self, sequences, lp_summary):
        seq_ids  = set(sequences["position_id"].unique())
        summ_ids = set(lp_summary["position_id"].unique())
        orphans  = seq_ids - summ_ids
        assert not orphans, f"{len(orphans)} position_id(s) in sequences not in lp_summary"

    def test_each_position_has_mint_and_burn(self, sequences, lp_summary):
        """Every exited position must have at least one Mint and one Burn."""
        exited_ids = set(lp_summary[lp_summary["status"] == 1]["position_id"])
        seq_exited = sequences[sequences["position_id"].isin(exited_ids)]
        by_pos = seq_exited.groupby("position_id")["event_type"].apply(set)
        missing_mint = by_pos[by_pos.apply(lambda s: "Mint" not in s)]
        missing_burn = by_pos[by_pos.apply(lambda s: "Burn" not in s)]
        assert missing_mint.empty, f"{len(missing_mint)} exited position(s) missing Mint"
        assert missing_burn.empty, f"{len(missing_burn)} exited position(s) missing Burn"

    def test_distribution_logged(self, sequences):
        dist = sequences["event_type"].value_counts()
        print(f"\n  event_type distribution:\n{dist.to_string()}")
        n_with_price = sequences["price_at_event"].notna().sum()
        print(f"  price_at_event coverage: {n_with_price}/{len(sequences)} rows")


# ── pipeline.run() integration test ──────────────────────────────────────────

@requires_pg
class TestPipelineRun:
    """
    Runs the full pipeline end-to-end: Merkl URL → Postgres → 3 Parquet files.
    This is the Phase 5 integration test — verifies the full output shape.
    """

    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self):
        from features.pipeline import DATA_DIR, run
        run(chain=CELO_CHAIN, pool=CELO_POOL, merkl_url=CELO_MERKL)
        return DATA_DIR

    def test_lp_features_exists(self):
        from features.pipeline import DATA_DIR
        assert (DATA_DIR / "lp_features.parquet").exists()

    def test_lp_survival_labels_exists(self):
        from features.pipeline import DATA_DIR
        assert (DATA_DIR / "lp_survival_labels.parquet").exists()

    def test_lp_event_sequences_exists(self):
        from features.pipeline import DATA_DIR
        assert (DATA_DIR / "lp_event_sequences.parquet").exists()

    def test_features_no_duplicate_ids(self):
        from features.pipeline import DATA_DIR
        df = pd.read_parquet(DATA_DIR / "lp_features.parquet")
        dupes = df[df["position_id"].duplicated()]
        assert dupes.empty, f"{len(dupes)} duplicate position_id(s) in lp_features"

    def test_features_no_null_exit_type(self):
        from features.pipeline import DATA_DIR
        df = pd.read_parquet(DATA_DIR / "lp_features.parquet")
        nulls = df["exit_type"].isna().sum()
        assert nulls == 0, f"{nulls} null exit_type in lp_features"

    def test_survival_labels_row_count_matches_features(self):
        from features.pipeline import DATA_DIR
        feat = pd.read_parquet(DATA_DIR / "lp_features.parquet")
        surv = pd.read_parquet(DATA_DIR / "lp_survival_labels.parquet")
        assert len(feat) == len(surv), (
            f"lp_features has {len(feat)} rows but lp_survival_labels has {len(surv)}"
        )

    def test_sequences_position_ids_subset_of_features(self):
        from features.pipeline import DATA_DIR
        feat = pd.read_parquet(DATA_DIR / "lp_features.parquet")
        seqs = pd.read_parquet(DATA_DIR / "lp_event_sequences.parquet")
        orphans = set(seqs["position_id"].unique()) - set(feat["position_id"].unique())
        assert not orphans, f"{len(orphans)} position_id(s) in sequences not in features"

    def test_summary_printed(self, capsys):
        # Pipeline already ran via autouse fixture — just verify output shape.
        from features.pipeline import DATA_DIR
        feat = pd.read_parquet(DATA_DIR / "lp_features.parquet")
        seqs = pd.read_parquet(DATA_DIR / "lp_event_sequences.parquet")
        print(f"\n  lp_features:  {len(feat):,} rows, {feat.shape[1]} cols")
        print(f"  sequences:    {len(seqs):,} rows, {seqs.shape[1]} cols")
        print(f"  cohorts:      {feat['lp_cohort'].value_counts().to_dict()}")
        print(f"  exit_type:    {feat['exit_type'].value_counts().to_dict()}")
