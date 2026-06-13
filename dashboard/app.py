"""
dashboard/app.py
─────────────────
LPulse Streamlit dashboard — LP retention analysis.

Run:
    streamlit run dashboard/app.py

Reads from data/{chain}/{pool}/ Parquet files written by features.pipeline.
Enumerates all available pools automatically.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.plots import duration_histogram, km_figure
from analysis.survival import run_km

BASE_DATA_DIR  = Path(__file__).parent.parent / "data"
REGISTRY_PATH  = Path(__file__).parent.parent / "ingestion" / "pools_registry.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _available_pools() -> list[dict]:
    """Enumerate pools that have lp_features.parquet output."""
    pools = []
    for path in sorted(BASE_DATA_DIR.glob("*/*/lp_features.parquet")):
        pools.append({
            "chain": path.parent.parent.name,
            "pool":  path.parent.name,
            "path":  path,
        })
    return pools


def _pool_label(entry: dict) -> str:
    try:
        with open(REGISTRY_PATH) as f:
            registry = yaml.safe_load(f)["pools"]
        match = next(
            (r for r in registry if r["pool_address"].lower() == entry["pool"].lower()),
            None,
        )
        if match:
            return f"{match['description']}  [{entry['chain']}]"
    except Exception:
        pass
    return f"{entry['chain']} / {entry['pool'][:12]}..."


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="LPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("LPulse")
st.sidebar.caption("LP retention analysis")

available = _available_pools()
if not available:
    st.error(
        "No pipeline output found. Run `python -m features.pipeline` first.\n\n"
        "Expected path: `data/{chain}/{pool}/lp_features.parquet`"
    )
    st.stop()

labels      = [_pool_label(p) for p in available]
selected_i  = st.sidebar.selectbox("Pool", range(len(labels)), format_func=lambda i: labels[i])
selected    = available[selected_i]

st.sidebar.divider()
st.sidebar.markdown(
    "**Data source:** on-chain events via HyperSync\n\n"
    "**Campaigns:** Merkl incentive windows\n\n"
    "**Model:** Kaplan-Meier (KM) survival"
)

# ── Load data ──────────────────────────────────────────────────────────────────

feat_path = selected["path"]
seq_path  = feat_path.parent / "lp_event_sequences.parquet"

lp_features = _load(str(feat_path))
sequences   = _load(str(seq_path)) if seq_path.exists() else pd.DataFrame()

pool_label = _pool_label(selected)

# ── Page header ────────────────────────────────────────────────────────────────

st.title(pool_label)
st.caption(
    "Do incentive campaigns attract sticky liquidity, or mercenary capital? "
    "KM curves compare LP retention between campaign and non-campaign entrants."
)

# ── Summary metrics ────────────────────────────────────────────────────────────

n_total    = len(lp_features)
n_exited   = int((lp_features["status"] == 1).sum())
n_censored = int((lp_features["status"] == 0).sum())
exited_df  = lp_features[lp_features["status"] == 1]
pct_range  = exited_df["exit_type"].eq("range_exit").mean()
pct_vol    = exited_df["exit_type"].eq("voluntary_exit").mean()
med_dur    = lp_features["duration_seconds"].median() / 86400

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Positions",       f"{n_total:,}")
col2.metric("Exited",          f"{n_exited:,}")
col3.metric("Censored",        f"{n_censored:,}")
col4.metric("Range exit",      f"{pct_range:.0%}")
col5.metric("Voluntary exit",  f"{pct_vol:.0%}")

st.divider()

# ── KM survival curve ──────────────────────────────────────────────────────────

st.subheader("Kaplan-Meier Retention Curve")
st.caption("Campaign LPs vs non-campaign LPs. Shaded area = 95% confidence interval. Dashed vertical = median survival.")

km_result = run_km(lp_features)
fig_km = km_figure(km_result, title=pool_label)
st.pyplot(fig_km)
plt.close(fig_km)

p_val = km_result["logrank"].p_value
med_c  = km_result["median_campaign"]
med_nc = km_result["median_non_campaign"]

col_a, col_b, col_c = st.columns(3)
col_a.metric("Log-rank p-value",            f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001",
             delta="significant" if p_val < 0.05 else "not significant",
             delta_color="normal" if p_val < 0.05 else "off")
col_b.metric("Median survival — campaign",      f"{med_c:.1f}d"  if med_c < float("inf") else "not reached")
col_c.metric("Median survival — non-campaign",  f"{med_nc:.1f}d" if med_nc < float("inf") else "not reached")

st.divider()

# ── Cohort + exit type breakdown ───────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Cohort distribution")
    cohort_counts = (
        lp_features["lp_cohort"]
        .value_counts()
        .rename_axis("cohort")
        .reset_index(name="positions")
    )
    cohort_counts["pct"] = (cohort_counts["positions"] / n_total * 100).round(1).astype(str) + "%"
    st.dataframe(cohort_counts, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Exit type distribution")
    exit_counts = (
        lp_features["exit_type"]
        .value_counts()
        .rename_axis("exit_type")
        .reset_index(name="positions")
    )
    exit_counts["pct"] = (exit_counts["positions"] / n_total * 100).round(1).astype(str) + "%"
    st.dataframe(exit_counts, use_container_width=True, hide_index=True)

st.divider()

# ── Duration histogram ─────────────────────────────────────────────────────────

st.subheader("Duration Distribution by Cohort (days)")
fig_hist = duration_histogram(lp_features, title=pool_label)
st.pyplot(fig_hist)
plt.close(fig_hist)

# ── Event sequence sample ──────────────────────────────────────────────────────

if not sequences.empty:
    st.divider()
    st.subheader("Event Sequence Sample")
    st.caption("First 200 rows of lp_event_sequences — LP-action events with price at event.")
    st.dataframe(sequences.head(200), use_container_width=True)

# ── Raw data expander ──────────────────────────────────────────────────────────

with st.expander("Raw lp_features (first 500 rows)"):
    st.dataframe(lp_features.head(500), use_container_width=True)
