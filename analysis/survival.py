"""
analysis/survival.py
─────────────────────
Kaplan-Meier survival analysis on lp_features.parquet output.

Compares LP retention between cohorts:
  - campaign (during_campaign) vs non-campaign (pre_campaign + post_campaign)

Returns fitted KMFitter objects + log-rank test result for downstream
plotting or Streamlit rendering.
"""

import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def run_km(lp_features: pd.DataFrame) -> dict:
    """
    Fit KM survival curves for campaign vs non-campaign LP cohorts.

    Parameters
    ----------
    lp_features : DataFrame from lp_features.parquet

    Returns
    -------
    dict with keys:
        kmf_campaign      : KaplanMeierFitter (fitted, label set)
        kmf_non_campaign  : KaplanMeierFitter (fitted, label set)
        logrank           : StatisticalResult from lifelines
        n_campaign        : int
        n_non_campaign    : int
        median_campaign   : float  (median survival in days)
        median_non_campaign : float
    """
    df = lp_features.copy()
    df["duration_days"] = df["duration_seconds"] / 86400

    campaign     = df[df["lp_cohort"] == "during_campaign"]
    non_campaign = df[df["lp_cohort"].isin(["pre_campaign", "post_campaign"])]

    kmf_c = KaplanMeierFitter(label=f"Campaign LPs (n={len(campaign):,})")
    kmf_c.fit(
        campaign["duration_days"],
        event_observed=(campaign["status"] == 1),
    )

    kmf_nc = KaplanMeierFitter(label=f"Non-campaign LPs (n={len(non_campaign):,})")
    kmf_nc.fit(
        non_campaign["duration_days"],
        event_observed=(non_campaign["status"] == 1),
    )

    result = logrank_test(
        campaign["duration_days"],
        non_campaign["duration_days"],
        event_observed_A=(campaign["status"] == 1),
        event_observed_B=(non_campaign["status"] == 1),
    )

    return {
        "kmf_campaign":         kmf_c,
        "kmf_non_campaign":     kmf_nc,
        "logrank":              result,
        "n_campaign":           len(campaign),
        "n_non_campaign":       len(non_campaign),
        "median_campaign":      float(kmf_c.median_survival_time_),
        "median_non_campaign":  float(kmf_nc.median_survival_time_),
    }
