"""
analysis/plots.py
──────────────────
Matplotlib figure builders for survival analysis output.
Returns Figure objects — rendering is the caller's responsibility
(plt.show(), st.pyplot(), savefig(), etc.).
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def km_figure(km_result: dict, title: str = "LP Retention Curve") -> plt.Figure:
    """
    Render a Kaplan-Meier survival curve comparing campaign vs non-campaign LPs.

    Parameters
    ----------
    km_result : dict returned by analysis.survival.run_km()
    title     : figure title (pool label recommended)

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    km_result["kmf_campaign"].plot_survival_function(
        ax=ax, ci_show=True, color="#3b82f6", linewidth=2
    )
    km_result["kmf_non_campaign"].plot_survival_function(
        ax=ax, ci_show=True, color="#f59e0b", linewidth=2
    )

    # Median survival markers
    for kmf, color in [
        (km_result["kmf_campaign"],     "#3b82f6"),
        (km_result["kmf_non_campaign"], "#f59e0b"),
    ]:
        median = kmf.median_survival_time_
        if median < float("inf"):
            ax.axvline(median, color=color, linestyle="--", alpha=0.5, linewidth=1)

    p = km_result["logrank"].p_value
    p_label = f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001"
    sig = "significant" if p < 0.05 else "not significant"
    ax.set_title(f"{title}\nlog-rank {p_label} ({sig})", fontsize=13, pad=12)
    ax.set_xlabel("Days since first mint", fontsize=11)
    ax.set_ylabel("Fraction still active", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


def duration_histogram(lp_features, title: str = "Duration Distribution") -> plt.Figure:
    """
    Histogram of LP duration by cohort.

    Parameters
    ----------
    lp_features : DataFrame from lp_features.parquet
    """
    import pandas as pd

    df = lp_features.copy()
    df["duration_days"] = df["duration_seconds"] / 86400

    cohort_colors = {
        "during_campaign": "#3b82f6",
        "pre_campaign":    "#f59e0b",
        "post_campaign":   "#10b981",
    }

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for cohort, color in cohort_colors.items():
        subset = df[df["lp_cohort"] == cohort]["duration_days"]
        if not subset.empty:
            ax.hist(subset, bins=60, alpha=0.55, label=cohort, color=color)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Duration (days)", fontsize=10)
    ax.set_ylabel("Positions", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
