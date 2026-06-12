"""
features/merkl.py
──────────────────
Minimal Merkl API client for fetching campaign windows for a given pool.

The only thing the feature pipeline needs from Merkl is:
    start_timestamp, end_timestamp  (Unix seconds)
per campaign, for a given opportunity (pool + chain).

This is intentionally thin — no caching, no retry, no persistence.
Once the pipeline moves to GCP, this becomes a Composer connection or
a Cloud Function that writes to BigQuery. The pandas caller doesn't change.
"""

from __future__ import annotations

import http.client
import json
from collections import OrderedDict

import pandas as pd


def _get_json(path: str) -> object:
    conn = http.client.HTTPSConnection("api.merkl.xyz")
    conn.request("GET", path, headers={"Accept": "*/*"})
    res = conn.getresponse()
    return json.loads(res.read().decode("utf-8"), object_pairs_hook=OrderedDict)


def _opportunity_id(chain_name: str, pool_type: str, explorer_address: str) -> str:
    data = _get_json(
        f"/v4/opportunities?chainName={chain_name}&type={pool_type}"
        f"&explorerAddress={explorer_address}"
    )
    if not data:
        raise ValueError(
            f"No Merkl opportunity found for chain={chain_name} "
            f"type={pool_type} address={explorer_address}"
        )
    return str(data[0]["id"])


def _parse_merkl_url(url: str) -> tuple[str, str, str]:
    """
    Parse a Merkl opportunity URL into (chain_name, pool_type, explorer_address).

    Expected format:
        https://app.merkl.xyz/opportunities/{chain}/{type}/{address}

    Example:
        https://app.merkl.xyz/opportunities/celo/CLAMM/0xF55791...
        → ("celo", "CLAMM", "0xF55791...")
    """
    parts = url.rstrip("/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse Merkl URL: {url!r}")
    return parts[-3], parts[-2], parts[-1]


def fetch_campaign_windows(merkl_url: str) -> pd.DataFrame:
    """
    Fetch all campaign windows for a Merkl opportunity URL.

    Parameters
    ----------
    merkl_url : str
        E.g. "https://app.merkl.xyz/opportunities/celo/CLAMM/0xF55791..."

    Returns
    -------
    DataFrame with columns:
        campaign_id       str
        start_timestamp   int  (Unix seconds)
        end_timestamp     int  (Unix seconds)
        opportunity_id    str
    """
    chain_name, pool_type, explorer_address = _parse_merkl_url(merkl_url)
    opp_id = _opportunity_id(chain_name, pool_type, explorer_address)

    campaigns_raw = _get_json(f"/v4/campaigns?opportunityId={opp_id}")

    rows = []
    for item in campaigns_raw:
        c = dict(item) if isinstance(item, OrderedDict) else item
        start = c.get("startTimestamp")
        end = c.get("endTimestamp")
        if start is None or end is None:
            continue
        rows.append(
            {
                "campaign_id": str(c.get("id", "")),
                "opportunity_id": opp_id,
                "start_timestamp": int(start),
                "end_timestamp": int(end),
            }
        )

    if not rows:
        raise ValueError(f"No campaigns found for opportunity {opp_id} (URL: {merkl_url!r})")

    return pd.DataFrame(rows)
