"""
features/_env.py
─────────────────
Centralised env loading. Checks for a local .env in the repo root first,
then falls back to ../financial-data-platform/.env (sibling repo, dev only).

Import and call setup() once at the top of any entry point:
    from features._env import setup; setup()

python-dotenv does not override already-set vars on a second load_dotenv()
call, so the precedence chain is: shell env > local .env > FDP .env.
"""

from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FDP_ENV   = _REPO_ROOT.parent / "financial-data-platform" / ".env"


def setup() -> None:
    load_dotenv(_REPO_ROOT / ".env")   # local .env if it exists (no-op otherwise)
    load_dotenv(_FDP_ENV)              # FDP fallback — won't override already-set vars
