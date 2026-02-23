#!/bin/bash
set -e

# ── 1. Sync project (volume-mounted at /app) ──────────────────────────────
if [ -f "/app/pyproject.toml" ]; then
    echo ">> Syncing openpi project from /app ..."
    cd /app
    GIT_LFS_SKIP_SMUDGE=1 uv sync 2>&1 | tail -5
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . > /dev/null 2>&1
    uv pip install "policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git" > /dev/null 2>&1
    cd - > /dev/null
fi

# ── 2. Optional: run environment verification ─────────────────────────────
# Pass --verify as the first argument to run tests/test_env_verification.py
# and exit with the test result.
#
# Usage:
#   docker exec <container> /usr/local/bin/entrypoint.sh --verify
#   docker compose ... run --rm openpi-dev --verify
if [ "${1}" = "--verify" ]; then
    echo ">> Running environment verification tests ..."
    shift
    exec /.venv/bin/python -m pytest tests/test_env_verification.py -v "$@"
fi

echo ">> Environment ready."

# ── 3. Execute user command (default: /bin/bash from CMD) ─────────────────
exec "$@"
