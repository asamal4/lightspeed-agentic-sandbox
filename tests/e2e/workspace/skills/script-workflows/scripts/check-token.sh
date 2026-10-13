#!/usr/bin/env bash
set -euo pipefail

: "${SKILL_CHAIN_TOKEN:?setup script did not export SKILL_CHAIN_TOKEN}"
printf '{"summary":"same-shell-chain","token":"%s","status":"ok"}\n' "$SKILL_CHAIN_TOKEN"
