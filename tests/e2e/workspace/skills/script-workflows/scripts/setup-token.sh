#!/usr/bin/env bash
set -euo pipefail

SKILL_CHAIN_TOKEN=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
export SKILL_CHAIN_TOKEN
