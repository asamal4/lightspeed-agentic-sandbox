#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "expected exactly one argument" >&2
    exit 2
fi

TOKEN=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
printf '{"summary":"argument:%s","token":"%s","status":"ok"}\n' "$1" "$TOKEN"
