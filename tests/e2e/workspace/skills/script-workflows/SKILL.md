---
name: script-workflows
description: Run one bundled script with an argument or a setup-and-check script chain.
---

# Script workflows

Use these procedures when asked to verify a single script or a command chain. Work from this skill's directory and use the JSON printed by the scripts as the source for your answer.

## Single-file procedure

Run `bash scripts/single-file.sh "<argument>"` with the exact argument provided in the request. Return the JSON printed by the script without changing its `summary`, `token`, or `status` fields.

## Command-chain procedure

Run the setup and check scripts as one shell command so the setup value remains in the same shell. Source the setup script; executing it in a child shell will not preserve its exported value:

```bash
source scripts/setup-token.sh && bash scripts/check-token.sh
```

Return the JSON printed by the check script without changing its fields. The check fails unless it receives the value exported by the setup script.
