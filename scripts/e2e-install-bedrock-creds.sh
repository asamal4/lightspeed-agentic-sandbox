#!/usr/bin/env bash
# Sync AWS Bedrock credentials into a cluster Secret for sandbox batch e2e Jobs.
#
# Konflux workspace secret bedrock-apitoken mounts keys aws_access_key_id and
# aws_secret_access_key under /var/run/credentials/. Batch Jobs expect
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY via envFrom.
#
# Optional env:
#   E2E_NAMESPACE   default: openshift-lightspeed
#   SECRET_NAME     default: llm-creds-bedrock

set -euo pipefail

E2E_NAMESPACE="${E2E_NAMESPACE:-openshift-lightspeed}"
SECRET_NAME="${SECRET_NAME:-llm-creds-bedrock}"
CRED_DIR="/var/run/credentials"

if ! command -v oc >/dev/null 2>&1; then
    echo "error: oc not found" >&2
    exit 1
fi

_read_key() {
    local env_val="$1"
    local file_name="$2"
    if [ -n "${env_val}" ]; then
        printf '%s' "${env_val}"
        return 0
    fi
    if [ -f "${CRED_DIR}/${file_name}" ]; then
        tr -d '[:space:]' < "${CRED_DIR}/${file_name}"
        return 0
    fi
    return 1
}

if ! AWS_ACCESS_KEY_ID="$(_read_key "${AWS_ACCESS_KEY_ID:-}" "aws_access_key_id")"; then
    echo "error: set AWS_ACCESS_KEY_ID or mount Konflux bedrock-apitoken" >&2
    exit 1
fi
if ! AWS_SECRET_ACCESS_KEY="$(_read_key "${AWS_SECRET_ACCESS_KEY:-}" "aws_secret_access_key")"; then
    echo "error: set AWS_SECRET_ACCESS_KEY or mount Konflux bedrock-apitoken" >&2
    exit 1
fi

if [ -z "${AWS_ACCESS_KEY_ID}" ] || [ -z "${AWS_SECRET_ACCESS_KEY}" ]; then
    echo "error: AWS credentials are empty" >&2
    exit 1
fi

tmp_access=$(mktemp)
tmp_secret=$(mktemp)
chmod 600 "${tmp_access}" "${tmp_secret}"
trap 'rm -f "${tmp_access}" "${tmp_secret}"' EXIT
printf '%s' "${AWS_ACCESS_KEY_ID}" > "${tmp_access}"
printf '%s' "${AWS_SECRET_ACCESS_KEY}" > "${tmp_secret}"

oc create secret generic "${SECRET_NAME}" \
    --namespace="${E2E_NAMESPACE}" \
    --from-file=AWS_ACCESS_KEY_ID="${tmp_access}" \
    --from-file=AWS_SECRET_ACCESS_KEY="${tmp_secret}" \
    --dry-run=client -o yaml | oc apply -f -

echo "Created/updated Secret ${SECRET_NAME} in ${E2E_NAMESPACE}"
echo
echo "Mount in batch Job via envFrom:"
echo "  envFrom:"
echo "    - secretRef:"
echo "        name: ${SECRET_NAME}"
echo
echo "Provider env for Bedrock:"
echo "  LIGHTSPEED_PROVIDER=bedrock"
echo "  LIGHTSPEED_MODEL=global.anthropic.claude-haiku-5-5"
echo "  AWS_REGION=us-east-1                 # Bedrock region"
