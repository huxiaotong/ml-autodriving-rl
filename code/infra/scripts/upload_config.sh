#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/highway-ppo-v001.yaml}"
BUCKET_NAME="${BUCKET_NAME:-$(stack_output ArtifactBucketName)}"
S3_KEY="${S3_KEY:-autodriving-rl/configs/${EXPERIMENT_NAME}.yaml}"

aws s3 cp "$CONFIG_PATH" "s3://${BUCKET_NAME}/${S3_KEY}" --region "$AWS_REGION"

echo "Uploaded config:"
echo "s3://${BUCKET_NAME}/${S3_KEY}"

