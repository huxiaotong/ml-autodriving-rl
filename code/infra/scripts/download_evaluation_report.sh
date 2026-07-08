#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TRAINING_JOB_NAME" >&2
  exit 1
fi

TRAINING_JOB_NAME="$1"
BUCKET_NAME="${BUCKET_NAME:-$(stack_output ArtifactBucketName)}"
REPORT_S3_URI="${REPORT_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/experiments/${EXPERIMENT_NAME}/evaluation/${TRAINING_JOB_NAME}/evaluation_report.json}"
LOCAL_REPORT="${LOCAL_REPORT:-$PROJECT_ROOT/outputs/${TRAINING_JOB_NAME}-evaluation_report.json}"

mkdir -p "$(dirname "$LOCAL_REPORT")"
aws s3 cp "$REPORT_S3_URI" "$LOCAL_REPORT" --region "$AWS_REGION"

echo "Downloaded:"
echo "$LOCAL_REPORT"

