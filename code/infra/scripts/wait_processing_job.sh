#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PROCESSING_JOB_NAME" >&2
  exit 1
fi

PROCESSING_JOB_NAME="$1"

aws sagemaker wait processing-job-completed-or-stopped \
  --region "$AWS_REGION" \
  --processing-job-name "$PROCESSING_JOB_NAME"

aws sagemaker describe-processing-job \
  --region "$AWS_REGION" \
  --processing-job-name "$PROCESSING_JOB_NAME" \
  --query "{ProcessingJobStatus:ProcessingJobStatus,FailureReason:FailureReason}" \
  --output json

