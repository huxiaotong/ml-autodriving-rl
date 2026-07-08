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

aws sagemaker wait training-job-completed-or-stopped \
  --region "$AWS_REGION" \
  --training-job-name "$TRAINING_JOB_NAME"

aws sagemaker describe-training-job \
  --region "$AWS_REGION" \
  --training-job-name "$TRAINING_JOB_NAME" \
  --query "{TrainingJobStatus:TrainingJobStatus,ModelArtifacts:ModelArtifacts.S3ModelArtifacts,FailureReason:FailureReason}" \
  --output json

