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
ROLE_ARN="${ROLE_ARN:-$(stack_output SageMakerExecutionRoleArn)}"
REPOSITORY_URI="${REPOSITORY_URI:-$(stack_output TrainingRepositoryUri)}"
IMAGE_TAG="${IMAGE_TAG:-v001}"
MODEL_PACKAGE_GROUP_NAME="${MODEL_PACKAGE_GROUP_NAME:-autodriving-rl-highway-ppo}"
MODEL_S3_URI="${MODEL_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/experiments/${EXPERIMENT_NAME}/training-output/${TRAINING_JOB_NAME}/output/model.tar.gz}"

if ! aws sagemaker describe-model-package-group \
  --region "$AWS_REGION" \
  --model-package-group-name "$MODEL_PACKAGE_GROUP_NAME" >/dev/null 2>&1; then
  aws sagemaker create-model-package-group \
    --region "$AWS_REGION" \
    --model-package-group-name "$MODEL_PACKAGE_GROUP_NAME" \
    --model-package-group-description "Candidate RL policies for autodriving-rl minimal highway PPO experiments"
fi

aws sagemaker create-model-package \
  --region "$AWS_REGION" \
  --model-package-group-name "$MODEL_PACKAGE_GROUP_NAME" \
  --model-package-description "Model package for ${TRAINING_JOB_NAME}" \
  --model-approval-status PendingManualApproval \
  --inference-specification "{
    \"Containers\": [
      {
        \"Image\": \"${REPOSITORY_URI}:${IMAGE_TAG}\",
        \"ModelDataUrl\": \"${MODEL_S3_URI}\"
      }
    ],
    \"SupportedContentTypes\": [\"application/json\"],
    \"SupportedResponseMIMETypes\": [\"application/json\"]
  }" \
  --customer-metadata-properties "Experiment=${EXPERIMENT_NAME},TrainingJobName=${TRAINING_JOB_NAME},ModelArtifact=${MODEL_S3_URI},ExecutionRole=${ROLE_ARN}" \
  --query ModelPackageArn \
  --output text

