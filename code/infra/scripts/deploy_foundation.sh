#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws

PROJECT_NAME="${PROJECT_NAME:-autodriving-rl}"
ARTIFACT_BUCKET_NAME="${ARTIFACT_BUCKET_NAME:-}"
ECR_REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-autodriving-rl-training}"
ENABLE_SAGEMAKER_FULL_ACCESS="${ENABLE_SAGEMAKER_FULL_ACCESS:-false}"

aws cloudformation deploy \
  --template-file "$PROJECT_ROOT/infra/cloudformation/foundation.yaml" \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
    ProjectName="$PROJECT_NAME" \
    ArtifactBucketName="$ARTIFACT_BUCKET_NAME" \
    EcrRepositoryName="$ECR_REPOSITORY_NAME" \
    EnableSageMakerFullAccess="$ENABLE_SAGEMAKER_FULL_ACCESS"

echo "Stack deployed: $STACK_NAME"
echo "ArtifactBucketName=$(stack_output ArtifactBucketName)"
echo "TrainingRepositoryUri=$(stack_output TrainingRepositoryUri)"
echo "SageMakerExecutionRoleArn=$(stack_output SageMakerExecutionRoleArn)"

