#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws

BUCKET_NAME="${BUCKET_NAME:-$(stack_output ArtifactBucketName)}"
ROLE_ARN="${ROLE_ARN:-$(stack_output SageMakerExecutionRoleArn)}"
REPOSITORY_URI="${REPOSITORY_URI:-$(stack_output TrainingRepositoryUri)}"
IMAGE_TAG="${IMAGE_TAG:-v001}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.large}"
TRAINING_JOB_NAME="${TRAINING_JOB_NAME:-${EXPERIMENT_NAME}-$(date +%Y%m%d-%H%M%S)}"
CONFIG_S3_URI="${CONFIG_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/configs/${EXPERIMENT_NAME}.yaml}"
OUTPUT_S3_URI="${OUTPUT_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/experiments/${EXPERIMENT_NAME}/training-output/}"

INPUT_CONFIG_FILE="$(mktemp)"
cat > "$INPUT_CONFIG_FILE" <<JSON
[
  {
    "ChannelName": "config",
    "DataSource": {
      "S3DataSource": {
        "S3DataType": "S3Prefix",
        "S3Uri": "${CONFIG_S3_URI}",
        "S3DataDistributionType": "FullyReplicated"
      }
    },
    "ContentType": "application/x-yaml",
    "InputMode": "File"
  }
]
JSON

aws sagemaker create-training-job \
  --region "$AWS_REGION" \
  --training-job-name "$TRAINING_JOB_NAME" \
  --role-arn "$ROLE_ARN" \
  --algorithm-specification "TrainingImage=${REPOSITORY_URI}:${IMAGE_TAG},TrainingInputMode=File" \
  --input-data-config "file://${INPUT_CONFIG_FILE}" \
  --output-data-config "S3OutputPath=${OUTPUT_S3_URI}" \
  --resource-config "InstanceType=${INSTANCE_TYPE},InstanceCount=1,VolumeSizeInGB=30" \
  --stopping-condition MaxRuntimeInSeconds=3600 \
  --enable-network-isolation false \
  --tags "Key=Project,Value=autodriving-rl" "Key=Experiment,Value=${EXPERIMENT_NAME}"

rm -f "$INPUT_CONFIG_FILE"

echo "Started training job: $TRAINING_JOB_NAME"
echo "Watch:"
echo "aws sagemaker describe-training-job --region $AWS_REGION --training-job-name $TRAINING_JOB_NAME"
echo "Expected model artifact:"
echo "${OUTPUT_S3_URI}${TRAINING_JOB_NAME}/output/model.tar.gz"
