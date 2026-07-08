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
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.m5.large}"
PROCESSING_JOB_NAME="${PROCESSING_JOB_NAME:-eval-${TRAINING_JOB_NAME}}"
CONFIG_S3_URI="${CONFIG_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/configs/${EXPERIMENT_NAME}.yaml}"
MODEL_S3_URI="${MODEL_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/experiments/${EXPERIMENT_NAME}/training-output/${TRAINING_JOB_NAME}/output/model.tar.gz}"
EVAL_OUTPUT_S3_URI="${EVAL_OUTPUT_S3_URI:-s3://${BUCKET_NAME}/autodriving-rl/experiments/${EXPERIMENT_NAME}/evaluation/${TRAINING_JOB_NAME}/}"

APP_SPEC_FILE="$(mktemp)"
INPUTS_FILE="$(mktemp)"
OUTPUTS_FILE="$(mktemp)"

cat > "$APP_SPEC_FILE" <<JSON
{
  "ImageUri": "${REPOSITORY_URI}:${IMAGE_TAG}",
  "ContainerEntrypoint": ["python", "/opt/program/evaluate.py"]
}
JSON

cat > "$INPUTS_FILE" <<JSON
[
  {
    "InputName": "config",
    "S3Input": {
      "S3Uri": "${CONFIG_S3_URI}",
      "LocalPath": "/opt/ml/processing/input/config",
      "S3DataType": "S3Prefix",
      "S3InputMode": "File",
      "S3DataDistributionType": "FullyReplicated",
      "S3CompressionType": "None"
    }
  },
  {
    "InputName": "model",
    "S3Input": {
      "S3Uri": "${MODEL_S3_URI}",
      "LocalPath": "/opt/ml/processing/input/model",
      "S3DataType": "S3Prefix",
      "S3InputMode": "File",
      "S3DataDistributionType": "FullyReplicated",
      "S3CompressionType": "None"
    }
  }
]
JSON

cat > "$OUTPUTS_FILE" <<JSON
{
  "Outputs": [
    {
      "OutputName": "evaluation",
      "S3Output": {
        "S3Uri": "${EVAL_OUTPUT_S3_URI}",
        "LocalPath": "/opt/ml/processing/output",
        "S3UploadMode": "EndOfJob"
      }
    }
  ]
}
JSON

aws sagemaker create-processing-job \
  --region "$AWS_REGION" \
  --processing-job-name "$PROCESSING_JOB_NAME" \
  --role-arn "$ROLE_ARN" \
  --app-specification "file://${APP_SPEC_FILE}" \
  --processing-resources "ClusterConfig={InstanceCount=1,InstanceType=${INSTANCE_TYPE},VolumeSizeInGB=30}" \
  --processing-inputs "file://${INPUTS_FILE}" \
  --processing-output-config "file://${OUTPUTS_FILE}" \
  --stopping-condition MaxRuntimeInSeconds=1800 \
  --tags "Key=Project,Value=autodriving-rl" "Key=Experiment,Value=${EXPERIMENT_NAME}"

rm -f "$APP_SPEC_FILE" "$INPUTS_FILE" "$OUTPUTS_FILE"

echo "Started processing job: $PROCESSING_JOB_NAME"
echo "Evaluation output:"
echo "$EVAL_OUTPUT_S3_URI"
