#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

require_command aws
require_command docker

REPOSITORY_URI="${REPOSITORY_URI:-$(stack_output TrainingRepositoryUri)}"
IMAGE_TAG="${IMAGE_TAG:-v001}"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${REPOSITORY_URI%/*}"

docker build \
  --platform linux/amd64 \
  -t "${REPOSITORY_URI}:${IMAGE_TAG}" \
  "$PROJECT_ROOT/training"

docker push "${REPOSITORY_URI}:${IMAGE_TAG}"

echo "Pushed image:"
echo "${REPOSITORY_URI}:${IMAGE_TAG}"

