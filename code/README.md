# Autodriving RL Minimal AWS Training Flow

This project builds a minimal AWS training flow for an autonomous-driving-inspired RL policy.

The first version uses:

- `highway-env` as the simulator
- `stable-baselines3` PPO as the algorithm
- Amazon S3 for configs, model artifacts, and reports
- Amazon ECR for the training/evaluation image
- SageMaker Training for training
- SageMaker Processing for offline evaluation
- SageMaker Model Registry for candidate model registration

## Structure

```text
autodriving-rl/
  docs/
    stage-*.md
  code/
    configs/
      highway-ppo-conservative.yaml
      highway-ppo-speedy.yaml
      highway-ppo-v001.yaml
    infra/
      cloudformation/foundation.yaml
      scripts/*.sh
    training/
      Dockerfile
      requirements.txt
      train.py
      evaluate.py
    tools/
      summarize_evaluations.py
```

## Prerequisites

- AWS CLI configured with credentials
- Docker running locally
- Permission to create CloudFormation stacks, S3 buckets, ECR repositories, IAM roles, and SageMaker jobs

Run commands from the code directory:

```bash
cd ML/autodriving-rl/code
```

Set your region:

```bash
export AWS_REGION=ap-northeast-1
export AWS_DEFAULT_REGION=ap-northeast-1
```

Optional custom names:

```bash
export STACK_NAME=autodriving-rl-foundation
export ARTIFACT_BUCKET_NAME=your-globally-unique-bucket-name
export EXPERIMENT_NAME=highway-ppo-v001
```

## 1. Deploy Foundation IaC

```bash
./infra/scripts/deploy_foundation.sh
```

This creates:

- S3 artifact bucket
- ECR training repository
- SageMaker execution role

## 2. Upload Config

```bash
./infra/scripts/upload_config.sh
```

## 3. Build and Push Training Image

```bash
./infra/scripts/build_push_ecr.sh
```

## 4. Start Training

```bash
./infra/scripts/start_training_job.sh
```

The script prints a generated training job name. Save it:

```bash
export TRAINING_JOB_NAME=highway-ppo-v001-YYYYMMDD-HHMMSS
```

Wait for completion:

```bash
./infra/scripts/wait_training_job.sh "$TRAINING_JOB_NAME"
```

## 5. Run Evaluation

```bash
./infra/scripts/start_evaluation_processing_job.sh "$TRAINING_JOB_NAME"
```

Wait for completion:

```bash
./infra/scripts/wait_processing_job.sh "eval-${TRAINING_JOB_NAME}"
```

Download report:

```bash
./infra/scripts/download_evaluation_report.sh "$TRAINING_JOB_NAME"
```

## 6. Register Candidate Model

```bash
./infra/scripts/register_model_package.sh "$TRAINING_JOB_NAME"
```

The model package is registered as `PendingManualApproval`.

## Reward Comparison Experiments

This capstone includes three configs:

| Config | Intent |
| --- | --- |
| `configs/highway-ppo-v001.yaml` | Baseline |
| `configs/highway-ppo-conservative.yaml` | More conservative reward |
| `configs/highway-ppo-speedy.yaml` | Stronger speed-seeking reward |

Run each experiment by setting `EXPERIMENT_NAME` and `CONFIG_PATH` before upload/training:

```bash
export EXPERIMENT_NAME=highway-ppo-conservative
export CONFIG_PATH=ML/autodriving-rl/code/configs/highway-ppo-conservative.yaml
./infra/scripts/upload_config.sh
./infra/scripts/start_training_job.sh
```

## Summarize Evaluation Reports

After downloading reports into `outputs/`, summarize them:

```bash
python tools/summarize_evaluations.py \
  --input-dir outputs \
  --csv-output outputs/experiments_summary.csv \
  --md-output outputs/experiments_summary.md
```

## Minimal Flow

```text
config.yaml
  -> S3
Dockerfile + train.py + evaluate.py
  -> ECR
SageMaker Training
  -> S3 model.tar.gz
SageMaker Processing
  -> S3 evaluation_report.json
Model Registry
  -> candidate model package
```

## Notes

This is a learning scaffold. It trains a simplified highway decision policy, not a deployable autonomous driving system.
