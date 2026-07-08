import argparse
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


class CheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_dir: Path, checkpoint_interval: int):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

    def _on_step(self) -> bool:
        if self.checkpoint_interval <= 0:
            return True
        if self.num_timesteps % self.checkpoint_interval == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_dir / f"model_step_{self.num_timesteps}.zip")
        return True


def resolve_config_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists() and candidate.is_file():
        return candidate
    search_root = candidate if candidate.is_dir() else candidate.parent
    matches = sorted(list(search_root.glob("*.yaml")) + list(search_root.glob("*.yml")))
    if not matches:
        raise FileNotFoundError(f"Could not find a YAML config at or under {search_root}")
    return matches[0]


def load_config(path: str) -> dict:
    resolved_path = resolve_config_path(path)
    print(f"Loading config from {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(config: dict, seed: int):
    env_id = config["environment"]["id"]
    env_config = config["environment"].get("config", {})
    render_mode = config["environment"].get("render_mode")
    env = gym.make(env_id, config=env_config, render_mode=render_mode)
    env.reset(seed=seed)
    return Monitor(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("TRAINING_CONFIG", "/opt/ml/input/data/config/config.yaml"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    args = parser.parse_args()

    config = load_config(args.config)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)

    env = make_env(config, seed)
    training_config = config["training"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(training_config.get("learning_rate", 0.0003)),
        n_steps=int(training_config.get("n_steps", 1024)),
        batch_size=int(training_config.get("batch_size", 64)),
        gamma=float(training_config.get("gamma", 0.99)),
        seed=seed,
        verbose=1,
    )

    callback = CheckpointCallback(
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=int(training_config.get("checkpoint_interval", 0)),
    )
    total_timesteps = int(training_config["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_model_path = model_dir / "model.zip"
    model.save(final_model_path)

    training_summary = {
        "experiment_name": config["experiment_name"],
        "algorithm": training_config.get("algorithm", "PPO"),
        "total_timesteps": total_timesteps,
        "seed": seed,
        "model_path": str(final_model_path),
    }
    with open(model_dir / "training_summary.json", "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)
    with open(model_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    env.close()


if __name__ == "__main__":
    main()
