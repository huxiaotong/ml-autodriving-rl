import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import yaml
from stable_baselines3 import PPO


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def extract_model_artifact(model_artifact: Path) -> Path:
    if model_artifact.is_dir():
        tarballs = sorted(model_artifact.rglob("model.tar.gz"))
        if tarballs:
            return extract_model_artifact(tarballs[0])
        return model_artifact
    if not model_artifact.exists():
        search_root = model_artifact.parent
        tarballs = sorted(search_root.rglob("model.tar.gz"))
        if not tarballs:
            raise FileNotFoundError(f"Could not find model.tar.gz at or under {search_root}")
        model_artifact = tarballs[0]
    extract_dir = Path(tempfile.mkdtemp(prefix="autodriving-rl-model-"))
    with tarfile.open(model_artifact, "r:gz") as tar:
        tar.extractall(extract_dir)
    return extract_dir


def make_env(config: dict, seed: int):
    env_id = config["environment"]["id"]
    env_config = config["environment"].get("config", {})
    render_mode = config["environment"].get("render_mode")
    env = gym.make(env_id, config=env_config, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def get_vehicle_snapshot(env) -> dict:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return {
            "speed": 0.0,
            "lane_index": None,
            "crashed": False,
        }
    return {
        "speed": as_float(getattr(vehicle, "speed", 0.0)),
        "lane_index": getattr(vehicle, "lane_index", None),
        "crashed": bool(getattr(vehicle, "crashed", False)),
    }


def get_front_distance(env) -> float | None:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    road = getattr(env.unwrapped, "road", None)
    if vehicle is None or road is None or not hasattr(road, "neighbour_vehicles"):
        return None

    try:
        front_vehicle, _ = road.neighbour_vehicles(vehicle)
    except Exception:
        return None

    if front_vehicle is None:
        return None

    try:
        distance = vehicle.lane_distance_to(front_vehicle)
    except Exception:
        return None

    return max(0.0, as_float(distance))


def evaluate_gates(metrics: dict, gates: dict) -> tuple[str, list[str], dict]:
    if not gates:
        return "needs_review", ["No evaluation gates configured"], {}

    gate_results = {}
    failure_reasons = []

    checks = {
        "min_success_rate": (metrics.get("success_rate"), ">="),
        "max_collision_rate": (metrics.get("collision_rate"), "<="),
        "max_unsafe_distance_rate": (metrics.get("unsafe_distance_rate"), "<="),
        "max_hard_brake_count_per_100_episodes": (
            metrics.get("hard_brake_count_per_100_episodes"),
            "<=",
        ),
    }

    for gate_name, (actual, operator) in checks.items():
        if gate_name not in gates:
            continue
        threshold = as_float(gates[gate_name])
        if actual is None:
            passed = False
        elif operator == ">=":
            passed = actual >= threshold
        else:
            passed = actual <= threshold
        gate_results[gate_name] = {
            "actual": actual,
            "operator": operator,
            "threshold": threshold,
            "status": "pass" if passed else "fail",
        }
        if not passed:
            failure_reasons.append(
                f"{gate_name} failed: actual={actual}, expected {operator} {threshold}"
            )

    if failure_reasons:
        return "fail", failure_reasons, gate_results
    if not gate_results:
        return "needs_review", ["No recognized gates configured"], gate_results
    return "pass", [], gate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("EVAL_CONFIG", "/opt/ml/processing/input/config"))
    parser.add_argument("--model-artifact", default=os.environ.get("MODEL_ARTIFACT", "/opt/ml/processing/input/model"))
    parser.add_argument("--output-dir", default=os.environ.get("EVAL_OUTPUT_DIR", "/opt/ml/processing/output"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = extract_model_artifact(Path(args.model_artifact))
    model_path = model_dir / "model.zip"
    if not model_path.exists():
        candidates = list(model_dir.rglob("model.zip"))
        if not candidates:
            raise FileNotFoundError(f"Could not find model.zip under {model_dir}")
        model_path = candidates[0]

    eval_config = config.get("evaluation", {})
    episodes = int(eval_config.get("episodes", 20))
    deterministic = bool(eval_config.get("deterministic", True))
    unsafe_distance_threshold = as_float(eval_config.get("unsafe_distance_threshold", 8.0))
    hard_brake_delta_threshold = as_float(eval_config.get("hard_brake_delta_threshold", 4.0))
    seed = int(config.get("seed", 42)) + 1000

    env = make_env(config, seed)
    model = PPO.load(model_path, env=env)

    episode_rewards = []
    episode_lengths = []
    episode_speeds = []
    episode_reports = []
    collisions = 0
    successes = 0
    unsafe_distance_events = 0
    hard_brake_events = 0
    lane_change_events = 0
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        length = 0
        info = {}
        speeds = []
        episode_unsafe_distance_events = 0
        episode_hard_brake_events = 0
        episode_lane_change_events = 0
        last_snapshot = get_vehicle_snapshot(env)

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            snapshot = get_vehicle_snapshot(env)
            front_distance = get_front_distance(env)

            if front_distance is not None and front_distance < unsafe_distance_threshold:
                unsafe_distance_events += 1
                episode_unsafe_distance_events += 1

            speed_delta = snapshot["speed"] - last_snapshot["speed"]
            if speed_delta < -hard_brake_delta_threshold:
                hard_brake_events += 1
                episode_hard_brake_events += 1

            if (
                snapshot["lane_index"] is not None
                and last_snapshot["lane_index"] is not None
                and snapshot["lane_index"] != last_snapshot["lane_index"]
            ):
                lane_change_events += 1
                episode_lane_change_events += 1

            speeds.append(snapshot["speed"])
            total_reward += float(reward)
            length += 1
            total_steps += 1
            last_snapshot = snapshot
            done = terminated or truncated

        crashed = bool(getattr(env.unwrapped.vehicle, "crashed", False))
        collisions += int(crashed)
        successes += int(not crashed)
        episode_rewards.append(total_reward)
        episode_lengths.append(length)
        episode_speeds.extend(speeds)
        episode_reports.append(
            {
                "episode": episode,
                "reward": total_reward,
                "length": length,
                "crashed": crashed,
                "success": not crashed,
                "average_speed": float(np.mean(speeds)) if speeds else 0.0,
                "unsafe_distance_events": episode_unsafe_distance_events,
                "hard_brake_events": episode_hard_brake_events,
                "lane_change_events": episode_lane_change_events,
            }
        )

    metrics = {
        "average_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "average_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "average_speed": float(np.mean(episode_speeds)) if episode_speeds else 0.0,
        "collision_rate": collisions / episodes if episodes else 0.0,
        "success_rate": successes / episodes if episodes else 0.0,
        "unsafe_distance_rate": unsafe_distance_events / total_steps if total_steps else 0.0,
        "hard_brake_count": hard_brake_events,
        "hard_brake_count_per_100_episodes": (hard_brake_events / episodes * 100) if episodes else 0.0,
        "lane_change_count": lane_change_events,
        "lane_change_count_per_episode": lane_change_events / episodes if episodes else 0.0,
    }
    decision, failure_reasons, gate_results = evaluate_gates(metrics, eval_config.get("gates", {}))

    report = {
        "experiment_name": config["experiment_name"],
        "evaluation_suite": eval_config.get("suite_name", "default"),
        "episodes": episodes,
        "model_path": str(model_path),
        "metrics": metrics,
        "gates": gate_results,
        "decision": decision,
        "failure_reasons": failure_reasons,
        "per_episode_metrics": episode_reports,
    }

    with open(output_dir / "evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
