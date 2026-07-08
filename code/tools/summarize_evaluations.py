import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = [
    "average_reward",
    "success_rate",
    "collision_rate",
    "unsafe_distance_rate",
    "average_speed",
    "average_episode_length",
    "hard_brake_count_per_100_episodes",
    "lane_change_count_per_episode",
]


def metric(report: dict, name: str):
    metrics = report.get("metrics", {})
    if name in metrics:
        return metrics[name]
    return report.get(name)


def load_reports(input_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(input_dir.rglob("evaluation_report.json")):
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        report["_source_path"] = str(path)
        reports.append(report)
    return reports


def flatten_report(report: dict) -> dict:
    row = {
        "experiment_name": report.get("experiment_name", ""),
        "evaluation_suite": report.get("evaluation_suite", ""),
        "episodes": report.get("episodes", ""),
        "decision": report.get("decision", "unknown"),
        "failure_reasons": "; ".join(report.get("failure_reasons", [])),
        "source_path": report.get("_source_path", ""),
    }
    for field in METRIC_FIELDS:
        row[field] = metric(report, field)
    return row


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_name",
        "evaluation_suite",
        "episodes",
        *METRIC_FIELDS,
        "decision",
        "failure_reasons",
        "source_path",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_name",
        "average_reward",
        "success_rate",
        "collision_rate",
        "unsafe_distance_rate",
        "average_speed",
        "decision",
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Evaluation Summary\n\n")
        if not rows:
            handle.write("No evaluation reports found.\n")
            return
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs")
    parser.add_argument("--csv-output", default="outputs/experiments_summary.csv")
    parser.add_argument("--md-output", default="outputs/experiments_summary.md")
    args = parser.parse_args()

    reports = load_reports(Path(args.input_dir))
    rows = [flatten_report(report) for report in reports]
    write_csv(rows, Path(args.csv_output))
    write_markdown(rows, Path(args.md_output))

    print(f"Loaded reports: {len(rows)}")
    print(f"Wrote CSV: {args.csv_output}")
    print(f"Wrote Markdown: {args.md_output}")


if __name__ == "__main__":
    main()

