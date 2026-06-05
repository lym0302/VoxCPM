#!/usr/bin/env python3
"""
Select the best or latest checkpoint from a VoxCPM training run.

Examples:
    # List all checkpoints with validation metrics
    python scripts/select_checkpoint.py --save_path exps/checkpoints/finetune_all --list

    # Print path of the best checkpoint (lowest val loss/total)
    python scripts/select_checkpoint.py --save_path exps/checkpoints/finetune_all

    # Print path of the latest checkpoint
    python scripts/select_checkpoint.py --save_path exps/checkpoints/finetune_all --mode latest

    # Use a different metric
    python scripts/select_checkpoint.py --save_path exps/checkpoints/finetune_all --metric loss/diff

    # Copy the selected checkpoint to a destination folder
    python scripts/select_checkpoint.py --save_path exps/checkpoints/finetune_all --copy_to exps/checkpoints/best
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def parse_val_log(log_file: Path) -> dict[int, dict[str, float]]:
    """
    Parse train.log and return {step: {metric_name: value}}.
    Log line format: [val] step 500: loss/total: 0.123456, loss/diff: 0.111111, ...
    """
    line_pattern = re.compile(r"\[val\] step (\d+): (.+?)(?:, log interval.*)?$")
    kv_pattern = re.compile(r"([\w/]+): ([+-]?[\d.]+(?:e[+-]?\d+)?)")

    step_metrics: dict[int, dict[str, float]] = {}
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            m = line_pattern.match(line.strip())
            if not m:
                continue
            step = int(m.group(1))
            metrics = {k: float(v) for k, v in kv_pattern.findall(m.group(2))}
            if metrics:
                step_metrics[step] = metrics
    return step_metrics


def find_checkpoints(save_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted list of (step, path) for all step_XXXXXXX folders."""
    result = []
    for d in save_dir.iterdir():
        if d.is_dir() and re.fullmatch(r"step_\d+", d.name):
            state_file = d / "training_state.json"
            if state_file.exists():
                try:
                    with open(state_file, encoding="utf-8") as f:
                        step = int(json.load(f).get("step", d.name.split("_")[1]))
                except Exception:
                    step = int(d.name.split("_")[1])
            else:
                step = int(d.name.split("_")[1])
            result.append((step, d))
    return sorted(result)


def select_best(
    checkpoints: list[tuple[int, Path]],
    step_metrics: dict[int, dict[str, float]],
    metric: str,
) -> tuple[int, Path] | None:
    """Return (step, path) with the lowest value for metric, or None if unavailable."""
    best_step, best_val, best_path = None, float("inf"), None
    for step, path in checkpoints:
        val = step_metrics.get(step, {}).get(metric)
        if val is not None and val < best_val:
            best_val, best_step, best_path = val, step, path
    if best_step is None:
        return None
    return best_step, best_path


def copy_checkpoint(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Select best or latest VoxCPM checkpoint")
    parser.add_argument("--save_path", required=True, help="Checkpoint save directory")
    parser.add_argument(
        "--mode",
        choices=["best", "latest"],
        default="best",
        help="Selection mode (default: best)",
    )
    parser.add_argument(
        "--metric",
        default="loss/total",
        help="Val metric to minimize in best mode (default: loss/total)",
    )
    parser.add_argument("--list", action="store_true", help="List all checkpoints and exit")
    parser.add_argument(
        "--copy_to",
        default="",
        help="If set, copy the selected checkpoint to this directory",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_path)
    if not save_dir.exists():
        print(f"Error: {save_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    checkpoints = find_checkpoints(save_dir)
    if not checkpoints:
        print(f"Error: no step_XXXXXXX checkpoints found in {save_dir}", file=sys.stderr)
        sys.exit(1)

    log_file = save_dir / "train.log"
    step_metrics: dict[int, dict[str, float]] = {}
    if log_file.exists():
        step_metrics = parse_val_log(log_file)
    else:
        print(f"Warning: train.log not found in {save_dir}", file=sys.stderr)

    if args.list:
        latest_step = checkpoints[-1][0]
        col_w = max(len(str(p)) for _, p in checkpoints)
        print(f"{'Step':>10}  {'Path':<{col_w}}  Metrics")
        print("-" * (14 + col_w + 40))
        for step, path in checkpoints:
            if step in step_metrics:
                metrics_str = "  ".join(
                    f"{k}: {v:.6f}" for k, v in sorted(step_metrics[step].items())
                )
            else:
                metrics_str = "(no val metrics)"
            tag = " <- latest" if step == latest_step else ""
            print(f"{step:>10}  {str(path):<{col_w}}  {metrics_str}{tag}")
        return

    # --- select ---
    if args.mode == "latest":
        selected_step, selected_path = checkpoints[-1]
        print(f"[select] latest  step={selected_step}  path={selected_path}", file=sys.stderr)
    else:
        if not step_metrics:
            print(
                f"Warning: no val metrics in train.log, falling back to latest checkpoint",
                file=sys.stderr,
            )
            selected_step, selected_path = checkpoints[-1]
        else:
            result = select_best(checkpoints, step_metrics, args.metric)
            if result is None:
                print(
                    f"Warning: metric '{args.metric}' not found in any checkpoint, "
                    f"falling back to latest",
                    file=sys.stderr,
                )
                selected_step, selected_path = checkpoints[-1]
            else:
                selected_step, selected_path = result
                best_val = step_metrics[selected_step][args.metric]
                print(
                    f"[select] best  step={selected_step}  {args.metric}={best_val:.6f}"
                    f"  path={selected_path}",
                    file=sys.stderr,
                )

    if args.copy_to:
        dst = Path(args.copy_to)
        copy_checkpoint(selected_path, dst)
        print(f"[select] copied to {dst}", file=sys.stderr)
        print(str(dst))
    else:
        print(str(selected_path))


if __name__ == "__main__":
    main()
