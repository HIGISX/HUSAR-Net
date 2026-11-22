"""Command-line entry points for HUSAR-Net."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from . import load_instance, load_policy, make_synthetic, save_instance, save_policy, solve, train_policy


def _output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hierarchical uncertainty-aware spatial allocation")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="solve a small synthetic instance")
    demo.add_argument("--seed", type=int, default=2026)
    demo.add_argument("--swap-seconds", type=float, default=1.0)

    example = commands.add_parser("example", help="write an example instance in NPZ format")
    example.add_argument("--seed", type=int, default=2026)
    example.add_argument("--output", type=Path, required=True)

    allocate = commands.add_parser("solve", help="solve an instance in NPZ format")
    allocate.add_argument("instance", type=Path)
    allocate.add_argument("--checkpoint", type=Path, help="use a custom riskmean policy")
    allocate.add_argument("--output", type=Path, help="write the allocation JSON to a file")
    allocate.add_argument("--swap-seconds", type=float, default=5.0)
    allocate.add_argument("--seed", type=int, default=5100)

    train = commands.add_parser("train", help="train a policy from NPZ instances")
    train.add_argument("--training", type=Path, nargs="+", required=True)
    train.add_argument("--validation", type=Path, nargs="+", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--imitation-epochs", type=int, default=12)
    train.add_argument("--ppo-rounds", type=int, default=16)
    train.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.command in {"demo", "example"}:
            instance = make_synthetic(seed=args.seed, N=120, G=12, P=4, B=24, K=12)
            if args.command == "example":
                save_instance(instance, _output_path(args.output))
                return 0
            result = solve(instance, swap_seconds=args.swap_seconds, seed=args.seed)
            output = None
        elif args.command == "train":
            policy = train_policy(
                (load_instance(path) for path in args.training),
                (load_instance(path) for path in args.validation),
                imitation_epochs=args.imitation_epochs, ppo_rounds=args.ppo_rounds,
                seed=args.seed, verbose=True,
            )
            save_policy(policy, _output_path(args.output))
            return 0
        else:
            instance = load_instance(args.instance)
            policy = load_policy(args.checkpoint) if args.checkpoint is not None else None
            result = solve(instance, policy, swap_seconds=args.swap_seconds, seed=args.seed)
            output = args.output
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))

    payload = json.dumps(asdict(result), indent=2, allow_nan=False)
    if output is None:
        print(payload)
    else:
        _output_path(output).write_text(payload + "\n", encoding="utf-8")
    return 0
