"""
Run the full multi-spawn validation pipeline (tiers 1–3).

Usage:
  python -m data_utils.validate_dataset \\
      --data_dir data/processed/multi_spawn_v2/training_ssr \\
      --crash_type ssr \\
      --delete_invalid
"""
from __future__ import annotations

import argparse
import os

from data_utils.validate_heuristic import run as run_tier3
from data_utils.validate_reachability import run as run_tier2
from data_utils.validate_spawns import run as run_tier1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiers 1→3 dataset validation")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--crash_type", required=True, choices=["ssl", "ssr", "re"])
    parser.add_argument("--num_worlds", type=int, default=16)
    parser.add_argument("--probe_steps", type=int, default=30)
    parser.add_argument("--delete_invalid", action="store_true")
    parser.add_argument("--skip_heuristic", action="store_true", help="Skip GPU tier-3")
    parser.add_argument("--heuristic_limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    reject_root = os.path.join(os.path.dirname(args.data_dir), "rejects")
    os.makedirs(reject_root, exist_ok=True)
    tag = os.path.basename(args.data_dir)

    print("\n>>> Tier 1: drivability")
    run_tier1(
        data_dir=args.data_dir,
        crash_type=args.crash_type,
        num_worlds=args.num_worlds,
        probe_steps=args.probe_steps,
        delete_invalid=args.delete_invalid,
        reject_log=os.path.join(reject_root, f"{tag}_tier1.txt"),
        device=args.device,
    )

    print("\n>>> Tier 2: static reachability")
    run_tier2(
        data_dir=args.data_dir,
        crash_type=args.crash_type,
        delete_invalid=args.delete_invalid,
        reject_log=os.path.join(reject_root, f"{tag}_tier2.txt"),
    )

    if not args.skip_heuristic:
        print("\n>>> Tier 3: heuristic rollout")
        run_tier3(
            data_dir=args.data_dir,
            crash_type=args.crash_type,
            num_worlds=args.num_worlds,
            delete_invalid=args.delete_invalid,
            reject_log=os.path.join(reject_root, f"{tag}_tier3.txt"),
            device=args.device,
            limit=args.heuristic_limit,
        )


if __name__ == "__main__":
    main()
