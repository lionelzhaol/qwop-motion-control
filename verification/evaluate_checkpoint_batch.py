"""Evaluate saved PPO snapshots without restarting PyTorch for each snapshot."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import evaluate_classic_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parent.parent
    repo = workspace / "QWOP_rl_training_public"
    for checkpoint in sorted(repo.glob(args.pattern)):
        update = int(checkpoint.stem.rsplit("_", 1)[1])
        output = workspace / "verification" / f"{args.prefix}{update}.json"
        if output.exists():
            continue
        sys.argv = ["evaluate_classic_policy.py", "--checkpoint", str(checkpoint),
                    "--episodes", "1", "--threads", "1", "--deterministic",
                    "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate_classic_policy.main()
        report = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps({"update": update, "time": report["success_time_min"],
                          "success": report["successes"],
                          "cadence": report["alternating_foot_landings_per_second"],
                          "bob": report["torso_height_std_after_5_seconds"],
                          "lean": report["lean_abs_error_from_5_degrees"]}), flush=True)


if __name__ == "__main__":
    main()
