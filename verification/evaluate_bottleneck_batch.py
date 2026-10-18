"""Evaluate new snapshots with the same start/hurdle/flat-running measurements."""
import argparse
import json
from pathlib import Path

from evaluate_gait_bottlenecks import evaluate_checkpoint, compact_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    args = parser.parse_args()
    verification = Path(__file__).resolve().parent
    repo = verification.parent / "QWOP_rl_training_public"
    for checkpoint in sorted(repo.glob(args.pattern)):
        output = verification / f"{checkpoint.stem}_bottlenecks.json"
        if output.exists():
            continue
        report = evaluate_checkpoint(checkpoint)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(compact_summary(report), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
