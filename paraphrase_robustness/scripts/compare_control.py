#!/usr/bin/env python3
"""Compare spread summary (turn0 vs turn3) for main vs control runs side by side."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from paraphrase_robustness.analyze_spread import summarize_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runs-main", type=Path, required=True)
    ap.add_argument("--runs-control", type=Path, required=True)
    ap.add_argument("--run-key-main", default=None)
    ap.add_argument("--run-key-control", default=None)
    ap.add_argument("--iteration", default="1")
    args = ap.parse_args()

    main_s = summarize_run(
        args.runs_main,
        args.manifest,
        run_key=args.run_key_main,
        iteration=args.iteration,
        kind_filter="main",
    )
    ctrl_s = summarize_run(
        args.runs_control,
        args.manifest,
        run_key=args.run_key_control,
        iteration=args.iteration,
        kind_filter="control",
    )

    out = {
        "main_mean_delta_std": main_s.get("mean_delta_std"),
        "control_mean_delta_std": ctrl_s.get("mean_delta_std"),
        "main_n_bases": main_s.get("n_bases"),
        "control_n_bases": ctrl_s.get("n_bases"),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
