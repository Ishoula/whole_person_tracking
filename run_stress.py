"""
Crossing stress test in the simulator: honest three-state model vs a naive
'never let go' baseline (no decay, no overlap cap, no mismatch penalty).

    python run_stress.py --seeds 10

Detector noise/recall are pushed until bodies really get swapped, then we
count frames where the followed body is the WRONG person, and how many of
those the system still labelled CONFIRMED.
"""
import argparse
import csv
import os
import tempfile

from src.identity_state import IdentityConfig
from src.sim import run_scenario
from src.whole_person import PipelineConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", default="reports/stress.md")
args = ap.parse_args()

naive = PipelineConfig(identity=IdentityConfig(tau=1e9, ambiguity_cap=1.0, mismatch_factor=1.0))
settings = [(0.92, 3.0), (0.70, 12.0), (0.50, 20.0)]
lines = ["| scenario | body recall | det. noise (px) | model | wrong-person frames | ...of which CONFIRMED |",
         "|---|---|---|---|---|---|"]
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "x.csv")
    for name in ("crossing_swap", "crossing_bounce"):
        for rec, noise in settings:
            for label, cfg in (("three-state", None), ("never-let-go", naive)):
                wrong = wrong_conf = 0
                for seed in range(args.seeds):
                    run_scenario(name, path, seed=seed, pipeline_cfg=cfg, body_recall=rec, det_noise=noise)
                    with open(path) as f:
                        for r in csv.DictReader(f):
                            if r["wrong"] == "1":
                                wrong += 1
                                wrong_conf += r["state"] == "CONFIRMED"
                lines.append(f"| {name} | {rec} | {noise} | {label} | {wrong} | {wrong_conf} |")
                print(lines[-1], flush=True)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(f"Totals over {args.seeds} seeds per row (30 fps, 12 s each).\n\n" + "\n".join(lines) + "\n")
print("wrote", args.out)
