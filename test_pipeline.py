"""Validate the notebook's data logic before spending GPU time on it.

Everything here runs on CPU in a couple of seconds. It parses the real notebook and executes
the actual functions defined in it, so these are tests of the code you are going to run, not of
a copy that can drift.

What it checks, in order of how badly each would corrupt the results if wrong:

  1. The train/val/test split cannot leak a fire across splits.
  2. Every temporal-stack reference frame is genuinely in the past and near the intended lag,
     including in sequences where the camera dropped frames.
  3. The operating-point threshold delivers the false-alarm rate it promises.
  4. The time-to-detection rule ignores pre-ignition frames, rejects single-frame spikes, and
     censors undetected fires instead of scoring them as instant detections.
  5. The emissions model scales as t^2 and ROS^2, and lands in the right order of magnitude.

Usage:
    python download_figlib.py --sequences 4     # a handful is enough
    python test_pipeline.py
"""

import ast
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).parent / "wildfire_smoke_detection.ipynb"
DATA = Path(__file__).parent / "data"

REF_LAG_S, REF_TOL_S, PERSISTENCE, TARGET_FPR, HORIZON = 240, 90, 2, 0.01, 40
FUEL_LOAD_T_HA, COMBUSTION_COMPLETE, EF_CO2_G_PER_KG = 30.0, 0.80, 1650.0

FRAME_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")

failures: list[str] = []


def ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def check(cond: bool, msg: str) -> None:
    if cond:
        ok(msg)
    else:
        failures.append(msg)
        print(f"  FAIL  {msg}")


def notebook_fn(needle: str) -> dict:
    """Exec the notebook cell defining `needle` and hand back its namespace.

    Only definitions are executed -- the surrounding statements in those cells consume live
    notebook state (dataframes, trained models), so the constants they need are injected here.
    """
    cells = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    for src in cells:
        if needle not in src:
            continue
        ns = dict(np=np, pd=pd, math=math, re=re,
                  REF_LAG_S=REF_LAG_S, REF_TOL_S=REF_TOL_S, PERSISTENCE=PERSISTENCE,
                  TARGET_FPR=TARGET_FPR, HORIZON=HORIZON, FUEL_LOAD_T_HA=FUEL_LOAD_T_HA,
                  COMBUSTION_COMPLETE=COMBUSTION_COMPLETE, EF_CO2_G_PER_KG=EF_CO2_G_PER_KG)
        tree = ast.parse(src)
        tree.body = [n for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
        exec(compile(tree, "<notebook>", "exec"), ns)
        return ns
    raise AssertionError(f"no notebook cell defines {needle!r}")


def build_index() -> pd.DataFrame:
    rows = []
    for seq_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for f in sorted(seq_dir.glob("*.jpg")):
            m = FRAME_RE.match(f.name)
            if not m:
                continue
            off = int(m.group(2))
            rows.append({"sequence": seq_dir.name, "fire": seq_dir.name[:8], "path": str(f),
                         "offset_s": off, "minute": int(round(off / 60)), "label": int(off >= 0)})
    return pd.DataFrame(rows).sort_values(["sequence", "offset_s"]).reset_index(drop=True)


def main() -> int:
    if not NOTEBOOK.exists():
        print(f"notebook not found: {NOTEBOOK}")
        return 1
    if not DATA.is_dir() or not any(DATA.iterdir()):
        print(f"no data in {DATA}. Run:  python download_figlib.py --sequences 4")
        return 1

    # -------------------------------------------------- 1. frame index
    frames = build_index()
    print(f"\n[1] frame index: {len(frames)} frames | {frames.sequence.nunique()} sequences | "
          f"{frames.fire.nunique()} fires | {frames.label.mean():.0%} post-ignition")
    check(len(frames) > 0, "frames found and filenames parsed")
    check(bool(frames.minute.between(-45, 45).all()), "all offsets fall in the +/-40 min window")
    check(0.35 < frames.label.mean() < 0.65, "classes are near-balanced by construction")

    # -------------------------------------------------- 2. reference matching
    attach_reference = notebook_fn("def attach_reference")["attach_reference"]
    paired = attach_reference(frames)
    print(f"\n[2] reference matching: {len(paired)}/{len(frames)} frames retained")
    check(bool((paired.ref_err_s <= REF_TOL_S).all()), f"every reference is within {REF_TOL_S}s of the target lag")

    off_by_path = frames.set_index("path")["offset_s"]
    lag = paired.offset_s.to_numpy() - off_by_path.loc[paired.ref_path].to_numpy()
    print(f"    realised lag: min {lag.min()}s  median {np.median(lag):.0f}s  max {lag.max()}s")
    check(bool(lag.min() > 0), "no frame is ever paired with a reference from its own future")
    check(abs(float(np.median(lag)) - REF_LAG_S) <= 5, f"median lag is ~{REF_LAG_S}s as intended")

    bad = []
    for seq, g in frames.groupby("sequence"):
        kept = set(paired[paired.sequence == seq].offset_s)
        dropped = sorted(set(g.offset_s) - kept)
        if dropped != sorted(g.offset_s)[:len(dropped)]:
            bad.append(seq)
    check(not bad, "the only dropped frames are each sequence's opening frames")

    # -------------------------------------------------- 3. split guard
    print("\n[3] split guard")
    fires = sorted(paired.fire.unique())
    np.random.default_rng(0).shuffle(fires)
    n = len(fires)
    n_test, n_val = max(1, round(0.25 * n)), max(1, round(0.20 * n))
    test_f, val_f = set(fires[:n_test]), set(fires[n_test:n_test + n_val])
    train_f = set(fires[n_test + n_val:])
    print(f"    {len(train_f)} train / {len(val_f)} val / {len(test_f)} test fires (of {n})")
    check(not (train_f & val_f) and not (train_f & test_f) and not (val_f & test_f),
          "the three fire sets are disjoint")

    seqs = {name: set(paired[paired.fire.isin(fs)].sequence) for name, fs in
            [("train", train_f), ("val", val_f), ("test", test_f)]}
    check(seqs["train"].isdisjoint(seqs["test"]),
          "no sequence -- hence no camera angle on one fire -- spans train and test")

    # -------------------------------------------------- 4. threshold and detection timing
    print("\n[4] operating point and time-to-detection")
    threshold_at_fpr = notebook_fn("def threshold_at_fpr")["threshold_at_fpr"]
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(1000), np.ones(1000)]
    s = np.r_[rng.normal(0.3, 0.1, 1000), rng.normal(0.7, 0.1, 1000)]
    tau = threshold_at_fpr(y, s, TARGET_FPR)
    fpr = float((s[y == 0] >= tau).mean())
    print(f"    tau={tau:.3f} -> realised FPR {fpr:.3%} (target {TARGET_FPR:.0%})")
    check(fpr <= TARGET_FPR + 1e-9, "threshold delivers no more than the requested false-alarm rate")

    ttd = notebook_fn("def time_to_detection")["time_to_detection"]
    demo = paired[paired.sequence == paired.sequence.iloc[0]].copy()
    minute = demo.minute.to_numpy()

    got = ttd(demo, np.where(minute >= 6, 0.9, 0.1), 0.5)
    check(list(got.values())[0] == 6 + PERSISTENCE - 1,
          f"persistence rule alerts at +{6 + PERSISTENCE - 1} when signal begins at +6")
    check(list(ttd(demo, np.where(minute == 3, 0.9, 0.1), 0.5).values())[0] is None,
          "an isolated single-frame spike does not raise an alert")
    check(list(ttd(demo, np.full(len(demo), 0.1), 0.5).values())[0] is None,
          "an undetected fire is censored, not scored as an instant detection")
    check(list(ttd(demo, np.where(minute < 0, 0.9, 0.1), 0.5).values())[0] is None,
          "pre-ignition frames cannot manufacture a detection time")

    # -------------------------------------------------- 5. emissions model
    print("\n[5] emissions model")
    co2 = notebook_fn("def co2_tonnes")["co2_tonnes"]
    check(abs(co2(60, 1) / co2(30, 1) - 4.0) < 1e-9, "area, so emissions, scale as t^2")
    check(abs(co2(30, 5) / co2(30, 1) - 25.0) < 1e-9, "emissions scale as ROS^2")
    per_ha = FUEL_LOAD_T_HA * COMBUSTION_COMPLETE * EF_CO2_G_PER_KG / 1000.0
    check(20 < per_ha < 80, f"per-hectare emissions ({per_ha:.0f} t CO2/ha) are the right order of magnitude")
    print("    sensitivity grid (t CO2):")
    for ros in (1, 5, 20):
        cells = "  ".join(f"{d:>2}min={co2(d, ros):>10,.1f}" for d in (10, 30, 60))
        print(f"      ROS {ros:>2} m/min:  {cells}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED -- the pipeline is safe to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
