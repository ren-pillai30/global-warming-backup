"""Download a subset of the HPWREN FIgLib wildfire-smoke archive.

FIgLib gives us something almost no other fire dataset does: every filename
encodes the offset in seconds from the moment the smoke plume first became
visible.  That free ground-truth clock is what lets us measure *how early* a
detector fires, instead of just how accurate it is.

Archive layout (verified 2026-09):
    https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/<sequence>.tgz
    523 sequences, ~87 MB each, 81 JPEGs per sequence (2048x1536),
    one frame per minute: 40 before ignition, the ignition frame, 40 after.
    Frame names look like  1465067280_+01680.jpg  ->  +1680 s  =  +28 min.

Run once; it skips anything already on disk.

    python download_figlib.py --sequences 30 --out data
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

INDEX_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html"
BASE_URL = "https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/"
UA = "Mozilla/5.0 (research; wildfire-smoke-detection coursework)"

# Frames are named <unix_epoch>_<signed offset in seconds>.jpg
FRAME_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")


def fire_date(sequence: str) -> str | None:
    """The YYYYMMDD prefix of a sequence name, used as the train/val/test group.

    Why group by date rather than by sequence: the same fire is routinely filmed
    by several cameras at once (the archive's 523 sequences cover only 338
    distinct fires, and one fire appears on 9 cameras).  Splitting on sequence
    name would put near-identical views of the *same* fire into both train and
    test and inflate every metric.  Grouping by date is one line and can never
    leak; at worst it conservatively merges two unrelated fires that happened on
    the same day, which costs a little diversity and nothing else.
    """
    m = re.match(r"^(\d{8})", sequence)
    return m.group(1) if m else None


def fetch_index() -> list[str]:
    req = urllib.request.Request(INDEX_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", "replace")
    return sorted({m.group(1) for m in re.finditer(r'href=["\']?([^"\'> ]+)\.tgz', html)})


def choose(sequences: list[str], target: int, per_date: int, seed: int) -> list[str]:
    """Pick ~`target` sequences, at most `per_date` per fire, deterministically.

    Capping per date buys fire diversity, and at this scale that is the binding
    constraint: for a 30-sequence budget, per_date=1 yields 30 distinct fires
    against 23 for per_date=2.  More distinct fires means more held-out fires,
    which is what the median time-to-detection is averaged over -- so the cap
    tightens the headline number as well as improving training coverage.
    """
    by_date: dict[str, list[str]] = defaultdict(list)
    for s in sequences:
        d = fire_date(s)
        if d:
            by_date[d].append(s)

    dates = sorted(by_date)
    random.Random(seed).shuffle(dates)

    picked: list[str] = []
    for d in dates:
        picked.extend(sorted(by_date[d])[:per_date])
        if len(picked) >= target:
            break
    return picked[:target]


def download_one(sequence: str, out: Path) -> tuple[str, int, str]:
    """Fetch + extract one sequence. Returns (name, n_frames, status)."""
    dest = out / sequence
    if dest.is_dir():
        n = len(list(dest.glob("*.jpg")))
        if n >= 70:  # a few sequences are legitimately short of the nominal 81
            return sequence, n, "cached"

    tgz = out / f"{sequence}.tgz"
    req = urllib.request.Request(BASE_URL + sequence + ".tgz", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, open(tgz, "wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)

        # Two packing layouts exist in the archive: most tarballs hold a bare
        # <sequence>/ directory, but some nest it under
        # Data/HPWREN-FIgLib/HPWREN-FIgLib-Data/<sequence>/.  Rather than special
        # -case both, pull the frames out flat by basename -- which also means we
        # never honour a path from the archive, so traversal is impossible.
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                base = Path(member.name).name
                if not FRAME_RE.match(base):
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(dest / base, "wb") as out_fh:
                    shutil.copyfileobj(src, out_fh)
    except Exception as exc:  # noqa: BLE001 - report and continue with other sequences
        return sequence, 0, f"FAILED: {type(exc).__name__}: {exc}"
    finally:
        tgz.unlink(missing_ok=True)

    n = len(list(dest.glob("*.jpg")))
    return sequence, n, "ok" if n else "FAILED: no frames matched the expected name format"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequences", type=int, default=30, help="how many sequences to fetch (~87 MB each)")
    ap.add_argument("--per-date", type=int, default=1,
                    help="max cameras per fire. 1 maximises the number of distinct fires, which "
                         "matters more than extra viewpoints at this dataset size")
    ap.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("fetching archive index ...", flush=True)
    everything = fetch_index()
    picked = choose(everything, args.sequences, args.per_date, args.seed)
    dates = {fire_date(s) for s in picked}
    print(f"{len(everything)} sequences available; fetching {len(picked)} "
          f"from {len(dates)} distinct fires (~{len(picked) * 87 / 1024:.1f} GB)\n", flush=True)

    ok, failed, total_frames = 0, [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, s, args.out): s for s in picked}
        for i, fut in enumerate(as_completed(futures), 1):
            name, n, status = fut.result()
            print(f"[{i:3d}/{len(picked)}] {name:<45s} {n:3d} frames  {status}", flush=True)
            if status in ("ok", "cached"):
                ok += 1
                total_frames += n
            else:
                failed.append(name)

    print(f"\n{ok}/{len(picked)} sequences, {total_frames} frames, {len(dates)} distinct fires")
    if failed:
        print(f"failed ({len(failed)}): {', '.join(failed)}\n  re-run to retry -- completed work is cached")
    return 1 if ok == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
