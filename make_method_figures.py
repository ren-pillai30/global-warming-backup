"""Generate the method figures that need no GPU and no trained model.

These three figures carry the argument of the project on their own, so they are worth having
before any training has run (for a progress review, say). The notebook reproduces the first two
inline during a full run; this script is the way to get them from a laptop with only the data.

    python download_figlib.py --sequences 3
    python make_method_figures.py

Writes figures/fig1_sequence_over_time.png, fig2_temporal_stack.png, fig3_why_224_fails.png
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA = Path(__file__).parent / "data"
FIGDIR = Path(__file__).parent / "figures"
FRAME_RE = re.compile(r"^(\d+)_([+-]\d+)\.jpg$")
REF_LAG_S = 240

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": False,
})


def index_sequence(seq_dir: Path) -> list[tuple[int, Path]]:
    """[(offset_seconds, path)] sorted in time."""
    out = []
    for f in seq_dir.glob("*.jpg"):
        m = FRAME_RE.match(f.name)
        if m:
            out.append((int(m.group(2)), f))
    return sorted(out)


def load_gray(path: Path, size: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def frame_at(frames, target_s):
    """The frame whose offset is nearest `target_s`."""
    return min(frames, key=lambda t: abs(t[0] - target_s))


def pick_sequence() -> tuple[str, list]:
    """Use the sequence with the strongest post-ignition change -- the clearest plume to show."""
    best, best_score = None, -1.0
    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        frames = index_sequence(d)
        if len(frames) < 20:
            continue
        cur = load_gray(frame_at(frames, 900)[1], 256)     # +15 min
        ref = load_gray(frame_at(frames, 900 - REF_LAG_S)[1], 256)
        score = float(np.abs(cur - ref).mean())
        if score > best_score:
            best, best_score = (d.name, frames), score
    if best is None:
        sys.exit(f"no usable sequences in {DATA}. Run: python download_figlib.py --sequences 3")
    return best


def fig1_sequence_over_time(name, frames):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, target_min in zip(axes, [-10, 0, 5, 20]):
        off, path = frame_at(frames, target_min * 60)
        ax.imshow(load_gray(path), cmap="gray", vmin=0, vmax=1)
        mins = round(off / 60)
        ax.set_title(f"t = {mins:+d} min" + ("   (pre-ignition)" if mins < 0 else ""), fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One fire over time -- {name}", y=1.02, fontsize=10)
    fig.savefig(FIGDIR / "fig1_sequence_over_time.png")
    plt.close(fig)


def fig2_temporal_stack(name, frames):
    off, path = frame_at(frames, 4 * 60)
    ref_off, ref_path = frame_at(frames, off - REF_LAG_S)
    cur, ref = load_gray(path, 448), load_gray(ref_path, 448)
    diff = np.abs(cur - ref)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    panels = [
        (cur,  f"channel 0:  gray(t),  t = {round(off/60):+d} min", dict(cmap="gray", vmin=0, vmax=1)),
        (ref,  f"channel 1:  gray(t - {round((off-ref_off)/60)} min)", dict(cmap="gray", vmin=0, vmax=1)),
        (diff, "channel 2:  |difference|", dict(cmap="magma")),
        (diff > 0.06, "channel 2, thresholded (illustration only)", dict(cmap="gray")),
    ]
    for ax, (im, title, kw) in zip(axes, panels):
        ax.imshow(im, **kw)
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    fig.suptitle("The single-frame baseline sees only the first panel. The temporal stack sees all three.",
                 y=1.03, fontsize=10)
    fig.savefig(FIGDIR / "fig2_temporal_stack.png")
    plt.close(fig)


def fig3_why_224_fails(name, frames):
    off, path = frame_at(frames, 6 * 60)
    src = Image.open(path).convert("L")
    w, h = src.size

    # Same frame, three input resolutions, all displayed at the same physical size, so the only
    # thing that varies is how much of the plume survives the resize a CNN would apply.
    # No cropping and no plume localisation: the comparison is the whole frame each time.
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, size in zip(axes, [None, 448, 224]):
        if size is None:
            shown = src.resize((896, 896), Image.BILINEAR)
            title = f"native frame ({w}x{h})"
        else:
            shown = src.resize((size, size), Image.BILINEAR).resize((896, 896), Image.NEAREST)
            title = f"resized to {size}x{size}, then magnified back"
        ax.imshow(np.asarray(shown), cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    # Compute the arithmetic from this frame's real width -- FIgLib resolution is not uniform
    # (older sequences are 2048x1536, newer ones 3072x2048).
    px = 60
    fig.suptitle(
        f"The same frame at t = {round(off/60):+d} min, as three input resolutions would present it.\n"
        f"An early plume spanning ~{px} px natively survives as ~{px*448/w:.0f} px at 448 "
        f"and ~{px*224/w:.0f} px at 224.",
        y=1.06, fontsize=10)
    fig.savefig(FIGDIR / "fig3_why_224_fails.png")
    plt.close(fig)


def main() -> int:
    if not DATA.is_dir() or not any(DATA.iterdir()):
        print(f"no data in {DATA}. Run:  python download_figlib.py --sequences 3")
        return 1
    FIGDIR.mkdir(exist_ok=True)

    name, frames = pick_sequence()
    print(f"using {name}  ({len(frames)} frames, "
          f"{round(frames[0][0]/60):+d} .. {round(frames[-1][0]/60):+d} min)")

    fig1_sequence_over_time(name, frames)
    fig2_temporal_stack(name, frames)
    fig3_why_224_fails(name, frames)

    for p in sorted(FIGDIR.glob("*.png")):
        print(f"  wrote {p}  ({p.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
