#!/usr/bin/env python3
"""make_art.py - paint the fly-connectome spike recording, headless, from a seed.

The seed picks where in the recording the painting starts, the style and palette
(unless you fix them) and drives the composition itself, so the same seed always
gives the same painting and every seed gives a different one.

examples:
  uv run make_art.py --seed 42                      # -> art/seed42_<style>_<palette>.png
  uv run make_art.py --seed 1 2 3 --style cutout  # several seeds, fixed style
  uv run make_art.py --seed 7 --style all           # one painting per style
  uv run make_art.py --seed 9 --video               # also an mp4 (uv add "imageio[ffmpeg]")
  uv run make_art.py --seed 5 --fly                 # the painting appears stroke by stroke, brushed
                                                    # on by a little fly; writes an mp4 (--fly-speed)
  uv run make_art.py                                # random seed, printed so you can reuse it
"""
import argparse
import os
import sys
import time

import numpy as np
import moderngl

import painter as P

ORBIT_TILT, FRAME_FILL = 0.25, 0.78          # same framing as realtime_art.py
NEURON_DECAY, DECAY_SPREAD = 0.88, 0.01
STEPS_PER_SEC = 60                           # sim steps per painted second, as in realtime_art.py
MAX_EDGES = 30_000
LUM_W = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ---------------------------------------------------------------- data
def load_network(data_dir, cache):
    sp_path = os.path.join(data_dir, "spikes.npy")
    if not os.path.exists(sp_path):
        sys.exit(f"[art] {sp_path} not found - run the training script first (or pass --data)")
    spikes = np.load(sp_path, mmap_mode="r")
    pos = np.load(os.path.join(data_dir, "positions.npy")).astype(np.float32)
    T, N = spikes.shape
    colors = edges = community = None
    signs = np.ones(N, np.float32)
    ra_path = os.path.join(data_dir, "render_assets.npz")
    if os.path.exists(ra_path):
        ra = np.load(ra_path)
        colors = ra["colors"].astype(np.float32)
        edges = (ra["e_pre"].astype(np.int64), ra["e_post"].astype(np.int64), ra["e_str"].astype(np.float32))
        signs = ra["signs"].astype(np.float32)
        if "community" in ra.files:
            community = ra["community"]
    W_sym = None
    if os.path.exists(cache):
        import scipy.sparse as sp
        z = np.load(cache, allow_pickle=True)
        if len(z["signs"]) == N:
            Wc = sp.csr_matrix((np.log1p(np.abs(z["vals"]).astype(np.float32)), (z["rows"], z["cols"])),
                               shape=(N, N))
            W_sym = (Wc + Wc.T).tocsr()
            if edges is None:
                co = Wc.tocoo()
                top = np.argsort(-co.data)[:MAX_EDGES]
                edges = (co.col[top], co.row[top], co.data[top])
    if colors is None:
        import colorsys
        colors = np.array([colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.85, 1.0) for i in range(N)], np.float32)
    if community is not None:
        groups, src = community, "Leiden communities"
    else:
        groups, src = np.unique(colors.round(3), axis=0, return_inverse=True)[1].ravel(), "colour groups"
    print(f"[art] {N} neurons, {T} steps, {len(np.unique(groups))} {src}, "
          f"{0 if edges is None else len(edges[0])} edges")
    return spikes, pos, colors, edges, signs, groups, W_sym


def repair_positions(p, Wcsr, seed=0):
    """Neurons without a soma location are stacked on one point; spread them over their graph neighbours."""
    p = p.copy()
    _, inv, cnt = np.unique(np.round(p, 1), axis=0, return_inverse=True, return_counts=True)
    miss = cnt[inv.ravel()] > max(3, int(0.02 * len(p)))
    n = int(miss.sum())
    if n == 0 or (~miss).sum() < 3 or Wcsr is None:
        return p
    rs = np.random.RandomState(seed + 1)
    scale = p[~miss].std(0).mean() + 1e-6
    p[miss] = p[~miss].mean(0) + rs.normal(0, 0.5 * scale, (n, 3))
    deg = np.asarray(Wcsr.sum(1)).ravel()
    mi = np.where(miss)[0]
    for _ in range(80):
        nb = (Wcsr @ p)[mi] / np.maximum(deg[mi], 1e-9)[:, None]
        ok = deg[mi] > 1e-9
        p[mi[ok]] = 0.7 * nb[ok] + 0.3 * p[mi[ok]]
    p[miss] += rs.normal(0, 0.04 * scale, (n, 3))
    return p


def canvas_layout(pos, W_sym, w, h, margin=0.10, mix=0.35):
    """Soma positions projected like realtime_art.py's front view, stretched over the canvas."""
    p = repair_positions(pos, W_sym)
    p = p - p.mean(0)
    ar = np.ptp(p, axis=0)
    fa = int(np.argmin(ar))
    if ar[fa] < 0.3 * ar.max():
        p[:, fa] *= 2.5
    p /= np.std(p) + 1e-6
    cam = 3.0 * float(np.max(np.linalg.norm(p, axis=1)))
    ct, st = np.cos(ORBIT_TILT), np.sin(ORBIT_TILT)

    def proj(a):
        ca, sa = np.cos(a), np.sin(a)
        xr, zr = p[:, 0] * ca - p[:, 2] * sa, p[:, 0] * sa + p[:, 2] * ca
        yt, zt = p[:, 1] * ct - zr * st, p[:, 1] * st + zr * ct
        zc = np.maximum(zt + cam, cam * 0.1)
        return xr / zc, yt / zc

    rmax = max(float(np.percentile(np.hypot(*proj(a)), 99.0)) for a in (0.0, np.pi / 3, 2 * np.pi / 3))
    fov = (FRAME_FILL * h / 2) / max(rmax, 1e-6)
    ux, uy = proj(0.0)
    x, y = w / 2 + fov * ux, h / 2 - fov * uy
    n = len(x)
    rank = lambda v: (np.argsort(np.argsort(v)) + 0.5) / n
    lin = lambda v: (v - v.min()) / (np.ptp(v) + 1e-9)
    u = mix * rank(x) + (1 - mix) * lin(x)
    v = mix * rank(y) + (1 - mix) * lin(y)
    return (margin + u * (1 - 2 * margin)) * w, (margin + v * (1 - 2 * margin)) * h


# ---------------------------------------------------------------- GL
def gl_context():
    err = None
    for kw in ({"backend": "egl"}, {}):          # EGL works without a display; fall back to the default
        try:
            return moderngl.create_standalone_context(require=330, **kw)
        except Exception as e:
            err = e
    sys.exit(f"[art] could not create a headless OpenGL 3.3 context: {err}")


def read_rgb(fbo, w, h):
    return np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(h, w, 3)[::-1].copy()


# ---------------------------------------------------------------- painting
def paint(ctx, net, lay, *, seed, style, palette, seconds, start, size, video_path, fps,
          fly=False, fly_speed=1.0):
    spikes, _, colors, edges, signs, groups, _ = net
    T, N = spikes.shape
    w, h = size
    tex = ctx.texture((w, h), 3)
    fbo = ctx.framebuffer([tex])

    pt = P.Painter(ctx, w, h, lay, groups, signs, edges, seed=seed, neuron_colors=colors)
    while pt.style != style:
        pt.next_style()
    pt.old_els = []                              # no cross-fade from the default style
    if palette:
        pt.pal = palette
    if fly:
        pt.fly_begin(fly_speed)                  # the painting hides until the fly brushes it
        pt.auto_recompose = False

    nd = np.array([NEURON_DECAY + DECAY_SPREAD, NEURON_DECAY, NEURON_DECAY - DECAY_SPREAD], np.float32)
    emit = (1.0 - nd) / (1.0 - nd).max()
    trail = np.zeros((N, 3), np.float32)
    step = start

    def advance():
        nonlocal step
        trail[...] *= nd
        spk = np.asarray(spikes[step % T]) > 0.5
        trail[spk] += colors[spk] * emit
        step += 1

    for _ in range(STEPS_PER_SEC):               # warm the colour trail so painting starts at once
        advance()

    writer = None
    if video_path:
        try:
            import imageio.v2 as imageio
        except ImportError:
            sys.exit('[art] --video needs imageio with ffmpeg: uv add "imageio[ffmpeg]"')
        writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=2)
    every = max(1, round(STEPS_PER_SEC / fps))

    dt = 1.0 / STEPS_PER_SEC
    min_frames = int(seconds * STEPS_PER_SEC)
    hold = int(min(4.0, 0.3 * seconds) * STEPS_PER_SEC)   # no recomposing right before the final frame
    frames = min_frames if not fly else max(min_frames, int(10 * 60 * STEPS_PER_SEC))
    t0 = time.time()
    for f in range(frames):
        if not fly:
            pt.auto_recompose = f < frames - hold
        advance()
        rate = float(np.asarray(spikes[(step - 1) % T]).mean()) * 20.0
        pt.frame(trail @ LUM_W, rate, dt, f * dt, fbo)
        if writer is not None and f % every == 0:
            writer.append_data(read_rgb(fbo, w, h))
        if fly:                                            # run until the fly has painted everything
            done = pt.fly_progress()
            if f >= min_frames and done >= 1.0:
                break
        else:
            done = (f + 1) / frames
        if f % STEPS_PER_SEC == 0:
            print(f"\r[art]   {style:<9} {done:5.0%}  ({time.time() - t0:5.1f}s)", end="", flush=True)
    print()
    img = read_rgb(fbo, w, h)
    if writer is not None:
        for _ in range(fps * 2):                 # hold the finished painting for two seconds
            writer.append_data(img)
        writer.close()
    fbo.release()
    tex.release()
    return img


def main():
    ap = argparse.ArgumentParser(description="Paint the fly-connectome spike recording from a seed.",
                                 epilog="examples:" + __doc__.split("examples:")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, nargs="+", help="one or more seeds (default: a random one)")
    ap.add_argument("--style", default="random", choices=P.STYLES + ["random", "all"])
    ap.add_argument("--palette", default="style", choices=list(P.PALETTES) + ["style", "random"],
                    help="'style' uses the style's own palette (default)")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long the network paints (default 30)")
    ap.add_argument("--start", type=int, help="first simulation step (default: chosen by the seed)")
    ap.add_argument("--size", default="1920x1080", help="WIDTHxHEIGHT (default 1920x1080)")
    ap.add_argument("--outdir", default="art")
    ap.add_argument("--video", action="store_true", help="also write an mp4 of the painting being made")
    ap.add_argument("--fly", action="store_true",
                    help="reveal mode: the painting appears stroke by stroke as a little fly brushes it (writes mp4)")
    ap.add_argument("--fly-speed", type=float, default=1.0, help="how fast the fly paints (default 1)")
    ap.add_argument("--fps", type=int, default=30, help="video frame rate (default 30)")
    ap.add_argument("--open", default=True, action=argparse.BooleanOptionalAction,
                    help="open the png in the default image viewer (default yes)")
    ap.add_argument("--data", default="out", help="folder with spikes.npy, positions.npy, render_assets.npz")
    ap.add_argument("--cache", default="connectome_cache.npz")
    args = ap.parse_args()

    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        sys.exit("[art] --size must look like 1920x1080")
    w, h = w - w % 2, h - h % 2                  # even sizes keep the video encoder happy
    seeds = args.seed or [int.from_bytes(os.urandom(4), "little") % 1_000_000]
    os.makedirs(args.outdir, exist_ok=True)

    from PIL import Image
    net = load_network(args.data, args.cache)
    lay = canvas_layout(net[1], net[6], w, h)
    ctx = gl_context()
    T = net[0].shape[0]

    for seed in seeds:
        rng = np.random.RandomState(seed)
        start = args.start if args.start is not None else int(rng.randint(T))
        styles = P.STYLES if args.style == "all" else \
            [P.STYLES[rng.randint(len(P.STYLES))] if args.style == "random" else args.style]
        for style in styles:
            if args.palette == "random":
                pal = list(P.PALETTES)[rng.randint(len(P.PALETTES))]
            else:
                pal = P.STYLE_PALETTE[style] if args.palette == "style" else args.palette
            name = f"seed{seed}_{style}_{pal}"
            print(f"[art] seed {seed}: {style} / {pal}, from step {start}")
            img = paint(ctx, net, lay, seed=seed, style=style, palette=pal, seconds=args.seconds, start=start,
                        size=(w, h),
                        video_path=os.path.join(args.outdir, name + ".mp4")
                        if (args.video or args.fly) else None,
                        fps=args.fps, fly=args.fly, fly_speed=args.fly_speed)
            path = os.path.join(args.outdir, name + ".png")
            Image.fromarray(img).save(path)
            print(f"[art]   -> {path}" + (f"  and {name}.mp4" if args.video else ""))
            if args.open:
                try:
                    if sys.platform == "win32":
                        os.startfile(path)                       # noqa: S606
                    elif sys.platform == "darwin":
                        import subprocess
                        subprocess.Popen(["open", path])
                    else:
                        import subprocess
                        subprocess.Popen(["xdg-open", path])
                except OSError as e:
                    print(f"[art]   (could not open {path}: {e})")


if __name__ == "__main__":
    main()
