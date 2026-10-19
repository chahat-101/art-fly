# Changes: dynamic painter + `make_art.py` CLI

Apply these top to bottom. Each **Find** block appears exactly once in the file as it is now (each was checked by applying it to a copy of your file), so a plain find-and-replace works. Indentation matters: copy the blocks exactly.

| File | Edits |
|---|---|
| `painter.py` | 12 |
| `realtime_art.py` | 3 |
| your training script (the one that writes `out/render_assets.npz`) | 2 |
| `make_art.py` | new file |

---

## `painter.py`

### 1. New motion constants

Recompose more often, and add the constants for cross-fades, springs, spin and drift.

**Find:**

```python
RECOMPOSE_MIN = 22.0    # s between recompositions on odour onsets
RECOMPOSE_MAX = 60.0    # s: recompose anyway
GHOST = 0.28            # how much of the previous composition stays underneath
MAX_ACCENTS = 60
```

**Replace with:**

```python
RECOMPOSE_MIN = 16.0    # s between recompositions on odour onsets
RECOMPOSE_MAX = 40.0    # s: recompose anyway
GHOST = 0.28            # how much of the previous composition stays underneath
MAX_ACCENTS = 80
XFADE = 1.6             # s cross-fade between the old and the new composition
SPRING_K, SPRING_DAMP = 6.0, 3.2   # motifs are damped springs around their home position
KICK = 260.0            # px/s a population burst pushes nearby motifs away
SPIN = 0.35             # rad/s a fully active motif turns
WANDER = 10.0           # px of slow drift
```

### 2. Extra per-element fields

Ripples, travelling pulses and sliding squares need a little state of their own.

**Find:**

```python
    __slots__ = ("kind", "g", "g2", "x", "y", "x2", "y2", "sx", "sy", "ang", "role", "seed", "param",
                 "rough", "soft", "amax", "z", "grow", "floor", "t0", "ink", "rgb")
```

**Replace with:**

```python
    __slots__ = ("kind", "g", "g2", "x", "y", "x2", "y2", "sx", "sy", "ang", "role", "seed", "param",
                 "rough", "soft", "amax", "z", "grow", "floor", "t0", "ink", "rgb",
                 "vr", "u", "v", "line", "tau")
```

### 3. Initialise the extra fields

**Find:**

```python
        self.grow, self.floor, self.t0, self.ink, self.rgb = grow, floor, t0, 1.0, None
```

**Replace with:**

```python
        self.grow, self.floor, self.t0, self.ink, self.rgb = grow, floor, t0, 1.0, None
        self.vr, self.u, self.v, self.line, self.tau = 0.0, 0.0, 0.0, None, 7.0    # accents / pulses
```

### 4. Motion state in `__init__`

Each community's motif gets a spring offset, a velocity, a rotation and a scale pulse.

**Find:**

```python
        self.fast = np.zeros(self.G)
        self.ink = np.zeros(self.G)
```

**Replace with:**

```python
        self.fast = np.zeros(self.G)
        self.prev_fast = np.zeros(self.G)
        self.ink = np.zeros(self.G)
        self.off = np.zeros((self.G, 2))              # spring displacement of each motif
        self.vel = np.zeros((self.G, 2))
        self.spin = np.zeros(self.G)                  # accumulated rotation
        self.spin_dir = self.rng.choice([-1.0, 1.0], self.G)
        self.pulse = np.zeros(self.G)                 # scale "breath" kicked by bursts
        self.pulse_v = np.zeros(self.G)
        self.old_els, self.t_old = [], -1e9
        self.auto_recompose = True                    # make_art.py switches this off near the end
```

### 5. `clear()` resets the motion too

**Find:**

```python
        self.ink[:] = 0
        self.fast[:] = 0
        self.inh_ink = 0.0
        self.compose()
```

**Replace with:**

```python
        self.ink[:] = 0
        self.fast[:] = 0
        self.inh_ink = 0.0
        self.off[:] = 0
        self.vel[:] = 0
        self.pulse[:] = 0
        self.pulse_v[:] = 0
        self.compose()
        self.old_els = []
```

### 6. `compose()` keeps the old composition for the cross-fade

Before this, a recompose was a hard cut. Now the old shapes fade out while the new ones fade in, including when you switch style with S.

**Find:**

```python
    def compose(self):
        self.t_compose = self.t
```

**Replace with:**

```python
    def compose(self):
        self.old_els, self.t_old = self.els, self.t     # fades out while the new one fades in
        self.t_compose = self.t
```

### 7. Replace `_burst()` with ripples, kicks and sliding squares

A burst now sends ripples out from the most active motif, gives it a scale pulse and pushes nearby motifs away (they spring back). In Mondrian, the Boogie-Woogie squares slide along their grid line. Also adds `_ripple()`, which `frame()` calls whenever a community switches on.

**Find:**

```python
    def _burst(self, gi):
        rng = self.rng
        if self.style == "mondrian":
            if not getattr(self, "m_lines", None):
                return
            x0, y0, x1, y1 = self.m_lines[rng.randint(len(self.m_lines))]
            u = rng.rand()
            q = self.m_T * 2.4
            for k in range(rng.randint(1, 4)):
                uu = np.clip(u + k * 0.06, 0, 1)
                e = El(K_RECT, -3, x0 + (x1 - x0) * uu, y0 + (y1 - y0) * uu, q, q,
                       role=int(rng.choice([0, 1, 2, 4])), rough=0.3, z=6, grow=False)
                self.accents.append(e)
        elif self.style != "rothko" and gi in self.main:
            x, y, ri = self.main[gi]
            for _ in range(rng.randint(2, 5)):
                a = rng.uniform(0, 2 * np.pi)
                d = ri * rng.uniform(1.1, 1.7)
                px, py = x + np.cos(a) * d, y + np.sin(a) * d
                kind = int(rng.choice([K_DISC, K_RING, K_LINE], p=[.6, .2, .2]))
                if kind == K_LINE:
                    L = rng.uniform(15, 45)
                    e = El(K_LINE, -3, px, py, 0, 1.2, x2=px + np.cos(a) * L, y2=py + np.sin(a) * L, role=3, z=6)
                elif kind == K_RING:
                    e = El(K_RING, -3, px, py, rng.uniform(6, 14), param=1.5, role=3, rough=0.4, z=6)
                else:
                    e = El(K_DISC, -3, px, py, rng.uniform(3, 9), role=int(rng.choice([3, 2, 0])), rough=0.5, z=6)
                self.accents.append(e)
        self.accents = self.accents[-MAX_ACCENTS:]
```

**Replace with:**

```python
    def _ripple(self, gi, n):
        """Thin rings that expand out of a motif and fade."""
        if self.style in ("mondrian", "rothko") or gi not in self.main:
            return
        x, y, ri = self.main[gi]
        for k in range(n):
            e = El(K_RING, -3, x, y, ri * (1.05 + 0.12 * k), param=1.3, role=3, rough=0.3, z=6,
                   amax=0.55, grow=False, g2=gi)
            e.vr, e.tau = ri * 0.9 + 40.0, 0.7 + 0.15 * k
            e.v = ri * 1.9 + 30.0                     # radius at which the ripple is gone
            self.accents.append(e)
        self.accents = self.accents[-MAX_ACCENTS:]

    def _burst(self, gi):
        rng, W = self.rng, self.w
        if self.style == "mondrian":
            if not getattr(self, "m_lines", None):
                return
            ln = self.m_lines[rng.randint(len(self.m_lines))]
            u, v = rng.rand(), rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.12)   # squares ride the line
            q = self.m_T * 2.4
            for k in range(rng.randint(1, 4)):
                e = El(K_RECT, -3, 0.0, 0.0, q, q, role=int(rng.choice([0, 1, 2, 4])), rough=0.3, z=6, grow=False)
                e.line, e.u, e.v, e.tau = ln, (u + 0.05 * k) % 1.0, v, 9.0
                x0, y0, x1, y1 = ln
                e.x, e.y = x0 + (x1 - x0) * e.u, y0 + (y1 - y0) * e.u
                self.accents.append(e)
            self.accents = self.accents[-MAX_ACCENTS:]
            return
        if gi not in self.main:
            return
        x, y, ri = self.main[gi]
        self.pulse_v[gi] += 5.0
        # push the other motifs away from the burst; their springs bring them back
        pos = np.stack([self.ax, self.ay], 1).astype(float)
        for j, (mx, my, _) in self.main.items():
            pos[j] = (mx, my)
        d = pos - np.array([x, y])
        dist = np.hypot(d[:, 0], d[:, 1]) + 1e-6
        kick = KICK * np.exp(-dist / (0.35 * W))
        kick[gi] = 0.0
        self.vel += d / dist[:, None] * kick[:, None]
        if self.style == "rothko":
            return
        self._ripple(gi, int(rng.randint(1, 3)))
        for _ in range(rng.randint(1, 4)):
            a = rng.uniform(0, 2 * np.pi)
            dd = ri * rng.uniform(1.1, 1.7)
            px, py = x + np.cos(a) * dd, y + np.sin(a) * dd
            kind = int(rng.choice([K_DISC, K_RING, K_LINE], p=[.6, .2, .2]))
            if kind == K_LINE:
                L = rng.uniform(15, 45)
                e = El(K_LINE, -3, px, py, 0, 1.2, x2=px + np.cos(a) * L, y2=py + np.sin(a) * L, role=3, z=6,
                       g2=gi)
            elif kind == K_RING:
                e = El(K_RING, -3, px, py, rng.uniform(6, 14), param=1.5, role=3, rough=0.4, z=6, g2=gi)
            else:
                e = El(K_DISC, -3, px, py, rng.uniform(3, 9), role=int(rng.choice([3, 2, 0])), rough=0.5, z=6,
                       g2=gi)
            self.accents.append(e)
        self.accents = self.accents[-MAX_ACCENTS:]
```

### 8. Replace `_instances()`

This is where the motion is drawn: spring offsets, drift, spin driven by activity, scale pulses, and Rothko fields that breathe and glow. Signal dots run along the wiring lines while both ends are active, and the whole composition cross-fades. It also fixes a bug: arcs and blobs used to scale their angle/amplitude parameter along with their size.

**Find:**

```python
    def _instances(self, pal):
        rows = []
        t = self.t
        for e in self.els + self.accents:
            p = self._presence(e)
            if p < 0.01:
                continue
            col = pal[e.role]
            if self.style == "mondrian" and e.kind == K_RECT and e.g >= 0:
                k = float(_smooth(0, 1, min(p * 1.3, 1.0)))
                col = PAPER_ARR * (1 - k) + col * k
                p = 1.0
            if e.kind == K_LINE:        # lines draw themselves from their start point
                rev = float(_smooth(0, 1, min(p * 1.4, 1.0)))
                x2 = e.x + (e.x2 - e.x) * rev
                y2 = e.y + (e.y2 - e.y) * rev
                L = np.hypot(x2 - e.x, y2 - e.y)
                if L < 0.5:
                    continue
                rows.append([(e.x + x2) / 2, (e.y + y2) / 2, L / 2, e.sy, np.arctan2(y2 - e.y, x2 - e.x),
                             K_LINE, e.seed, 0, *col, e.amax, e.rough, e.soft])
                continue
            sc = 1.0
            if e.grow:
                sc = 0.55 + 0.45 * float(_smooth(0, 1, min(p * 1.6, 1.0)))
                if e.g >= 0:
                    sc *= 1.0 + 0.035 * self.fast[e.g]
            a = e.amax * (float(_smooth(0, 0.35, p)) if e.grow else min(1.0, p))
            dx = 3 * np.sin(t * 0.13 + e.seed) if self.style in ("kandinsky", "cutout") else 0.0
            dy = 3 * np.cos(t * 0.11 + e.seed * 1.7) if self.style in ("kandinsky", "cutout") else 0.0
            rows.append([e.x + dx, e.y + dy, e.sx * sc, e.sy * sc, e.ang, e.kind, e.seed, e.param * sc,
                         *col, a, e.rough, e.soft])
        return np.asarray(rows, "f4").reshape(-1, 14)
```

**Replace with:**

```python
    def _goff(self, g):
        if g is None or g < 0 or self.style in ("mondrian", "rothko"):
            return 0.0, 0.0
        return self.off[g, 0], self.off[g, 1]

    def _instances(self, pal, dt):
        rows = []
        t = self.t
        live = self.style in ("kandinsky", "cutout")
        fin = float(_smooth(0, XFADE, t - self.t_compose))
        fout = 1.0 - float(_smooth(0, XFADE, t - self.t_old))
        layers = [(e, fout) for e in self.old_els] if fout > 0.01 else []
        layers += [(e, fin) for e in self.els] + [(e, 1.0) for e in self.accents]
        for e, xf in layers:
            p = self._presence(e)
            if p * xf < 0.01:
                continue
            col = pal[e.role]
            base = e.g if e.g >= 0 else e.g2
            ox, oy = self._goff(base)
            if self.style == "mondrian" and e.kind == K_RECT and e.g >= 0:
                k = float(_smooth(0, 1, min(p * 1.3, 1.0)))
                col = PAPER_ARR * (1 - k) + col * k
                p = 1.0
            if e.kind == K_LINE:        # lines draw themselves from their start point
                ex, ey = self._goff(e.g2 if e.g2 is not None else base)
                x0, y0 = e.x + ox, e.y + oy
                x1, y1 = e.x2 + ex, e.y2 + ey
                rev = float(_smooth(0, 1, min(p * 1.4, 1.0)))
                x2, y2 = x0 + (x1 - x0) * rev, y0 + (y1 - y0) * rev
                L = np.hypot(x2 - x0, y2 - y0)
                if L < 0.5:
                    continue
                rows.append([(x0 + x2) / 2, (y0 + y2) / 2, L / 2, e.sy, np.arctan2(y2 - y0, x2 - x0),
                             K_LINE, e.seed, 0, *col, e.amax * xf, e.rough, e.soft])
                if e.g >= 0 and e.g2 is not None and rev > 0.95:     # signals running along the wiring
                    act = min(self.fast[e.g], self.fast[e.g2])
                    if act > 0.2:
                        e.u = (e.u + dt * (0.12 + 0.5 * act)) % 1.0
                        r = e.sy * 2.0 + 2.5
                        pc = pal[2] if self.style == "kandinsky" else pal[0]
                        for k in range(2):
                            u = (e.u + 0.5 * k) % 1.0
                            rows.append([x0 + (x2 - x0) * u, y0 + (y2 - y0) * u, r, r, 0, K_DISC, e.seed, 0,
                                         *pc, min(1.0, (act - 0.2) * 3.0) * xf, 0.3, 0])
                continue
            sc = 1.0
            if e.grow:
                sc = 0.55 + 0.45 * float(_smooth(0, 1, min(p * 1.6, 1.0)))
            if live and base is not None and base >= 0:
                sc *= 1.0 + 0.035 * self.fast[base] + 0.08 * self.pulse[base]
            sx, sy, soft, ang = e.sx * sc, e.sy * sc, e.soft, e.ang
            prm = e.param * sc if e.kind == K_RING else e.param
            if live and e.g >= 0:
                ox += WANDER * np.sin(t * 0.13 + e.seed)
                oy += WANDER * np.cos(t * 0.11 + e.seed * 1.7)
                if e.kind in (K_TRI, K_HALF, K_ARC, K_BLOB) or (e.kind == K_RECT and e.z < 3):
                    ang += self.spin[e.g]
            if self.style == "rothko" and e.kind == K_SOFT and e.g >= 0:
                f = self.fast[e.g]
                sy *= 1.0 + 0.03 * np.sin(t * 0.25 + e.seed) + 0.05 * f + 0.06 * self.pulse[e.g]
                soft *= 1.0 + 0.6 * f
                col = col * (1 - 0.15 * f) + pal[4] * (0.15 * f)
            a = e.amax * (float(_smooth(0, 0.35, p)) if e.grow else min(1.0, p)) * xf
            rows.append([e.x + ox, e.y + oy, sx, sy, ang, e.kind, e.seed, prm, *col, a, e.rough, soft])
        return np.asarray(rows, "f4").reshape(-1, 14)
```

### 9. Physics step in `frame()`

Springs, pulses and spin are integrated every frame, and a community switching on sends out a ripple.

**Find:**

```python
        # odour episodes and bursts
        self.rate_ema = rate if self.rate_ema is None else 0.995 * self.rate_ema + 0.005 * rate
```

**Replace with:**

```python
        # motion: motifs are damped springs; bursts kick them, activity turns them
        self.vel += (-SPRING_K * self.off - SPRING_DAMP * self.vel) * dt
        self.off += self.vel * dt
        self.pulse_v += (-30.0 * self.pulse - 6.0 * self.pulse_v) * dt
        self.pulse += self.pulse_v * dt
        self.spin += self.spin_dir * SPIN * self.fast * dt
        rise = np.flatnonzero((self.fast > 0.5) & (self.prev_fast <= 0.5))
        self.prev_fast = self.fast.copy()
        for gi in rise[:3]:                                   # a community switching on ripples
            self._ripple(int(gi), 1)
            self.pulse_v[gi] += 3.0

        # odour episodes and bursts
        self.rate_ema = rate if self.rate_ema is None else 0.995 * self.rate_ema + 0.005 * rate
```

### 10. Let the CLI stop recompositions near the end

**Find:**

```python
        if (onset and since > RECOMPOSE_MIN) or since > RECOMPOSE_MAX:
            self.recompose()
```

**Replace with:**

```python
        if self.auto_recompose and ((onset and since > RECOMPOSE_MIN) or since > RECOMPOSE_MAX):
            self.recompose()
```

### 11. More frequent bursts; accents expand and slide

**Find:**

```python
        if rn > 1.6 and self.burst_cd <= 0:
            self.burst_cd = 1.0
            self._burst(int(np.argmax(self.fast)))
        for e in self.accents:
            e.ink *= np.exp(-dt / 12.0)
        self.accents = [e for e in self.accents if e.ink > 0.02]
```

**Replace with:**

```python
        if rn > 1.5 and self.burst_cd <= 0:
            self.burst_cd = 0.6
            self._burst(int(np.argmax(self.fast)))
        for e in self.accents:
            e.ink *= np.exp(-dt / e.tau)
            if e.vr:                                   # ripples expand, thin out and vanish
                e.sx += e.vr * dt
                e.sy = e.sx
                if e.sx > e.v:
                    e.ink = 0.0
            if e.line is not None:                     # Boogie-Woogie squares ride their line
                e.u += e.v * dt
                if e.u < 0 or e.u > 1:
                    e.v, e.u = -e.v, float(np.clip(e.u, 0, 1))
                x0, y0, x1, y1 = e.line
                e.x, e.y = x0 + (x1 - x0) * e.u, y0 + (y1 - y0) * e.u
        self.accents = [e for e in self.accents if e.ink > 0.02]
```

### 12. Pass `dt` to `_instances()`

**Find:**

```python
        rows = self._instances(PALETTES[self.pal])
```

**Replace with:**

```python
        rows = self._instances(PALETTES[self.pal], dt)
```

---

## `realtime_art.py`

### 13. Seed from the environment

Lets you choose the composition seed without editing the file, e.g. `ART_SEED=42 uv run realtime_art.py`.

**Find:**

```python
SEED = 0
```

**Replace with:**

```python
SEED = int(os.environ.get("ART_SEED", "0"))  # composition seed: ART_SEED=42 uv run realtime_art.py
```

### 14. `R` also restarts the painting

**Find:**

```python
        st.t = 0.0
        st.trail[:] = 0
        for f, t in pipe["accum"]:
```

**Replace with:**

```python
        st.t = 0.0
        st.trail[:] = 0
        painter.clear()
        for f, t in pipe["accum"]:
```

### 15. `N` recomposes on demand

**Find:**

```python
    elif key == glfw.KEY_S and action == glfw.PRESS:
        print(f"[rt] style {painter.next_style()}")
```

**Replace with:**

```python
    elif key == glfw.KEY_S and action == glfw.PRESS:
        print(f"[rt] style {painter.next_style()}")
    elif key == glfw.KEY_N and action == glfw.PRESS:
        painter.recompose()
        print("[rt] new composition")
```

---

## Training script

This is your training script (the one that writes `out/render_assets.npz`).

### 16. Seed from the environment (optional)

Lets you record different neural activity without editing the file, e.g. `FLY_SEED=3 uv run <training script>.py`. It still writes to `out/`, so copy that folder away first if you want to keep the previous run.

**Find:**

```python
SEED = 0
```

**Replace with:**

```python
SEED = int(os.environ.get("FLY_SEED", "0"))
```

### 17. Export the Leiden communities

Without this the painter falls back to colour groups, which split every community into excitatory and inhibitory halves.

**Find:**

```python
    colors=neuron_colors, e_pre=e_pre, e_post=e_post, e_str=e_str, signs=signs,
)
```

**Replace with:**

```python
    colors=neuron_colors, e_pre=e_pre, e_post=e_post, e_str=e_str, signs=signs,
    community=community_ids,
)
```

---

## `make_art.py` (new file)

Save the attached `make_art.py` next to `painter.py` and `realtime_art.py`. It needs no window: it creates an offscreen OpenGL context, replays the spike recording through the painter and saves the result.

The seed picks where in the recording the painting starts. Unless you fix them with flags, it also picks the style and palette, and it drives the composition itself. The same seed always gives the same painting.

```bash
uv run make_art.py --seed 42                      # -> art/seed42_<style>_<palette>.png
uv run make_art.py --seed 1 2 3 --style mondrian  # several seeds, fixed style
uv run make_art.py --seed 7 --style all           # one painting per style
uv run make_art.py --seed 9 --video               # also an mp4 of it being painted
uv run make_art.py --help
```

`--video` needs `uv add "imageio[ffmpeg]"`. Other flags: `--seconds` (how long the network paints, default 30), `--palette`, `--start`, `--size 3840x2160`, `--outdir`, `--data` and `--cache`.

To get different *neural activity* rather than a different painting of the same recording, re-run the training script with another `FLY_SEED` (edit 16).

## New keys in `realtime_art.py`

| Key | Does |
|---|---|
| `S` | next style (now cross-fades) |
| `P` | next palette |
| `N` | new composition |
| `C` | clear the canvas |
| `R` | restart the recording and the painting |
