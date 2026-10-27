"""painter.py - abstract-composition renderer for "signals of a smaller brain" (mode 4).

No brush agents. The painting is a composition of flat, hand-cut shapes that the
network paints in and out:

  * every large Leiden community owns one motif (circle, plane, triangle, arc, cell,
    cut-paper blob ...). It is placed near the community's anatomical centroid, then
    relaxed so the canvas has hierarchy, spacing and a focal point on a third.
  * a motif is painted in while its community fires and fades slowly when it is quiet,
    so the painting shows which parts of the antennal lobe the current odour recruits.
  * strongly connected pairs of communities are joined by black lines, which draw
    themselves while both ends are active.
  * inhibition-dominated moments draw free black lines; population bursts drop small
    accents (dots, rings, Boogie-Woogie squares) near the most active motif.
  * a new odour episode recomposes the canvas; the old composition stays ghosted
    underneath like an underpainting.

Styles (S): kandinsky, mondrian, cutout (Matisse), rothko.   Palettes (P).
"""
import numpy as np
import moderngl


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


# roles: [dominant, secondary, accent, ink, light]
PALETTES = {k: np.array([_hex(c) for c in v], np.float32) for k, v in {
    "bauhaus": ["#D7263D", "#1B2F5B", "#F2B705", "#141414", "#E4DCCB"],
    "klee":    ["#E07A5F", "#3D5A80", "#F2CC8F", "#1F1F1F", "#98C1D9"],
    "matisse": ["#E4572E", "#23408E", "#F3A712", "#141414", "#76B041"],
    "rothko":  ["#B23A2A", "#3B1F2B", "#E89F45", "#1A1416", "#D9B38C"],
    "miro":    ["#D62828", "#003049", "#FCBF49", "#111111", "#6A994E"],
}.items()}
PAPER = (0.945, 0.925, 0.878)
PAPER_ARR = np.array(PAPER, np.float32)
STYLES = ["kandinsky", "cutout"]
STYLE_PALETTE = {"kandinsky": "bauhaus", "cutout": "matisse"}

MAIN_GROUPS = 20        # communities that get a full motif; the rest become small dots
FADE_TAU = 14.0         # s for an idle motif to fade
GROW_TAU = 0.9          # s for an active motif to paint in
RECOMPOSE_MIN = 16.0    # s between recompositions on odour onsets
RECOMPOSE_MAX = 40.0    # s: recompose anyway
GHOST = 0.28            # how much of the previous composition stays underneath
MAX_ACCENTS = 80
XFADE = 1.6             # s cross-fade between the old and the new composition
SPRING_K, SPRING_DAMP = 6.0, 3.2   # motifs are damped springs around their home position
KICK = 260.0            # px/s a population burst pushes nearby motifs away
SPIN = 0.35             # rad/s a fully active motif turns
WANDER = 10.0           # px of slow drift

# fly reveal: the painting only shows where a little fly has brushed it
FLY_SPEED = 380.0       # px/s flight speed at fly_speed = 1
FLY_RATE = 13.0         # brush dabs per second at fly_speed = 1
FLY_BRUSH = 26.0        # default brush radius, px
FLY_TO_BRUSH = (13.0, 25.0)   # fly body -> brush tip offset (canvas space, y down)

K_DISC, K_RING, K_RECT, K_TRI, K_LINE, K_ARC, K_BLOB, K_SOFT, K_HALF = range(9)

# ------------------------------------------------------------------ shaders
VS_SHAPE = """
#version 330
uniform vec2 u_res;
in vec2 in_quad; in vec4 in_a; in vec4 in_b; in vec4 in_c; in vec2 in_d;
// a: cx cy sx sy | b: angle kind seed param | c: rgba | d: rough soft
out vec2 v_q; flat out vec4 v_a; flat out vec4 v_b; flat out vec4 v_c; flat out vec2 v_d;
void main(){
    int k = int(in_b.y + 0.5);
    vec2 s = in_a.zw;
    float pad = in_d.x*2.5 + in_d.y + 3.0;
    vec2 ext;
    if (k == 2 || k == 7) ext = s + pad;
    else if (k == 4)      ext = vec2(s.x + s.y, s.y) + pad;
    else { float R = max(s.x, s.y); ext = vec2(R*1.55 + in_b.w + pad); }
    vec2 q = in_quad*ext;
    float c = cos(in_b.x), sn = sin(in_b.x);
    vec2 p = in_a.xy + vec2(c*q.x - sn*q.y, sn*q.x + c*q.y);
    gl_Position = vec4(p/u_res*2.0 - 1.0, 0.0, 1.0);
    v_q = q; v_a = in_a; v_b = in_b; v_c = in_c; v_d = in_d;
}"""

NOISE = """
float h2(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)))*43758.5453); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f*f*(3.0 - 2.0*f);
    return mix(mix(h2(i), h2(i + vec2(1,0)), f.x), mix(h2(i + vec2(0,1)), h2(i + vec2(1,1)), f.x), f.y);
}"""

FS_SHAPE = """
#version 330
in vec2 v_q; flat in vec4 v_a; flat in vec4 v_b; flat in vec4 v_c; flat in vec2 v_d;
out vec4 f;
""" + NOISE + """
float sdBox(vec2 p, vec2 b){ vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float sdTri(vec2 p, float r){
    const float k = 1.7320508;
    p.x = abs(p.x) - r; p.y = p.y + r/k;
    if (p.x + k*p.y > 0.0) p = vec2(p.x - k*p.y, -k*p.x - p.y)/2.0;
    p.x -= clamp(p.x, -2.0*r, 0.0);
    return -length(p)*sign(p.y);
}
float sdArc(vec2 p, float ra, float hs, float th){
    float a = atan(p.y, p.x);
    if (abs(a) <= hs) return abs(length(p) - ra) - th;
    vec2 e = ra*vec2(cos(hs), sin(hs));
    return min(length(p - e), length(p - vec2(e.x, -e.y))) - th;
}
void main(){
    int k = int(v_b.y + 0.5);
    vec2 p = v_q, s = v_a.zw;
    float seed = v_b.z, prm = v_b.w, d;
    if      (k == 0) d = length(p) - s.x;
    else if (k == 1) d = abs(length(p) - s.x) - prm;
    else if (k == 2 || k == 7) d = sdBox(p, s);
    else if (k == 3) d = sdTri(p, s.x*0.866);
    else if (k == 4) d = length(p - vec2(clamp(p.x, -s.x, s.x), 0.0)) - s.y;
    else if (k == 5) d = sdArc(p, s.x, prm, s.y);
    else if (k == 6) {
        float a = atan(p.y, p.x);
        float w = 1.0 + prm*(sin(3.0*a + seed) + 0.6*sin(5.0*a + seed*2.3) + 0.3*sin(9.0*a + seed*3.1));
        d = (length(p/s) - w)*min(s.x, s.y);
    }
    else d = max(length(p) - s.x, -p.y);                       // half disc

    float r = v_d.x;                                            // hand-cut / brushed edge
    if (k == 7) d += (vnoise(p*0.012 + seed) - 0.5)*2.0*r + (vnoise(p*0.05 + seed*3.0) - 0.5)*0.5*r;
    else        d += (vnoise(p*0.05 + seed) - 0.5)*2.0*r + (vnoise(p*0.3 + seed*3.0) - 0.5)*0.7*r;

    float a;
    if (v_d.y > 0.0) a = 1.0 - smoothstep(-v_d.y, v_d.y*0.35, d);
    else             a = 1.0 - smoothstep(-0.8, 0.8, d);
    if (a <= 0.002) discard;

    // gouache: slow value variation, faint brush direction, pooled pigment at the edge
    vec2 g = gl_FragCoord.xy;
    float tex = 0.92 + 0.08*vnoise(g*0.012 + seed) + 0.05*(vnoise(vec2(p.x*0.015, p.y*0.5) + seed) - 0.5);
    if (k == 7) tex = 0.9 + 0.12*vnoise(p*vec2(0.004, 0.02) + seed) + 0.06*(vnoise(vec2(p.x*0.01, p.y*0.25)) - 0.5);
    float edge = (v_d.y > 0.0) ? 0.0 : smoothstep(-7.0, 0.0, d);
    vec3 c = v_c.rgb*tex*(1.0 - 0.08*edge);
    f = vec4(c, a*v_c.a);
}"""

VS_QUAD = """
#version 330
in vec2 in_quad; out vec2 v_uv;
void main(){ v_uv = in_quad*0.5 + 0.5; gl_Position = vec4(in_quad, 0.0, 1.0); }"""

FS_COPY = """
#version 330
uniform sampler2D u_tex; uniform float u_alpha; in vec2 v_uv; out vec4 f;
void main(){ f = vec4(texture(u_tex, v_uv).rgb, u_alpha); }"""

FS_WASH = """
#version 330
uniform vec3 u_paper; uniform float u_alpha; out vec4 f;
void main(){ f = vec4(u_paper, u_alpha); }"""

VS_DAB = """
#version 330
uniform vec2 u_res;
in vec2 in_quad; in vec4 in_dab;                  // x y r a
out vec2 v_q; flat out vec4 v_dab;
void main(){
    vec2 q = in_quad*in_dab.z*1.35 + in_dab.xy;
    v_q = q; v_dab = in_dab;
    gl_Position = vec4(q/u_res*2.0 - 1.0, 0.0, 1.0);
}"""

FS_DAB = """
#version 330
in vec2 v_q; flat in vec4 v_dab; out vec4 f;
""" + NOISE + """
void main(){
    float w = 0.85 + 0.3*(vnoise(v_q*0.05 + v_dab.xy*0.003) - 0.5);          // ragged brush load
    float a = v_dab.w * w * (1.0 - smoothstep(v_dab.z*0.55, v_dab.z, length(v_q - v_dab.xy)));
    f = vec4(a);
}"""

FS_MASKED = """
#version 330
uniform sampler2D u_layer, u_mask; in vec2 v_uv; out vec4 f;
void main(){
    float m = min(1.0, texture(u_mask, v_uv).r);
    f = vec4(texture(u_layer, v_uv).rgb, m);
}"""

FS_POST = """
#version 330
uniform sampler2D u_tex; in vec2 v_uv; out vec4 f;
""" + NOISE + """
void main(){
    vec3 c = texture(u_tex, v_uv).rgb;
    vec2 g = gl_FragCoord.xy;
    float paper = 0.975 + 0.025*vnoise(g*0.35) + 0.015*(vnoise(g*vec2(0.02, 0.6)) - 0.5);   // paper tooth + fibre
    c *= paper;
    vec2 q = v_uv*2.0 - 1.0;
    c *= 1.0 - 0.07*dot(q, q);
    c += (h2(g) - 0.5)/255.0*2.0;
    f = vec4(c, 1.0);
}"""


def _smooth(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0, 1)
    return u * u * (3 - 2 * u)


class El:
    __slots__ = ("kind", "g", "g2", "x", "y", "x2", "y2", "sx", "sy", "ang", "role", "seed", "param",
                 "rough", "soft", "amax", "z", "grow", "floor", "t0", "ink", "rgb",
                 "vr", "u", "v", "line", "tau")

    def __init__(self, kind, g, x, y, sx, sy=None, ang=0.0, role=0, seed=0.0, param=0.0, rough=1.2,
                 soft=0.0, amax=0.94, z=1, grow=True, g2=None, x2=None, y2=None, floor=0.0, t0=0.0):
        self.kind, self.g, self.g2 = kind, g, g2
        self.x, self.y, self.x2, self.y2 = x, y, x2, y2
        self.sx, self.sy = sx, (sx if sy is None else sy)
        self.ang, self.role, self.seed, self.param = ang, role, seed, param
        self.rough, self.soft, self.amax, self.z = rough, soft, amax, z
        self.grow, self.floor, self.t0, self.ink, self.rgb = grow, floor, t0, 1.0, None
        self.vr, self.u, self.v, self.line, self.tau = 0.0, 0.0, 0.0, None, 7.0    # accents / pulses


class Painter:
    def __init__(self, ctx, w, h, layout_xy, groups, signs, edges=None, seed=0, neuron_colors=None):
        self.ctx, self.w, self.h = ctx, w, h
        self.rng = np.random.RandomState(seed)
        x, y = (np.asarray(v, np.float64) for v in layout_xy)
        self.inh = np.asarray(signs) < 0

        # communities relabelled by size rank (0 = largest)
        _, g = np.unique(np.asarray(groups), return_inverse=True)
        g = g.ravel()
        sizes = np.bincount(g)
        rank = np.empty_like(sizes)
        rank[np.argsort(-sizes)] = np.arange(len(sizes))
        self.g = rank[g]
        self.G = len(sizes)
        self.g_n = np.bincount(self.g, minlength=self.G).astype(np.float64)
        cx = np.bincount(self.g, x, self.G) / self.g_n
        cy = np.bincount(self.g, y, self.G) / self.g_n
        self.M = min(MAIN_GROUPS, self.G)
        # anatomical anchors: stretch the main communities' centroids over the canvas
        lo_x, hi_x = cx[:self.M].min(), cx[:self.M].max()
        lo_y, hi_y = cy[:self.M].min(), cy[:self.M].max()
        ux = (cx - lo_x) / (hi_x - lo_x) if hi_x - lo_x > 1e-6 else self.rng.rand(self.G)
        uy = (cy - lo_y) / (hi_y - lo_y) if hi_y - lo_y > 1e-6 else self.rng.rand(self.G)
        self.ax = (0.12 + 0.76 * np.clip(ux, 0, 1)) * w
        self.ay = (0.14 + 0.72 * np.clip(uy, 0, 1)) * h

        # which community pairs are strongly wired
        self.S = np.zeros((self.G, self.G))
        if edges is not None:
            pre, post, s = edges
            gp, gq = self.g[np.asarray(pre)], self.g[np.asarray(post)]
            m = gp != gq
            np.add.at(self.S, (gp[m], gq[m]), np.asarray(s)[m])
            self.S += self.S.T

        # mean anatomical colour per community (the fly's brain badge)
        self.com_col = None
        if neuron_colors is not None:
            cn = np.asarray(neuron_colors, np.float64)
            self.com_col = np.stack([np.bincount(self.g, cn[:, k], self.G) / self.g_n
                                     for k in range(3)], 1) + 1e-9

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
        self.act_ref = 1e-9
        self.inh_ema, self.inh_fast, self.inh_ink = None, 0.0, 0.0
        self.rate_ema, self.quiet = None, 0.0
        self.burst_cd = 0.0
        # fly reveal state
        self.fly_on = False
        self.fly_speed = 1.0                          # dab/flight rate multiplier
        self.fly_x, self.fly_y = w * 0.07, h * 0.90
        self.fly_ang = -np.pi / 2
        self.fly_st = "rest"                          # rest | fly | work
        self.fly_plan, self.fly_set, self.fly_prog = [], set(), {}
        self.fly_i = 0
        self.fly_dabs, self.fly_di, self.fly_dabt = [], 0, 0.0
        self.fly_stamp = []                           # (x, y, r, alpha) dabs for this frame
        self.t = 0.0
        self.style = "kandinsky"
        self.pal = STYLE_PALETTE[self.style]
        self.accents, self.els, self.main = [], [], {}
        self._build_gl()
        self.clear()

    # ---------------------------------------------------------------- GL
    def _build_gl(self):
        ctx, w, h = self.ctx, self.w, self.h

        def fbo():
            t = ctx.texture((w, h), 4, dtype="f2")
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
            return ctx.framebuffer([t]), t

        self.canvas, self.canvas_tex = fbo()
        self.under, self.under_tex = fbo()
        self.layer, self.layer_tex = fbo()
        rt = ctx.texture((w, h), 1, dtype="f2")       # reveal mask: paint shows where this is filled
        rt.filter = (moderngl.LINEAR, moderngl.LINEAR)
        rt.repeat_x = rt.repeat_y = False
        self.reveal_fbo, self.reveal_tex = ctx.framebuffer([rt]), rt
        q = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32))
        self.p_shape = ctx.program(vertex_shader=VS_SHAPE, fragment_shader=FS_SHAPE)
        self.p_copy = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_COPY)
        self.p_wash = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_WASH)
        self.p_post = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_POST)
        self.p_dab = ctx.program(vertex_shader=VS_DAB, fragment_shader=FS_DAB)
        self.p_mask = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_MASKED)
        self.p_shape["u_res"] = (w, h)
        self.p_dab["u_res"] = (w, h)
        self.b_inst = ctx.buffer(reserve=1024 * 14 * 4, dynamic=True)
        self.b_dab = ctx.buffer(reserve=128 * 6 * 4, dynamic=True)
        self.v_shape = ctx.vertex_array(self.p_shape, [
            (q, "2f", "in_quad"), (self.b_inst, "4f 4f 4f 2f/i", "in_a", "in_b", "in_c", "in_d")])
        self.v_copy = ctx.vertex_array(self.p_copy, [(q, "2f", "in_quad")])
        self.v_wash = ctx.vertex_array(self.p_wash, [(q, "2f", "in_quad")])
        self.v_post = ctx.vertex_array(self.p_post, [(q, "2f", "in_quad")])
        self.v_mask = ctx.vertex_array(self.p_mask, [(q, "2f", "in_quad")])
        self.v_dab = ctx.vertex_array(self.p_dab, [
            (q, "2f", "in_quad"), (self.b_dab, "4f/i", "in_dab")])

    def _wash(self, fbo, alpha):
        fbo.use()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.p_wash["u_paper"], self.p_wash["u_alpha"] = PAPER, float(alpha)
        self.v_wash.render(moderngl.TRIANGLE_STRIP)

    def _copy(self, tex, alpha):
        tex.use(0)
        self.p_copy["u_tex"], self.p_copy["u_alpha"] = 0, float(alpha)
        self.v_copy.render(moderngl.TRIANGLE_STRIP)

    # ----------------------------------------------------------- controls
    def clear(self):
        self.ctx.disable(moderngl.BLEND)
        self._wash(self.under, 1.0)
        self._wash(self.canvas, 1.0)
        self.reveal_fbo.clear(0, 0, 0, 0)
        self.ink[:] = 0
        self.fast[:] = 0
        self.inh_ink = 0.0
        self.off[:] = 0
        self.vel[:] = 0
        self.pulse[:] = 0
        self.pulse_v[:] = 0
        self.compose()
        self.old_els = []
        if self.fly_on:
            self.fly_prog = {}                        # the fly repaints everything from scratch

    def fly_begin(self, speed=1.0):
        """Reveal mode on: the canvas hides until the fly brushes it, and the fly
        repaints the whole composition from scratch."""
        self.fly_speed = float(speed)
        self.fly_prog = {}
        self.fly_plan = self._fly_plan()
        self.fly_set = set(self.els)
        self.fly_i = 0
        self.fly_dabs, self.fly_di, self.fly_dabt = [], 0, 0.0
        self.fly_stamp = []
        self.fly_st = "fly"
        self.fly_x, self.fly_y = self.w * 0.07, self.h * 0.90
        self.fly_ang = -np.pi / 2
        self.fly_on = True
        self.reveal_fbo.clear(0, 0, 0, 0)

    def fly_stop(self):
        self.fly_on = False
        self.fly_stamp = []

    def fly_progress(self):
        plan = self.fly_plan
        if self.fly_on and plan:
            return float(np.mean([self.fly_prog.get(e, 0.0) for e in plan]))
        return 1.0

    def next_style(self):
        self.style = STYLES[(STYLES.index(self.style) + 1) % len(STYLES)]
        self.pal = STYLE_PALETTE[self.style]
        self.ctx.disable(moderngl.BLEND)
        self._wash(self.under, 1.0)
        self.compose()
        return self.style

    def next_palette(self):
        names = list(PALETTES)
        self.pal = names[(names.index(self.pal) + 1) % len(names)]
        return self.pal

    def recompose(self):
        """Ghost the current painting into the underpainting and start a new composition."""
        ctx = self.ctx
        self.under.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._copy(self.canvas_tex, GHOST)
        ctx.disable(moderngl.BLEND)
        self.compose()

    # -------------------------------------------------------- composition
    def _roles(self, n, first=None):
        w = np.array([0.40, 0.30, 0.12, 0.10, 0.08])
        out, acc = [], 0
        for i in range(n):
            r = int(self.rng.choice(5, p=w))
            if r == 2:
                acc += 1
                if acc > 2:
                    r = int(self.rng.choice([0, 1]))
            out.append(r)
        if first is not None and n:
            out[0] = first
        return out

    def _radii(self, r0, decay, rmin):
        r = self.h * r0 * decay ** np.arange(self.M) * self.rng.lognormal(0, 0.35, self.M)
        return np.maximum(r, rmin * self.h)

    def _relax(self, r, overlap):
        W, H = self.w, self.h
        anchor = np.stack([self.ax[:self.M], self.ay[:self.M]], 1) + self.rng.normal(0, 0.09 * W, (self.M, 2))
        p = anchor.copy()
        thirds = np.array([[W / 3, H / 3], [2 * W / 3, H / 3], [W / 3, 2 * H / 3], [2 * W / 3, 2 * H / 3]])
        focus = thirds[np.argmin(((thirds - p[0]) ** 2).sum(1))]
        for _ in range(90):
            d = p[:, None, :] - p[None, :, :]
            dist = np.sqrt((d ** 2).sum(-1)) + 1e-6
            push = np.clip((r[:, None] + r[None, :]) * overlap - dist, 0, None) / dist
            np.fill_diagonal(push, 0)
            wgt = r[None, :] / (r[:, None] + r[None, :])          # small motifs give way
            p += 0.5 * (d * (push * wgt)[..., None]).sum(1)
            p += 0.02 * (anchor - p)
            p[0] += 0.12 * (focus - p[0])
            p[:, 0] = np.clip(p[:, 0], 0.7 * r + 0.04 * W, W - 0.7 * r - 0.04 * W)
            p[:, 1] = np.clip(p[:, 1], 0.7 * r + 0.05 * H, H - 0.7 * r - 0.05 * H)
        return p

    def _pairs(self, k):
        S = self.S[:self.M, :self.M]
        iu = np.triu_indices(self.M, 1)
        v = S[iu]
        order = np.argsort(-v)[:k]
        return [(int(iu[0][o]), int(iu[1][o])) for o in order if v[o] > 0]

    def compose(self):
        self.old_els, self.t_old = self.els, self.t     # fades out while the new one fades in
        self.t_compose = self.t
        self.accents = []
        self.main = {}
        els = getattr(self, "_c_" + self.style)()
        els.sort(key=lambda e: e.z)
        self.els = els
        self.fly_plan = self._fly_plan()
        self.fly_i = 0
        self.fly_st, self.fly_dabs, self.fly_di = "fly", [], 0
        self.fly_set = set(els)

    def _fly_plan(self):
        """Painting order: big background shapes first, small details last."""
        size = lambda e: (np.hypot(e.x2 - e.x, e.y2 - e.y) * max(e.sy, 1) if e.kind == K_LINE
                          else abs(e.sx * e.sy) + 4.0)
        return sorted(self.els, key=lambda e: (e.z, -size(e)))

    def _c_kandinsky(self):
        rng, W, H, M = self.rng, self.w, self.h, self.M
        r = self._radii(0.19, 0.76, 0.028)
        p = self._relax(r, 0.78)
        roles = self._roles(M)
        els = []
        for i in range(M):
            x, y, ri, sd = p[i, 0], p[i, 1], r[i], rng.uniform(0, 100)
            self.main[i] = (x, y, ri)
            if i == 0:   # focal circle with a soft halo and a thin orbit
                els.append(El(K_DISC, i, x, y, ri * 1.3, role=2, amax=0.55, soft=ri * 0.12, rough=3, z=0, seed=sd))
                els.append(El(K_DISC, i, x, y, ri, role=int(rng.choice([1, 3])), amax=0.97, rough=1.5, z=1, seed=sd))
                els.append(El(K_RING, i, x, y, ri * 1.55, param=1.6, role=3, amax=0.85, rough=0.6, z=4, seed=sd))
                continue
            kind = int(rng.choice([K_DISC, K_RING, K_TRI, K_HALF, K_RECT, K_ARC], p=[.32, .1, .18, .12, .14, .14]))
            ang = rng.uniform(-np.pi, np.pi)
            if kind == K_RECT:
                els.append(El(K_RECT, i, x, y, ri * rng.uniform(0.6, 1.0), ri * rng.uniform(0.25, 0.8),
                              ang=rng.uniform(-0.6, 0.6), role=roles[i], rough=1.0, z=2, seed=sd))
            elif kind == K_RING:
                els.append(El(K_RING, i, x, y, ri, param=max(2.0, ri * rng.uniform(0.06, 0.14)),
                              role=roles[i], rough=1.0, z=2, seed=sd))
            elif kind == K_ARC:
                els.append(El(K_ARC, i, x, y, ri, max(1.5, ri * 0.07), ang=ang, param=rng.uniform(0.6, 1.6),
                              role=int(rng.choice([1, 3])), rough=0.6, z=3, seed=sd))
            elif kind == K_TRI:
                els.append(El(K_TRI, i, x, y, ri * 1.1, ang=ang, role=roles[i], rough=1.0, z=2, seed=sd))
            elif kind == K_HALF:
                els.append(El(K_HALF, i, x, y, ri, ang=ang, role=roles[i], rough=1.0, z=2, seed=sd))
            else:
                els.append(El(K_DISC, i, x, y, ri, role=roles[i], amax=0.9, rough=1.2, z=2, seed=sd))
                if rng.rand() < 0.4:   # concentric circles
                    inner = (roles[i] + int(rng.randint(1, 4))) % 5
                    els.append(El(K_DISC, i, x, y, ri * 0.55, role=inner, rough=1.0, z=3, seed=sd + 1))
        # chequer patch next to a mid-size motif
        if M > 3:
            i = int(rng.randint(2, min(6, M)))
            x, y, ri = self.main[i]
            a = rng.uniform(0, 2 * np.pi)
            ox, oy = x + np.cos(a) * ri * 1.6, y + np.sin(a) * ri * 1.6
            q = max(6.0, ri * 0.2)
            for u in range(3):
                for v in range(3):
                    if (u + v) % 2 == 0 or rng.rand() < 0.3:
                        els.append(El(K_RECT, i, ox + (u - 1) * q * 2.2, oy + (v - 1) * q * 2.2, q, q,
                                      role=[2, 3, 0, 1][(u * 3 + v) % 4], rough=0.6, z=3, seed=rng.uniform(0, 100)))
        # lines along the strongest inter-community wiring
        for i, j in self._pairs(7):
            a, b = p[i], p[j]
            d = b - a
            if np.hypot(*d) < 0.08 * W:
                continue
            s0, s1 = a - d * rng.uniform(0.15, 0.4), b + d * rng.uniform(0.15, 0.4)
            els.append(El(K_LINE, i, s0[0], s0[1], 0, rng.uniform(1.0, 2.6), g2=j, x2=s1[0], y2=s1[1],
                          role=3, rough=0.3, z=5))
        # a fan of thin lines off the second motif
        if M > 1:
            x, y, ri = self.main[1]
            base = rng.uniform(0, 2 * np.pi)
            for k in range(5):
                a = base + k * 0.08
                els.append(El(K_LINE, 1, x, y, 0, 0.9, x2=x + np.cos(a) * H * 0.3, y2=y + np.sin(a) * H * 0.3,
                              role=3, rough=0.2, z=5))
        # free lines driven by inhibition
        for _ in range(3):
            cx, cy, a, L = rng.uniform(0.2, 0.8) * W, rng.uniform(0.2, 0.8) * H, rng.uniform(0, np.pi), \
                rng.uniform(0.3, 0.6) * W
            els.append(El(K_LINE, -2, cx - np.cos(a) * L / 2, cy - np.sin(a) * L / 2, 0, rng.uniform(0.8, 1.8),
                          x2=cx + np.cos(a) * L / 2, y2=cy + np.sin(a) * L / 2, role=3, rough=0.3, z=5))
        els += self._dots(rng)
        els += self._fill(rng, 70, "kandinsky")
        return els

    def _c_cutout(self):
        rng, M = self.rng, self.M
        r = self._radii(0.16, 0.8, 0.03)
        p = self._relax(r, 0.72)
        roles = self._roles(M, first=1)
        els = []
        for i in range(M):
            x, y, ri, sd = p[i, 0], p[i, 1], r[i], rng.uniform(0, 100)
            self.main[i] = (x, y, ri)
            ang = rng.uniform(-np.pi, np.pi)
            if i < 3:      # big cut-paper planes
                els.append(El(K_RECT, i, x, y, ri * rng.uniform(0.75, 1.1), ri * rng.uniform(0.55, 0.95),
                              ang=rng.uniform(-0.5, 0.5), role=roles[i], rough=2.2, z=1 + i * 0.1, seed=sd))
                continue
            kind = rng.choice(3, p=[0.55, 0.28, 0.17])
            if kind == 0:
                els.append(El(K_BLOB, i, x, y, ri * rng.uniform(0.7, 1.05), ri * rng.uniform(0.7, 1.2), ang=ang,
                              param=rng.uniform(0.1, 0.22), role=roles[i], rough=1.8, z=2, seed=sd))
            elif kind == 1:   # leaf / seaweed
                els.append(El(K_BLOB, i, x, y, ri * 0.32, ri * 1.15, ang=ang, param=rng.uniform(0.15, 0.25),
                              role=roles[i], rough=1.6, z=2, seed=sd))
            else:
                els.append(El(K_DISC, i, x, y, ri * 0.7, role=roles[i], rough=2.0, z=2, seed=sd))
        els += self._dots(rng)
        els += self._fill(rng, 45, "cutout")
        return els

    def _fill(self, rng, n, style):
        """Small marks scattered into the empty parts of the canvas; they appear over the first seconds."""
        W, H = self.w, self.h
        pts = np.array([[m[0], m[1]] for m in self.main.values()])
        rad = np.array([m[2] for m in self.main.values()])
        placed, els = [], []
        for _ in range(n * 12):
            if len(els) >= n:
                break
            x, y = rng.uniform(0.03, 0.97) * W, rng.uniform(0.04, 0.96) * H
            if (np.hypot(pts[:, 0] - x, pts[:, 1] - y) < rad * 0.9).any():
                continue
            if any(np.hypot(px - x, py - y) < rng.uniform(0.01, 0.07) * W for px, py in placed):
                continue
            placed.append((x, y))
            sd, t0, a = rng.uniform(0, 100), self.t + rng.uniform(0, 10), rng.uniform(0, 2 * np.pi)
            sz = min(0.06, 0.008 * rng.pareto(1.6) + 0.008) * H
            role = int(rng.choice([0, 1, 2, 3, 4], p=[.3, .25, .15, .2, .1]))
            if style == "cutout":
                if rng.rand() < 0.6:
                    els.append(El(K_BLOB, -1, x, y, sz * 1.6, sz * rng.uniform(0.5, 1.5), ang=a,
                                  param=rng.uniform(0.1, 0.25), role=role, rough=1.6, z=2, seed=sd, t0=t0))
                else:
                    els.append(El(K_DISC, -1, x, y, sz, role=role, rough=1.8, z=2, seed=sd, t0=t0))
            else:
                k = rng.choice(4, p=[.4, .25, .2, .15])
                if k == 0:
                    els.append(El(K_DISC, -1, x, y, sz * 0.6, role=role, rough=0.8, z=6, seed=sd, t0=t0))
                elif k == 1:
                    els.append(El(K_RING, -1, x, y, sz, param=1.4, role=3, rough=0.6, z=4, seed=sd, t0=t0))
                elif k == 2:
                    L = sz * rng.uniform(2, 5)
                    els.append(El(K_LINE, -1, x, y, 0, rng.uniform(0.8, 1.6), x2=x + np.cos(a) * L,
                                  y2=y + np.sin(a) * L, role=3, rough=0.3, z=5, t0=t0))
                else:
                    els.append(El(K_TRI, -1, x, y, sz, ang=a, role=role, rough=1.0, z=3, seed=sd, t0=t0))
        return els

    def _c_mondrian(self):
        rng, W, H = self.rng, self.w, self.h
        cells, lines = [(0.0, 0.0, float(W), float(H))], []
        for _ in range(rng.randint(8, 12)):
            area = np.array([(c[2] - c[0]) * (c[3] - c[1]) for c in cells])
            ok = np.array([max(c[2] - c[0], c[3] - c[1]) > 0.3 * H for c in cells], float)
            if ok.sum() == 0:
                break
            i = int(rng.choice(len(cells), p=area * ok / (area * ok).sum()))
            x0, y0, x1, y1 = cells.pop(i)
            cw, ch = x1 - x0, y1 - y0
            vert = (rng.rand() < (0.75 if cw > ch else 0.25)) if min(cw, ch) > 0.3 * H else cw > ch
            if vert:
                s = x0 + cw * rng.uniform(0.25, 0.75)
                cells += [(x0, y0, s, y1), (s, y0, x1, y1)]
                lines.append((s, y0, s, y1))
            else:
                s = y0 + ch * rng.uniform(0.25, 0.75)
                cells += [(x0, y0, x1, s), (x0, s, x1, y1)]
                lines.append((x0, s, x1, s))
        T = max(3.0, 0.006 * H)
        area = np.array([(c[2] - c[0]) * (c[3] - c[1]) for c in cells])
        order = np.argsort(-area)
        cand = list(order[1:])                                      # the biggest cell stays white
        rng.shuffle(cand)
        n_col = min(len(cand), max(3, min(self.M, 7)))
        free_groups = list(range(self.M))
        els = []
        for k, ci in enumerate(cand):
            x0, y0, x1, y1 = cells[ci]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if not free_groups:
                break
            gi = min(free_groups, key=lambda q: (self.ax[q] - mx) ** 2 + (self.ay[q] - my) ** 2)
            free_groups.remove(gi)
            role = [0, 1, 2][k] if k < 3 else (int(rng.choice([0, 1, 2, 3], p=[.35, .3, .25, .1])) if k < n_col else 4)
            els.append(El(K_RECT, gi, mx, my, (x1 - x0) / 2 - T, (y1 - y0) / 2 - T, role=role,
                          amax=0.95 if role != 4 else 0.55, rough=0.5, grow=False, z=1, seed=rng.uniform(0, 100)))
            self.main[gi] = (mx, my, min(x1 - x0, y1 - y0) / 2)
        for k, (x0, y0, x1, y1) in enumerate(lines):
            els.append(El(K_LINE, -1, x0, y0, 0, T, x2=x1, y2=y1, role=3, rough=0.3, z=5, t0=self.t + 0.25 * k))
        self.m_lines, self.m_T = lines, T
        return els

    def _c_rothko(self):
        rng, W, H = self.rng, self.w, self.h
        els = [El(K_SOFT, -1, W / 2, H / 2, W / 2 + 30, H / 2 + 30, role=1, soft=4, rough=3, amax=1.0,
                  grow=False, z=0, t0=self.t)]
        k = min(3, self.M)
        top = sorted(range(k), key=lambda q: -self.ay[q])            # stack in anatomical order
        wts = np.sqrt(self.g_n[top])
        m, gap = 0.08 * H, 0.045 * H
        avail = H - 2 * m - gap * (k - 1)
        hs = avail * wts / wts.sum()
        yy = H - m
        roles = [0, 2, 4] if rng.rand() < 0.5 else [2, 0, 4]
        for n, gi in enumerate(top):
            cy = yy - hs[n] / 2
            els.append(El(K_SOFT, gi, W / 2, cy, W * 0.39, hs[n] / 2, role=roles[n], soft=34, rough=9,
                          amax=0.95, grow=False, floor=0.3, z=1, seed=rng.uniform(0, 100)))
            self.main[gi] = (W / 2, cy, hs[n] / 2)
            yy -= hs[n] + gap
        return els

    def _dots(self, rng):
        els = []
        for gi in range(self.M, min(self.G, self.M + 60)):
            els.append(El(K_DISC, gi, self.ax[gi] + rng.normal(0, 20), self.ay[gi] + rng.normal(0, 20),
                          rng.uniform(4, 11), role=int(rng.choice([3, 2, 1], p=[.6, .2, .2])), rough=0.6, z=6,
                          seed=rng.uniform(0, 100)))
        return els

    # ---------------------------------------------------------- fly reveal
    def _fly_dabs(self, e):
        """Brush dab positions over one element's footprint, plus the brush radius."""
        rng, r = self.rng, 0.0
        pts = []
        jitter = rng.uniform
        if e.kind == K_LINE:
            L = float(np.hypot(e.x2 - e.x, e.y2 - e.y))
            r = float(np.clip(e.sy * 2.5 + 4.0, 7.0, 26.0))
            n = int(np.clip(2.2 * L / max(0.75 * r, 1.0), 3, 40))
            ux, uy = (e.x2 - e.x) / max(L, 1e-6), (e.y2 - e.y) / max(L, 1e-6)
            px, py = -uy, ux
            for k in range(n):
                u = (k + 0.5) / n
                j = rng.normal(0.0, 0.3 * e.sy + 0.2 * r)
                pts.append((e.x + (e.x2 - e.x) * u + px * j, e.y + (e.y2 - e.y) * u + py * j))
        elif e.kind == K_RECT:
            r = float(np.clip(0.16 * max(e.sx, e.sy) + 8.0, 10.0, 56.0))
            nx = int(np.clip(2 * e.sx / (0.7 * r), 1, 12))
            ny = int(np.clip(2 * e.sy / (0.7 * r), 1, 12))
            c, sn = np.cos(e.ang), np.sin(e.ang)
            for v in range(ny):
                for u in range(nx):
                    lx = (u + 0.5) / nx * 2 - 1
                    ly = (v + 0.5) / ny * 2 - 1
                    lx += rng.normal(0, 0.10)
                    ly += rng.normal(0, 0.10)
                    pts.append((e.x + c * lx * e.sx - sn * ly * e.sy,
                                e.y + sn * lx * e.sx + c * ly * e.sy))
        else:
            r = float(np.clip(0.13 * np.sqrt(max(e.sx * e.sy, 1.0)) + 8.0, 9.0, 52.0))
            a2 = e.sx * e.sy / (r * r)
            n = int(np.clip(2.4 * a2, 6, 56))
            for _ in range(n):
                aa = jitter(0, 2 * np.pi)
                rad = np.sqrt(1.05 * jitter())                  # sqrt keeps density flat
                pts.append((e.x + np.cos(aa) * rad * e.sx, e.y + np.sin(aa) * rad * e.sy))
        return pts, r

    def _fly_update(self, dt):
        self.fly_stamp = []
        if not self.fly_on:
            return
        speed = FLY_SPEED * self.fly_speed
        dab_t = self.fly_x + FLY_TO_BRUSH[0], self.fly_y + FLY_TO_BRUSH[1]
        if self.fly_st == "rest":                      # hover quietly in a corner, "listening"
            hx, hy = 0.84 * self.w, 0.11 * self.h
            tx = hx + 0.07 * self.w * np.cos(self.t * 0.5)
            ty = hy + 0.05 * self.h * np.sin(self.t * 0.9)
            d = np.hypot(tx - self.fly_x, ty - self.fly_y)
            if d > 4.0:
                u = min(speed * 0.3 * dt, d)
                self.fly_x += (tx - self.fly_x) / d * u
                self.fly_y += (ty - self.fly_y) / d * u
                self._fly_turn(np.arctan2(ty - self.fly_y, tx - self.fly_x), 4.0 * dt)
            return
        if self.fly_i >= len(self.fly_plan):           # a recompose can only lengthen the plan
            self.fly_st = "rest"
            return
        e = self.fly_plan[self.fly_i]
        while self.fly_prog.get(e, 0.0) >= 1.0:        # already painted (e.g. after a recompose)
            self.fly_i += 1
            if self.fly_i >= len(self.fly_plan):
                self.fly_st = "rest"
                return
            e = self.fly_plan[self.fly_i]
        if self.fly_di >= len(self.fly_dabs):
            if not self.fly_dabs:                      # fresh element: plan its strokes
                self.fly_dabs, self.fly_r = self._fly_dabs(e)
                if not self.fly_dabs:
                    self.fly_prog[e] = 1.0
                    self.fly_i += 1
                    self.fly_st = "fly"
                    return
                self.fly_di = 0
            else:                                      # last dab done: element is painted
                self.fly_prog[e] = 1.0
                self.fly_i += 1
                self.fly_dabs, self.fly_di = [], 0
                self.fly_st = "fly"
                return
        tx, ty = self.fly_dabs[self.fly_di]
        bx, by = tx - FLY_TO_BRUSH[0], ty - FLY_TO_BRUSH[1]     # body stops where the tip touches
        d = np.hypot(bx - self.fly_x, by - self.fly_y)
        if d > max(2.0, speed * dt):
            u = min(speed * dt, d)
            self._fly_turn(np.arctan2(by - self.fly_y, bx - self.fly_x), 6.0 * dt)
            self.fly_x += (bx - self.fly_x) / d * u
            self.fly_y += (by - self.fly_y) / d * u
            return
        self.fly_x, self.fly_y = bx, by
        self.fly_stamp.append((tx, ty, self.fly_r, 0.9))
        n = len(self.fly_dabs)
        prev = self.fly_prog.get(e, 0.0)
        self.fly_prog[e] = max(prev, (self.fly_di + 1) / n)
        self.fly_di += 1
        self.fly_st = "work"

    def _fly_turn(self, ta, u):
        d = (ta - self.fly_ang + np.pi) % (2 * np.pi) - np.pi
        self.fly_ang += d * min(1.0, u)

    def _fly_rows(self, rows, pal):
        """The fly itself: body, wings, brush and its little community badge (brain)."""
        if not self.fly_on:
            return
        t = self.t
        fx, fy = self.fly_x, self.fly_y
        ang = self.fly_ang
        tx, ty = fx + FLY_TO_BRUSH[0], fy + FLY_TO_BRUSH[1]
        # wings
        wf = np.sin(t * 36.0) * 0.30
        for s in (-1.0, 1.0):
            wa = ang + s * (1.20 + wf)
            rows.append([fx - np.cos(ang) * 4.0, fy - np.sin(ang) * 4.0 - 2.0 * s * np.sin(wa) * 0.2,
                         2.6, 10.5, wa, K_DISC, 4.0, 0, *pal[4], 0.40, 0.6, 0])
        # brush handle and bristle
        L = np.hypot(tx - fx, ty - fy) + 1e-3
        rows.append([(fx + tx) / 2, (fy + ty) / 2, L / 2, 1.8, np.arctan2(ty - fy, tx - fx),
                     K_LINE, 4.0, 0, *pal[3], 0.95, 0.25, 0])
        if self.fly_st == "work" and self.fly_i < len(self.fly_plan):
            e = self.fly_plan[self.fly_i]
            rows.append([tx, ty, 4.4, 4.4, 0, K_DISC, 4.0, 0, *pal[min(e.role, 4)], 0.85, 0.15, 0])
        # body, head, stripes
        rows.append([fx, fy, 5.5, 9.5, ang, K_DISC, 4.0, 0, *pal[3], 0.96, 0.6, 0])
        rows.append([fx + np.cos(ang) * 7.0, fy + np.sin(ang) * 7.0, 3.2, 3.4, ang, K_DISC, 4.0, 0,
                     *pal[3], 1.0, 0.4, 0])
        rows.append([fx - np.cos(ang) * 5.0, fy - np.sin(ang) * 5.0, 2.4, 4.6, ang, K_DISC, 4.0, 0,
                     *pal[2], 0.55, 0.4, 0])
        # brain badge: the community this fly is painting, glowing with its activity
        g = -1
        if self.fly_st == "work" and self.fly_i < len(self.fly_plan):
            e = self.fly_plan[self.fly_i]
            g = e.g if e.g >= 0 else (e.g2 if e.g2 is not None and e.g2 >= 0 else -1)
        if g < 0:
            g = int(np.argmax(self.fast))
        by = fy - 30.0 + 1.6 * np.sin(t * 3.1)
        bc = pal[3] if self.com_col is None else np.clip(self.com_col[g], 0, 1)
        rows.append([fx, by, 6.5, 6.5, 0, K_RING, 4.0, 1.7, *bc, 0.85, 0.3, 0])
        rows.append([fx, by, 3.2, 3.2, 0, K_DISC, 4.0, 0, *bc, 0.15 + 0.75 * float(self.fast[g]), 0.2, 0])

    # ------------------------------------------------------------ accents
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

    # -------------------------------------------------------------- frame
    def _presence(self, e):
        if e.g == -1:
            return float(_smooth(0.0, 1.2, self.t - e.t0))
        if e.g == -2:
            return self.inh_ink
        if e.g == -3:
            return e.ink
        p = self.ink[e.g] if e.g2 is None else np.sqrt(self.ink[e.g] * self.ink[e.g2])
        return max(float(p), e.floor)

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
            if self.fly_on and e in self.fly_set:      # only the fly's brush reveals a fresh element
                p = max(p, self.fly_prog.get(e, 0.0))
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
        self._fly_rows(rows, pal)
        return np.asarray(rows, "f4").reshape(-1, 14)

    def frame(self, lum, rate, dt, t, target):
        ctx = self.ctx
        self.t = t
        lum = np.asarray(lum, np.float64)

        # community activity -> how much each motif is painted in
        act = np.bincount(self.g, lum, self.G) / self.g_n
        self.act_ref = max(self.act_ref * np.exp(-dt / 20.0), act.max(), 1e-9)
        a = np.clip(act / (0.6 * self.act_ref), 0, 1)
        self.fast += (a - self.fast) * (1 - np.exp(-dt / 0.25))
        tgt = _smooth(0.12, 0.65, self.fast)
        self.ink = np.where(tgt > self.ink, self.ink + (tgt - self.ink) * (1 - np.exp(-dt / GROW_TAU)),
                            self.ink * np.exp(-dt / FADE_TAU))

        # inhibition-dominated moments
        tot = lum.sum() + 1e-9
        share = lum[self.inh].sum() / tot if self.inh.any() else 0.0
        self.inh_ema = share if self.inh_ema is None else self.inh_ema + (share - self.inh_ema) * (1 - np.exp(-dt / 10))
        it = float(_smooth(1.0, 1.35, share / (self.inh_ema + 1e-9)))
        self.inh_ink = self.inh_ink + (it - self.inh_ink) * (1 - np.exp(-dt / GROW_TAU)) if it > self.inh_ink \
            else self.inh_ink * np.exp(-dt / FADE_TAU)

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
        rn = rate / (self.rate_ema + 1e-9)
        if rn < 0.6:
            self.quiet += dt
        onset = rn > 1.2 and self.quiet > 0.8
        if rn > 1.2:
            self.quiet = 0.0
        since = t - self.t_compose
        if self.auto_recompose and ((onset and since > RECOMPOSE_MIN) or since > RECOMPOSE_MAX):
            self.recompose()
        self.burst_cd -= dt
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

        # paint (in fly mode the shapes go to a scratch layer, which the fly's
        # reveal mask composites onto fresh paper)
        self._fly_update(dt)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._wash(self.under, 1 - np.exp(-dt / 40.0))          # old compositions slowly sink into the paper
        dest = self.layer if self.fly_on else self.canvas
        dest.use()
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.disable(moderngl.BLEND)
        self._copy(self.under_tex, 1.0)
        ctx.enable(moderngl.BLEND)
        rows = self._instances(PALETTES[self.pal], dt)
        if len(rows):
            self.b_inst.write(rows.tobytes())
            self.v_shape.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(rows))
        ctx.disable(moderngl.BLEND)
        if self.fly_on:
            self.reveal_fbo.use()
            ctx.viewport = (0, 0, self.w, self.h)
            if self.fly_stamp:
                ctx.enable(moderngl.BLEND)
                ctx.blend_func = moderngl.ONE, moderngl.ONE
                self.b_dab.write(np.asarray(self.fly_stamp, "f4").tobytes())
                self.v_dab.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(self.fly_stamp))
            ctx.disable(moderngl.BLEND)
            self.canvas.use()
            ctx.viewport = (0, 0, self.w, self.h)
            self._wash(self.canvas, 1.0)                        # fresh paper under the brush
            self.layer_tex.use(0)
            self.reveal_tex.use(1)
            self.p_mask["u_layer"], self.p_mask["u_mask"] = 0, 1
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.v_mask.render(moderngl.TRIANGLE_STRIP)
            ctx.disable(moderngl.BLEND)

        target.use()
        ctx.viewport = (0, 0, *target.size)
        self.canvas_tex.use(0)
        self.p_post["u_tex"] = 0
        self.v_post.render(moderngl.TRIANGLE_STRIP)
