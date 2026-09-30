import os
import sys
import numpy as np
import moderngl
from painter import Painter


# --------------------------------------------------------------------- config
RES = (1920, 1080)  # FHD
OUTDIR = "out"
CACHE = "connectome_cache.npz"

STEPS_PER_SEC = 60.0  # sim steps consumed per second while playing
SCREEN_DECAY0 = 0.90  # screen-space trail persistence (UP/DOWN at runtime)
DECAY_SPREAD = 0.01  # chromatic spread around the base decay
NEURON_DECAY = 0.88  # per-neuron colour-trail decay (CPU side)
EXPOSURE0 = 1.0
FRAME_FILL = 0.78  # cloud fills this fraction of frame height
SPLAT_SIZE = 15.0  # base glow radius in pixels at camera distance
SPLAT_GAIN = 0.85
EDGE_ALPHA = 0.04
MAX_EDGES = 30_000
ORBIT_SPEED = 0.12  # radians per second
ORBIT_TILT = 0.25
BLOOM_W = 0.65
BLOOM_ITERS = 3
BG = (0.010, 0.008, 0.020)
VIGNETTE = 0.40
SEED = int(os.environ.get("ART_SEED", "0"))  # composition seed: ART_SEED=42 uv run realtime_art.py

# particle flow-field (abstract mode)
PARTICLES = int(os.environ.get("RT_PARTICLES", "60000"))
P_GAIN = 0.16  # per-particle additive brightness
P_SIZE = 3.2  # point size at camera distance
SWIRL, PULL, NOISE_F = 1.35, 0.30, 0.30

# --- modern art + fly ---
PAPER = (0.94, 0.91, 0.85)
PAINT_FADE = 0.0008  # painting builds up instead of fogging out
MAX_AGENTS = 600
SPAWN_BASE, SPAWN_GAIN = 2, 10  # new brushes per frame (+ per unit of activity)
SPAWN_POWER = 1.0  # >1 favours hottest neurons (centre), <1 spreads out
AGENT_SPEED = 220.0  # px / s
AGENT_LIFE = (1.5, 4.0)  # s  -> strokes ~300-900 px long
SUBSTEPS = 2  # dabs per agent per frame
FLOW_SCALE = 0.0018  # lower = bigger, calmer swirls
INH_ERASE = 0.5  # fraction of inhibitory brushes that paint with paper colour
FLOW_TIME = 0.03  # how fast the field itself drifts
SHAPE_COOLDOWN = 2.0  # s between geometric accents
MAX_STROKES = MAX_AGENTS * SUBSTEPS


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]


PALETTES = {
    k: np.array([_hex(c) for c in v], np.float32)
    for k, v in {
        "bauhaus": ["#E63946", "#F4C430", "#1D3557", "#111111", "#F1FAEE"],
        "matisse": ["#E4572E", "#29335C", "#F3A712", "#A8C686", "#669BBC"],
        "rothko": ["#8C1C13", "#BF4342", "#E7D7C1", "#A78A7F", "#2B2D42"],
        "miro": ["#D62828", "#003049", "#FCBF49", "#111111", "#EAE2B7"],
    }.items()
}

# ---------------------------------------------------------------- data loading
spikes = np.load(f"{OUTDIR}/spikes.npy").astype(np.float32)  # [T, N]
pos_raw = np.load(f"{OUTDIR}/positions.npy").astype(np.float32)  # [N, 3]
T, N = spikes.shape
print(f"[rt] {N} neurons, {T} sim steps")

_edges = None
_signs = np.ones(N, np.float32)
colors = None
COMMUNITY = None
if os.path.exists(f"{OUTDIR}/render_assets.npz"):
    _ra = np.load(f"{OUTDIR}/render_assets.npz")
    if "community" in _ra.files:
        COMMUNITY = _ra["community"]
    colors = _ra["colors"].astype(np.float32)
    _edges = (
        _ra["e_pre"].astype(np.int32),
        _ra["e_post"].astype(np.int32),
        _ra["e_str"].astype(np.float32),
    )
    _signs = _ra["signs"].astype(np.float32)
    print(f"[rt] render_assets: colors + {len(_edges[0])} edges")

W_sym = None
z = None
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    import scipy.sparse as sp

    if len(z["signs"]) != N:
        print(
            f"[rt] cache is for {len(z['signs'])} neurons but spikes have {N} - ignoring cache"
        )
        z = None
if z is not None and os.path.exists(CACHE):
    _w = np.abs(z["vals"]).astype(np.float32)
    Wc = sp.csr_matrix((np.log1p(_w), (z["rows"], z["cols"])), shape=(N, N))
    W_sym = (Wc + Wc.T).tocsr()
    if _edges is None:
        k = min(MAX_EDGES, Wc.nnz)
        co = Wc.tocoo()
        top = np.argsort(-co.data)[:k]
        _edges = (
            co.col[top].astype(np.int32),
            co.row[top].astype(np.int32),
            co.data[top],
        )
        print(f"[rt] edges from cache: {k}")

if colors is None:  # fallback palette: golden-angle hue wheel
    import colorsys

    hues = (np.arange(N) * 0.381966) % 1.0
    colors = np.array(
        [colorsys.hsv_to_rgb(h, 0.85, 1.0) for h in hues], dtype=np.float32
    )
    print("[rt] no render_assets.npz - using golden-angle fallback colours")

PAL_IDX = np.unique(colors.round(3), axis=0, return_inverse=True)[1].ravel()


def repair_positions(p, Wcsr, seed=SEED):
    """Neurons with no soma location arrive stacked on one point (renders as a
    single white blob). Spread them by harmonic interpolation on the graph:
    each sits at the weighted mean of its neighbours, iterated."""
    p = p.copy()
    _, inv, cnt = np.unique(
        np.round(p, 1), axis=0, return_inverse=True, return_counts=True
    )
    miss = cnt[inv] > max(3, int(0.02 * len(p)))
    n = int(miss.sum())
    if n == 0 or (~miss).sum() < 3 or Wcsr is None:
        return p, 0
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
    return p, n


pos3, _nrep = repair_positions(pos_raw, W_sym)
if _nrep:
    print(f"[rt] re-laid out {_nrep} stacked neurons via graph harmonic fill")

pos3 = pos3 - pos3.mean(0)
ar = np.ptp(pos3, axis=0)
fa = int(np.argmin(ar))
if ar[fa] < 0.3 * ar.max():
    pos3[:, fa] *= 2.5  # un-flatten a pancake neuropil
pos3 /= np.std(pos3) + 1e-6
CAM = 3.0 * float(np.max(np.linalg.norm(pos3, axis=1)))

# auto-frame: FOV such that the cloud fills FRAME_FILL of the frame height
_rmax = 0.0
for _a in (0.0, np.pi / 3, 2 * np.pi / 3):
    ca, sa = np.cos(_a), np.sin(_a)
    xr = pos3[:, 0] * ca - pos3[:, 2] * sa
    zr = pos3[:, 0] * sa + pos3[:, 2] * ca
    ct, st = np.cos(ORBIT_TILT), np.sin(ORBIT_TILT)
    yt = pos3[:, 1] * ct - zr * st
    zt = pos3[:, 1] * st + zr * ct
    zc = np.maximum(zt + CAM, CAM * 0.1)
    _rmax = max(_rmax, float(np.percentile(np.sqrt(xr**2 + yt**2) / zc, 99.0)))
FOV = (FRAME_FILL * RES[1] / 2) / max(_rmax, 1e-6)
ZMIN, ZMAX = CAM - 1.2 * _rmax * CAM, CAM + 1.2 * _rmax * CAM
print(f"[rt] auto-framed FOV={FOV:.0f}px  cam={CAM:.1f}")

# ------------------------------------------------------------------- shaders
VS_POINT = """
#version 330
uniform vec2 u_res; uniform float u_angle, u_tilt, u_cam, u_fov, u_size;
in vec2 in_quad; in vec3 in_pos; in vec3 in_col;
out vec2 v_uv; out vec3 v_col; out float v_zn;
void main(){
    vec3 p = in_pos;
    float ca=cos(u_angle), sa=sin(u_angle);
    float xr=p.x*ca - p.z*sa, zr=p.x*sa + p.z*ca;
    float ct=cos(u_tilt), st=sin(u_tilt);
    float yt=p.y*ct - zr*st, zt=p.y*st + zr*ct;
    float zc=max(zt+u_cam, u_cam*0.1);
    vec2 c = vec2(u_fov*xr/zc, u_fov*yt/zc);
    float sz = clamp(u_size*u_cam/zc, 3.0, 60.0);
    vec2 pix = c + in_quad*sz;
    gl_Position = vec4(pix.x/(u_res.x*0.5), -pix.y/(u_res.y*0.5), 0.0, 1.0);
    v_uv = in_quad; v_col = in_col;
    v_zn = clamp((zc - u_cam*0.55)/(u_cam*0.9), 0.0, 1.0);
}"""

FS_POINT = """
#version 330
in vec2 v_uv; in vec3 v_col; in float v_zn;
out vec4 f;
void main(){
    float d = length(v_uv);
    float g = exp(-d*d*6.0) * smoothstep(1.0, 0.82, d);
    vec3 c = v_col;
    float grey = dot(c, vec3(0.2126,0.7152,0.0722));
    c = mix(c, vec3(grey), 0.35*v_zn);         // atmospheric desaturation
    c.b += 0.06*v_zn;                          // cool shift with depth
    f = vec4(c * g * (1.0 - 0.45*v_zn), 1.0);
}"""

VS_EDGE = """
#version 330
uniform vec2 u_res; uniform float u_angle, u_tilt, u_cam, u_fov;
in vec3 in_pos; in vec3 in_col;
out vec3 v_col;
void main(){
    vec3 p = in_pos;
    float ca=cos(u_angle), sa=sin(u_angle);
    float xr=p.x*ca - p.z*sa, zr=p.x*sa + p.z*ca;
    float ct=cos(u_tilt), st=sin(u_tilt);
    float yt=p.y*ct - zr*st, zt=p.y*st + zr*ct;
    float zc=max(zt+u_cam, u_cam*0.1);
    vec2 c = vec2(u_fov*xr/zc, u_fov*yt/zc);
    gl_Position = vec4(c.x/(u_res.x*0.5), -c.y/(u_res.y*0.5), 0.0, 1.0);
    v_col = in_col;
}"""

FS_EDGE = """
#version 330
uniform float u_alpha;
in vec3 v_col; out vec4 f;
void main(){ f = vec4(v_col * u_alpha, 1.0); }"""

VS_QUAD = """
#version 330
in vec2 in_quad; out vec2 v_uv;
void main(){ v_uv = in_quad*0.5+0.5; gl_Position = vec4(in_quad, 0.0, 1.0); }"""

FS_DECAY = """
#version 330
uniform sampler2D u_tex; uniform vec3 u_decay;
in vec2 v_uv; out vec4 f;
void main(){ f = vec4(texture(u_tex, v_uv).rgb * u_decay, 1.0); }"""

FS_BLUR = """
#version 330
uniform sampler2D u_tex; uniform vec2 u_dir;
in vec2 v_uv; out vec4 f;
void main(){
    vec2 px = u_dir / vec2(textureSize(u_tex, 0));
    vec3 s = texture(u_tex, v_uv).rgb * 0.227;
    s += (texture(u_tex, v_uv + px*1.385).rgb + texture(u_tex, v_uv - px*1.385).rgb) * 0.316;
    s += (texture(u_tex, v_uv + px*3.231).rgb + texture(u_tex, v_uv - px*3.231).rgb) * 0.070;
    f = vec4(s, 1.0);
}"""

FS_COMPOSITE = """
#version 330
uniform sampler2D u_scene; uniform sampler2D u_bloom;
uniform float u_exposure, u_bloom_w, u_vignette, u_aspect;
uniform int u_kfold;
uniform vec3 u_bg;
in vec2 v_uv; out vec4 f;
vec3 aces(vec3 x){ return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), 0.0, 1.0); }
void main(){
    vec2 uv = v_uv;
    if (u_kfold > 0){
        vec2 q = uv - 0.5; q.x *= u_aspect;
        float ang = atan(q.y, q.x), r = length(q);
        float seg = 6.28318530 / float(u_kfold);
        ang = abs(mod(ang, seg) - 0.5*seg);
        q = vec2(cos(ang), sin(ang)) * r; q.x /= u_aspect;
        uv = q + 0.5;
    }
    vec3 hdr = texture(u_scene, uv).rgb + u_bloom_w * texture(u_bloom, uv).rgb;
    vec3 c = aces(hdr * u_exposure);
    c = u_bg + c * (1.0 - u_bg);                       // bg composited AFTER tonemap
    vec2 q = v_uv*2.0-1.0;
    c *= clamp(1.0 - u_vignette*dot(q,q), 0.0, 1.0);
    // triangular dither: smooth HDR glows quantised straight to 8 bits band
    // visibly; +-1 LSB of noise trades the bands for invisible grain
    float dn = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    c += (dn - 0.5) / 255.0;
    f = vec4(c, 1.0);
}"""


VS_PSIM = """
#version 330
uniform float u_dt, u_time, u_rate, u_bound, u_swirl, u_pull, u_noise;
uniform int u_n;
uniform vec4 u_attr[64];
uniform vec4 u_acol[64];
in vec3 in_pos; in vec3 in_vel; in vec3 in_col; in vec2 in_meta;
out vec3 out_pos; out vec3 out_vel; out vec3 out_col; out vec2 out_meta;
vec3 hash3(float n){ return fract(sin(vec3(n, n+1.37, n+2.71)) * 43758.5453); }
void main(){
    vec3 p = in_pos, v = in_vel, c = in_col;
    float life = in_meta.x, seed = in_meta.y;
    vec3 acc = vec3(0.0);
    for (int i = 0; i < u_n; i++){
        vec3 d = u_attr[i].xyz - p;
        float r2 = dot(d, d) + 0.03;
        vec3 dir = d * inversesqrt(r2);
        vec3 tang = cross(dir, vec3(0.16, 0.98, 0.12));
        acc += (u_swirl * tang + u_pull * dir) * (u_attr[i].w / r2);
    }
    acc += u_noise * vec3(sin(p.y*2.3 + u_time*0.9),
                          sin(p.z*1.9 + u_time*1.23),
                          sin(p.x*2.1 + u_time*0.71));
    v = v * 0.90 + acc * u_dt;
    p += v * u_dt;
    life -= u_dt * (0.25 + 0.8 * u_rate);
    if (life <= 0.0 || dot(p, p) > u_bound * u_bound){
        vec3 h = hash3(seed * 719.0 + u_time * 13.7);
        if (u_n > 0){
            int k = int(min(h.x * float(u_n), float(u_n) - 0.5));
            p = u_attr[k].xyz + (h * 2.0 - 1.0) * 0.35;
            c = u_acol[k].rgb;
        } else {
            p = (h * 2.0 - 1.0) * u_bound * 0.4;
        }
        v = vec3(0.0);
        life = 0.6 + h.z * 1.8;
        seed = fract(seed + 0.6180339887);
    }
    out_pos = p; out_vel = v; out_col = c; out_meta = vec2(life, seed);
}"""

VS_PART = """
#version 330
uniform vec2 u_res; uniform float u_angle, u_tilt, u_cam, u_fov, u_psize;
in vec3 in_pos; in vec3 in_col;
out vec3 v_col;
void main(){
    vec3 p = in_pos;
    float ca=cos(u_angle), sa=sin(u_angle);
    float xr=p.x*ca - p.z*sa, zr=p.x*sa + p.z*ca;
    float ct=cos(u_tilt), st=sin(u_tilt);
    float yt=p.y*ct - zr*st, zt=p.y*st + zr*ct;
    float zc=max(zt+u_cam, u_cam*0.1);
    vec2 c = vec2(u_fov*xr/zc, u_fov*yt/zc);
    gl_Position = vec4(c.x/(u_res.x*0.5), -c.y/(u_res.y*0.5), 0.0, 1.0);
    gl_PointSize = clamp(u_psize*u_cam/zc, 1.0, 5.0);
    float zn = clamp((zc - u_cam*0.55)/(u_cam*0.9), 0.0, 1.0);
    v_col = in_col * (1.0 - 0.5*zn);
}"""

FS_PART = """
#version 330
uniform float u_pgain;
in vec3 v_col; out vec4 f;
void main(){
    float a = smoothstep(0.5, 0.1, length(gl_PointCoord - 0.5));
    f = vec4(v_col * a * u_pgain, 1.0);
}"""

VS_STROKE = """
#version 330
uniform vec2 u_res;
in vec2 in_quad; in vec4 in_s; in vec4 in_c; in vec2 in_m;   // m: seed, load
out vec2 v_uv; out vec3 v_col; flat out float v_seed; out float v_load;
void main(){
    vec2 d = vec2(cos(in_s.z), sin(in_s.z)), n = vec2(-d.y, d.x);
    vec2 p = in_s.xy + d*in_quad.x*in_s.w + n*in_quad.y*in_c.w;
    gl_Position = vec4(p/u_res*2.0 - 1.0, 0, 1);
    v_uv = in_quad; v_col = in_c.rgb; v_seed = in_m.x; v_load = in_m.y;
}"""

FS_STROKE = """
#version 330
in vec2 v_uv; in vec3 v_col; flat in float v_seed; in float v_load; out vec4 f;
float h(float n){ return fract(sin(n)*43758.5453); }
void main(){
    float lane    = floor((v_uv.y*0.5 + 0.5)*12.0);        // bristle lanes run along the stroke
    float bristle = 0.35 + 0.65*h(lane + v_seed*7.13);
    float r       = h(lane*3.1 + v_seed);
    float dry     = smoothstep(r - 0.15, r + 0.15, v_load); // lanes fade out as the brush dries
    float side    = smoothstep(1.0, 0.7, abs(v_uv.y));
    float ends    = smoothstep(1.0, 0.3, abs(v_uv.x));       // soft, so overlapping dabs fuse
    f = vec4(v_col, 0.6*bristle*mix(0.25, 1.0, dry)*side*ends);
}"""

FS_SHAPE = """
#version 330
uniform int u_kind;
in vec2 v_uv; in vec3 v_col; flat in float v_seed; out vec4 f;
void main(){
    float r = length(v_uv), a;
    if      (u_kind == 0) a = smoothstep(1.0, 0.97, r);                               // disc
    else if (u_kind == 1) a = smoothstep(0.09, 0.06, abs(r - 0.85));                  // ring
    else if (u_kind == 2) a = smoothstep(1.0, 0.98, abs(v_uv.x)) * smoothstep(1.0, 0.9, abs(v_uv.y)); // bar
    else a = 0.22*(1.0 - smoothstep(0.6, 1.0, abs(v_uv.x)))*(1.0 - smoothstep(0.5, 1.0, abs(v_uv.y))); // Rothko field
    f = vec4(v_col, a);
}"""

FS_WASH = """
#version 330
uniform vec3 u_paper; uniform float u_fade; out vec4 f;
void main(){ f = vec4(u_paper, u_fade); }"""

FS_CANVAS = """
#version 330
uniform sampler2D u_tex; uniform vec2 u_res;
in vec2 v_uv; out vec4 f;
float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898, 78.233)))*43758.5453); }
void main(){
    vec2 px = 2.0/u_res;
    vec3 c = texture(u_tex, v_uv).rgb;
    vec3 b = 0.25*(texture(u_tex, v_uv+vec2(px.x,0)).rgb + texture(u_tex, v_uv-vec2(px.x,0)).rgb
                 + texture(u_tex, v_uv+vec2(0,px.y)).rgb + texture(u_tex, v_uv-vec2(0,px.y)).rgb);
    c *= max(1.0 - 0.2*length(c - b), 0.0);
    vec2 g = gl_FragCoord.xy;
    c *= 0.955 + 0.045*(0.5 + 0.25*(sin(g.x*1.3) + sin(g.y*1.3)));  // canvas weave
    c += (h(g) - 0.5)*0.035;                               // grain
    f = vec4(c, 1.0);
}"""


# ---------------------------------------------------------------- GL pipeline
def build_pipeline(ctx, w, h):
    pipe = {}
    pipe["p_point"] = ctx.program(vertex_shader=VS_POINT, fragment_shader=FS_POINT)
    pipe["p_edge"] = ctx.program(vertex_shader=VS_EDGE, fragment_shader=FS_EDGE)
    pipe["p_decay"] = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_DECAY)
    pipe["p_blur"] = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_BLUR)
    pipe["p_comp"] = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_COMPOSITE)

    quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32))
    pipe["quad"] = quad

    pipe["b_pos"] = ctx.buffer(pos3.astype("f4").tobytes())
    pipe["b_col"] = ctx.buffer(np.zeros((N, 3), np.float32).tobytes(), dynamic=True)
    pipe["vao_pts"] = ctx.vertex_array(
        pipe["p_point"],
        [
            (quad, "2f", "in_quad"),
            (pipe["b_pos"], "3f/i", "in_pos"),
            (pipe["b_col"], "3f/i", "in_col"),
        ],
    )

    if _edges is not None:
        ep, eo, es = _edges
        vp = np.empty((len(ep) * 2, 3), np.float32)
        vp[0::2], vp[1::2] = pos3[ep], pos3[eo]
        strength = (es / (es.max() + 1e-9))[:, None].astype(np.float32)
        vc = np.empty((len(ep) * 2, 3), np.float32)
        vc[0::2] = colors[ep] * strength
        vc[1::2] = colors[eo] * strength * 0.3
        pipe["vao_edges"] = ctx.vertex_array(
            pipe["p_edge"],
            [
                (ctx.buffer(vp.tobytes()), "3f", "in_pos"),
                (ctx.buffer(vc.tobytes()), "3f", "in_col"),
            ],
        )
        pipe["n_edge_v"] = len(ep) * 2
    else:
        pipe["vao_edges"], pipe["n_edge_v"] = None, 0

    # particle flow field: ping-pong buffers updated by transform feedback
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    pipe["p_psim"] = ctx.program(
        vertex_shader=VS_PSIM, varyings=["out_pos", "out_vel", "out_col", "out_meta"]
    )
    pipe["p_part"] = ctx.program(vertex_shader=VS_PART, fragment_shader=FS_PART)
    R = float(np.max(np.linalg.norm(pos3, axis=1)))
    rs = np.random.RandomState(SEED + 9)
    pdat = np.zeros((PARTICLES, 11), np.float32)
    pdat[:, 0:3] = rs.normal(0, 0.5 * R, (PARTICLES, 3))
    pdat[:, 6:9] = 0.25
    pdat[:, 9] = rs.uniform(0.0, 2.0, PARTICLES)  # staggered lifetimes
    pdat[:, 10] = rs.uniform(0.0, 1.0, PARTICLES)  # per-particle seed
    pipe["pbuf"] = [ctx.buffer(pdat.tobytes()), ctx.buffer(pdat.tobytes())]
    pipe["vao_psim"] = [
        ctx.vertex_array(
            pipe["p_psim"],
            [(b, "3f 3f 3f 2f", "in_pos", "in_vel", "in_col", "in_meta")],
        )
        for b in pipe["pbuf"]
    ]
    pipe["vao_pdraw"] = [
        ctx.vertex_array(pipe["p_part"], [(b, "3f 12x 3f 8x", "in_pos", "in_col")])
        for b in pipe["pbuf"]
    ]
    pipe["np"] = PARTICLES
    pipe["p_psim"]["u_bound"] = 2.2 * R
    pipe["p_psim"]["u_swirl"], pipe["p_psim"]["u_pull"], pipe["p_psim"]["u_noise"] = (
        SWIRL,
        PULL,
        NOISE_F,
    )

    def fbo(sw, sh):
        t = ctx.texture((sw, sh), 4, dtype="f2")
        t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return ctx.framebuffer([t]), t

    pipe["accum"] = [fbo(w, h), fbo(w, h)]  # ping-pong HDR accumulation
    bw, bh = w // 2, h // 2
    pipe["blur"] = [fbo(bw, bh), fbo(bw, bh)]  # ping-pong bloom
    pipe["vao_decay"] = ctx.vertex_array(pipe["p_decay"], [(quad, "2f", "in_quad")])
    pipe["vao_blur"] = ctx.vertex_array(pipe["p_blur"], [(quad, "2f", "in_quad")])
    pipe["vao_comp"] = ctx.vertex_array(pipe["p_comp"], [(quad, "2f", "in_quad")])
    for f, t in pipe["accum"] + pipe["blur"]:
        f.use()
        ctx.clear(0, 0, 0, 1)
    return pipe


def build_paint(ctx, pipe, w, h):
    tex = ctx.texture((w, h), 4, dtype="f2")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    pipe["paint"] = (ctx.framebuffer([tex]), tex)
    pipe["paint"][0].use()
    ctx.clear(*PAPER, 1.0)
    q = pipe["quad"]
    pipe["p_stroke"] = ctx.program(vertex_shader=VS_STROKE, fragment_shader=FS_STROKE)
    pipe["p_shape"] = ctx.program(vertex_shader=VS_STROKE, fragment_shader=FS_SHAPE)
    pipe["p_wash"] = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_WASH)
    pipe["p_canvas"] = ctx.program(vertex_shader=VS_QUAD, fragment_shader=FS_CANVAS)
    pipe["b_strk"] = ctx.buffer(reserve=MAX_STROKES * 40, dynamic=True)
    pipe["b_shp"] = ctx.buffer(reserve=40, dynamic=True)
    inst = lambda p, b: ctx.vertex_array(
        p, [(q, "2f", "in_quad"), (b, "4f 4f 2f/i", "in_s", "in_c", "in_m")]
    )
    pipe["vao_strk"] = inst(pipe["p_stroke"], pipe["b_strk"])
    pipe["vao_shp"] = inst(pipe["p_shape"], pipe["b_shp"])
    pipe["vao_wash"] = ctx.vertex_array(pipe["p_wash"], [(q, "2f", "in_quad")])
    pipe["vao_canv"] = ctx.vertex_array(pipe["p_canvas"], [(q, "2f", "in_quad")])
    for k in ("p_stroke", "p_shape", "p_canvas"):
        pipe[k]["u_res"] = (w, h)


def project2d(p, angle, w, h):
    ca, sa = np.cos(angle), np.sin(angle)
    xr, zr = p[:, 0] * ca - p[:, 2] * sa, p[:, 0] * sa + p[:, 2] * ca
    ct, s_ = np.cos(ORBIT_TILT), np.sin(ORBIT_TILT)
    yt, zt = p[:, 1] * ct - zr * s_, p[:, 1] * s_ + zr * ct
    zc = np.maximum(zt + CAM, CAM * 0.1)
    k = FOV * h / RES[1]
    return (
        w / 2 + k * xr / zc,
        h / 2 - k * yt / zc,
    )  # GL pixel coords, matches the glow view


class Brushes:
    def __init__(self, n, rng):
        self.p = np.zeros((n, 2), np.float32)
        self.col = np.zeros((n, 3), np.float32)
        self.w = np.zeros(n, np.float32)
        self.age = np.zeros(n, np.float32)
        self.life = np.ones(n, np.float32)
        self.alive = np.zeros(n, bool)
        self.seed = rng.uniform(0, 100, n).astype(np.float32)
        self.bias = rng.normal(0, 0.35, n).astype(np.float32)  # radians of steady curl


def flow(x, y, t):
    """Curl of a smooth scalar field: divergence-free, so neighbouring strokes
    swirl side by side instead of crossing at random angles."""
    f = FLOW_SCALE
    a, b = x * f + t * FLOW_TIME, y * f - t * FLOW_TIME * 0.7
    c = (x + y) * f * 0.6 + t * FLOW_TIME * 0.5
    dpx = f * (
        np.cos(a) * np.cos(b) + 0.3 * np.cos(c)
    )  # d/dx of sin(a)cos(b) + 0.5 sin(c)
    dpy = f * (-np.sin(a) * np.sin(b) + 0.3 * np.cos(c))  # d/dy
    vx, vy = dpy, -dpx
    n = np.hypot(vx, vy) + 1e-9
    return vx / n, vy / n


def spawn_brushes(st, lum, w, h, pal):
    B = st.br
    free = np.flatnonzero(~B.alive)[: int(SPAWN_BASE + SPAWN_GAIN * min(st.rate, 2.0))]
    if not len(free) or lum.max() <= 0:
        return
    pr = lum**SPAWN_POWER
    pr /= pr.sum()
    idx = st.rng.choice(N, len(free), p=pr)
    B.p[free] = np.column_stack([st.lay[0][idx], st.lay[1][idx]]) + st.rng.normal(
        0, 25, (len(free), 2)
    )
    col = pal[PAL_IDX[idx] % len(pal)].copy()
    erase = (_signs[idx] < 0) & (st.rng.rand(len(free)) < INH_ERASE)
    col[erase] = PAPER  # inhibition carves negative space
    B.col[free] = col
    B.w[free] = 2 + 7 * (lum[idx] / lum.max())
    B.w[free[col.sum(1) < 0.4]] *= 0.35  # black = thin accents
    B.age[free] = 0
    B.life[free] = st.rng.uniform(*AGENT_LIFE, len(free))
    B.alive[free] = True


def step_brushes(st, w, h):
    B = st.br
    i = np.flatnonzero(B.alive)
    if not len(i):
        return np.zeros((0, 10), "f4")
    dts = st.dt / SUBSTEPS
    stp = AGENT_SPEED * dts
    out = []
    for _ in range(SUBSTEPS):
        u_prev = np.clip(B.age[i] / B.life[i], 0, 1)
        fx, fy = flow(B.p[i, 0], B.p[i, 1], st.time)
        cb, sb = np.cos(B.bias[i] * u_prev), np.sin(B.bias[i] * u_prev)
        dx, dy = fx * cb - fy * sb, fx * sb + fy * cb
        B.p[i, 0] += dx * stp
        B.p[i, 1] += dy * stp
        B.age[i] += dts
        u = np.clip(B.age[i] / B.life[i], 0, 1)
        taper = np.sqrt(np.sin(np.pi * u))  # swell in, thin out
        out.append(
            np.column_stack(
                [
                    B.p[i, 0],
                    B.p[i, 1],
                    np.arctan2(dy, dx),
                    np.full(len(i), stp * 0.9 + 1.0),
                    B.col[i],
                    B.w[i] * taper,
                    B.seed[i],
                    1.0 - 0.7 * u,
                ]
            )
        )
    x, y = B.p[i, 0], B.p[i, 1]
    B.alive[
        i[(B.age[i] >= B.life[i]) | (x < -50) | (x > w + 50) | (y < -50) | (y > h + 50)]
    ] = False
    return np.concatenate(out).astype("f4")


def maybe_shape(st, lum, w, h, pal):
    st.rate_ema = 0.98 * st.rate_ema + 0.02 * st.rate
    st.shape_cd = getattr(st, "shape_cd", 0.0) - st.dt
    if st.shape_cd > 0 or st.rate < 1.6 * st.rate_ema or st.rng.rand() > 0.25:
        return None, 0
    st.shape_cd = SHAPE_COOLDOWN
    j = int(np.argmax(lum))
    x, y = [st.lay[0][j]], [st.lay[1][j]]
    kind = int(st.rng.choice(4, p=[0.4, 0.3, 0.25, 0.05]))  # Rothko fields rare
    L = st.rng.uniform(25, 140) * (3 if kind == 3 else 1)
    W = L * (0.6 if kind == 3 else 0.06 if kind == 2 else 1.0)
    return np.array(
        [
            [
                x[0],
                y[0],
                st.rng.uniform(0, np.pi),
                L,
                *pal[st.rng.randint(len(pal))],
                W,
                0.0,
                1.0,
            ]
        ],
        "f4",
    ), kind


def canvas_layout(w, h, margin=0.08, mix=0.6):
    x, y = project2d(pos3, 0.0, w, h)
    rank = lambda v: (np.argsort(np.argsort(v)) + 0.5) / N
    lin = lambda v: (v - v.min()) / (np.ptp(v) + 1e-9)
    u = mix * rank(x) + (1 - mix) * lin(x)
    v = mix * rank(y) + (1 - mix) * lin(y)
    return (margin + u * (1 - 2 * margin)) * w, (margin + v * (1 - 2 * margin)) * h


def flow(x, y, t):
    vx, vy = np.zeros_like(x), np.zeros_like(y)
    for f, amp, sp in ((FLOW_SCALE, 1.0, 1.0), (FLOW_SCALE * 2.7, 0.45, -1.6)):
        a, b = x * f + t * FLOW_TIME * sp, y * f - t * FLOW_TIME * 0.7 * sp
        vx += amp * (-np.sin(a) * np.sin(b))  # d/dy of sin(a)cos(b)
        vy -= amp * (np.cos(a) * np.cos(b))  # -d/dx
    n = np.hypot(vx, vy) + 1e-9
    return vx / n, vy / n


class State:
    def __init__(self, w, h):
        self.t = 0.0
        self.playing = True
        self.orbit = True
        self.angle = 0.0
        self.edges_on = True
        self.decay = SCREEN_DECAY0
        self.exposure = EXPOSURE0
        self.trail = np.zeros((N, 3), np.float32)
        self.col_s = np.zeros((N, 3), np.float32)
        self.gain = None
        self.ping = 0
        self.mode = 3  # 1 neurons, 2 particles, 3 both
        self.kfold = 0  # kaleidoscope folds (0 = off)
        self.pping = 0
        self.time = 0.0
        self.dt = 1.0 / 60.0
        self.rate = 0.0
        self.nk = 0
        self.attr = np.zeros((64, 4), np.float32)
        self.acol = np.zeros((64, 4), np.float32)


def neuron_decay_vec():
    d = np.array(
        [NEURON_DECAY + DECAY_SPREAD, NEURON_DECAY, NEURON_DECAY - DECAY_SPREAD],
        np.float32,
    )
    return d, (1.0 - d) / (1.0 - d).max()  # hue-neutral emission (see offline notes)


ND, EMIT = neuron_decay_vec()


def advance(st, steps):
    for _ in range(int(steps)):
        st.trail *= ND
        spk = spikes[int(st.t) % T] > 0.5
        st.trail[spk] += colors[spk] * EMIT
        st.t += 1


def upload_colors(st, pipe):
    """Two choppiness sources fixed here. (1) Normalising by a per-frame p99
    makes global brightness pump as spikes shift the percentile - the gain is
    now an EMA. (2) A spike used to add its full colour in one frame (square-
    wave onset = strobing); uploaded colours now follow an attack/release
    filter, so lit regions breathe instead of flickering."""
    lum = st.trail @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    p99 = float(np.percentile(lum, 99.0)) if (lum > 0).any() else 1.0
    st.gain = p99 if st.gain is None else 0.92 * st.gain + 0.08 * p99
    target = np.clip(st.trail * (SPLAT_GAIN / max(st.gain, 1e-6)), 0, 2.2)
    rising = target > st.col_s
    st.col_s = np.where(
        rising, 0.55 * st.col_s + 0.45 * target, 0.80 * st.col_s + 0.20 * target
    ).astype(np.float32)
    pipe["b_col"].write(st.col_s.astype("f4").tobytes())
    return lum


def render_paint(ctx, pipe, st, lum, w, h, target):
    pal = PALETTES[st.pal_name]
    fbo, tex = pipe["paint"]
    fbo.use()
    ctx.viewport = (0, 0, w, h)
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (
        moderngl.SRC_ALPHA,
        moderngl.ONE_MINUS_SRC_ALPHA,
    )  # paint layers, not glow
    pipe["p_wash"]["u_paper"], pipe["p_wash"]["u_fade"] = PAPER, PAINT_FADE
    pipe["vao_wash"].render(moderngl.TRIANGLE_STRIP)
    spawn_brushes(st, lum, w, h, pal)
    s = step_brushes(st, w, h)
    if len(s):
        pipe["b_strk"].write(s.tobytes())
        pipe["vao_strk"].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(s))
    shp, kind = maybe_shape(st, lum, w, h, pal)
    if shp is not None:
        pipe["b_shp"].write(shp.tobytes())
        pipe["p_shape"]["u_kind"] = kind
        pipe["vao_shp"].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=1)
    ctx.disable(moderngl.BLEND)
    target.use()
    ctx.viewport = (0, 0, *target.size)
    tex.use(0)
    pipe["p_canvas"]["u_tex"] = 0
    pipe["vao_canv"].render(moderngl.TRIANGLE_STRIP)


def compute_attractors(st, lum):
    """Top active neurons become the flow field's swirl centres; particles born
    there inherit that neuron's community colour."""
    k = min(64, N)
    idx = np.argpartition(-lum, k - 1)[:k]
    m = float(lum[idx].max())
    if m <= 1e-9:
        st.nk = 0
        return
    sel = idx[lum[idx] > 0.02 * m][:64]
    st.nk = len(sel)
    st.attr[: st.nk, :3] = pos3[sel]
    st.attr[: st.nk, 3] = lum[sel] / m
    st.acol[: st.nk, :3] = colors[sel]


def render_frame(ctx, pipe, st, w, h):
    src, dst = pipe["accum"][st.ping], pipe["accum"][1 - st.ping]
    ctx.disable(moderngl.BLEND)
    dst[0].use()
    src[1].use(0)
    pipe["p_decay"]["u_tex"] = 0
    pipe["p_decay"]["u_decay"] = (st.decay + 0.006, st.decay, st.decay - 0.006)
    pipe["vao_decay"].render(moderngl.TRIANGLE_STRIP)

    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.ONE, moderngl.ONE  # additive HDR
    for name in ("p_point", "p_edge"):
        p = pipe[name]
        p["u_res"] = (w, h)
        p["u_angle"] = st.angle
        p["u_tilt"] = ORBIT_TILT
        p["u_cam"] = CAM
        p["u_fov"] = FOV
    if st.edges_on and pipe["vao_edges"] is not None:
        pipe["p_edge"]["u_alpha"] = EDGE_ALPHA
        pipe["vao_edges"].render(moderngl.LINES, vertices=pipe["n_edge_v"])
    if st.mode in (1, 3):
        pipe["p_point"]["u_size"] = SPLAT_SIZE
        pipe["vao_pts"].render(moderngl.TRIANGLE_STRIP, vertices=4, instances=N)

    if st.mode in (2, 3) and pipe["np"] > 0:
        ps = pipe["p_psim"]
        ps["u_dt"] = st.dt
        ps["u_time"] = st.time
        ps["u_rate"] = st.rate
        ps["u_n"] = st.nk
        ps["u_attr"].write(st.attr.tobytes())
        ps["u_acol"].write(st.acol.tobytes())
        pipe["vao_psim"][st.pping].transform(
            pipe["pbuf"][1 - st.pping], moderngl.POINTS, vertices=pipe["np"]
        )
        st.pping = 1 - st.pping
        pp = pipe["p_part"]
        pp["u_res"] = (w, h)
        pp["u_angle"] = st.angle
        pp["u_tilt"] = ORBIT_TILT
        pp["u_cam"] = CAM
        pp["u_fov"] = FOV
        pp["u_psize"] = P_SIZE
        pp["u_pgain"] = P_GAIN
        pipe["vao_pdraw"][st.pping].render(moderngl.POINTS, vertices=pipe["np"])

    st.ping = 1 - st.ping
    scene_tex = pipe["accum"][st.ping][1]

    ctx.disable(moderngl.BLEND)
    b0, b1 = pipe["blur"]
    b0[0].use()
    scene_tex.use(0)
    pipe["p_blur"]["u_tex"] = 0
    pipe["p_blur"]["u_dir"] = (1.0, 0.0)
    pipe["vao_blur"].render(moderngl.TRIANGLE_STRIP)
    for _ in range(BLOOM_ITERS):
        b1[0].use()
        b0[1].use(0)
        pipe["p_blur"]["u_dir"] = (0.0, 1.0)
        pipe["vao_blur"].render(moderngl.TRIANGLE_STRIP)
        b0[0].use()
        b1[1].use(0)
        pipe["p_blur"]["u_dir"] = (1.0, 0.0)
        pipe["vao_blur"].render(moderngl.TRIANGLE_STRIP)
    return scene_tex, b0[1]


def composite(ctx, pipe, st, scene_tex, bloom_tex, target):
    target.use()
    ctx.viewport = (0, 0, *target.size)
    ctx.disable(moderngl.BLEND)
    scene_tex.use(0)
    bloom_tex.use(1)
    p = pipe["p_comp"]
    p["u_scene"], p["u_bloom"] = 0, 1
    p["u_exposure"], p["u_bloom_w"], p["u_vignette"] = st.exposure, BLOOM_W, VIGNETTE
    p["u_aspect"], p["u_kfold"], p["u_bg"] = (
        target.size[0] / target.size[1],
        st.kfold,
        BG,
    )
    pipe["vao_comp"].render(moderngl.TRIANGLE_STRIP)


# ------------------------------------------------------------- windowed mode
import glfw

if not glfw.init():
    sys.exit("glfw init failed")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)  # accum/bloom FBOs are fixed-size
win = glfw.create_window(RES[0], RES[1], "signals of a smaller brain", None, None)
if not win:
    glfw.terminate()
    sys.exit("window creation failed")
glfw.make_context_current(win)
glfw.swap_interval(1)  # vsync
ctx = moderngl.create_context()
fbw, fbh = glfw.get_framebuffer_size(win)
pipe = build_pipeline(ctx, fbw, fbh)
st = State(fbw, fbh)
st.edges_on = False
lay = canvas_layout(
    fbw, fbh, margin=0.10, mix=0.35
)  # lower mix = more clustering, more empty canvas
painter = Painter(
    ctx,
    fbw,
    fbh,
    lay,
    COMMUNITY if COMMUNITY is not None else PAL_IDX,
    _signs,
    _edges,
    seed=SEED,
)
st.mode = 4
st.orbit = True


def on_key(_w, key, _sc, action, mods):
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    step = 20 if (mods & glfw.MOD_SHIFT) else 1
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(win, True)
    elif key == glfw.KEY_SPACE and action == glfw.PRESS:
        st.playing = not st.playing
    elif key == glfw.KEY_RIGHT:
        advance(st, step)
    elif key == glfw.KEY_LEFT:
        st.t = max(
            0.0, st.t - step
        )  # trail is causal; scrubbing back just moves the clock
    elif key == glfw.KEY_E and action == glfw.PRESS:
        st.edges_on = not st.edges_on
    elif key == glfw.KEY_UP:
        st.decay = min(0.995, st.decay + 0.005)
        print(f"[rt] screen decay {st.decay:.3f}")
    elif key == glfw.KEY_DOWN:
        st.decay = max(0.50, st.decay - 0.005)
        print(f"[rt] screen decay {st.decay:.3f}")
    elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
        st.exposure = min(6.0, st.exposure * 1.1)
    elif key == glfw.KEY_MINUS:
        st.exposure = max(0.05, st.exposure / 1.1)
    elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3) and action == glfw.PRESS:
        st.mode = key - glfw.KEY_1 + 1
        print(f"[rt] mode {st.mode} ({['neurons', 'particles', 'both'][st.mode - 1]})")
    elif key == glfw.KEY_K and action == glfw.PRESS:
        st.kfold = {0: 6, 6: 8, 8: 0}[st.kfold]
        print(f"[rt] kaleidoscope {st.kfold or 'off'}")
    elif key == glfw.KEY_O and action == glfw.PRESS:
        st.orbit = not st.orbit
    elif key == glfw.KEY_R and action == glfw.PRESS:
        st.t = 0.0
        st.trail[:] = 0
        painter.clear()
        for f, t in pipe["accum"]:
            f.use()
            ctx.clear(0, 0, 0, 1)
    elif key == glfw.KEY_4 and action == glfw.PRESS:
        st.mode = 4
    elif key == glfw.KEY_C and action == glfw.PRESS:
        painter.clear()
    elif key == glfw.KEY_P and action == glfw.PRESS:
        print(f"[rt] palette {painter.next_palette()}")
    elif key == glfw.KEY_S and action == glfw.PRESS:
        print(f"[rt] style {painter.next_style()}")
    elif key == glfw.KEY_N and action == glfw.PRESS:
        painter.recompose()
        print("[rt] new composition")


glfw.set_key_callback(win, on_key)

# The window's own framebuffer. NOT ctx.detect_framebuffer(): that returns
# whichever framebuffer is bound at call time, and build_pipeline() finishes by
# .use()-ing the blur FBOs to clear them. Detecting here therefore handed back a
# half-res offscreen buffer, so every composited frame landed there and the
# window stayed black forever. ctx.screen is the default framebuffer; glo 0 is
# the explicit fallback.
screen = ctx.screen if ctx.screen is not None else ctx.detect_framebuffer(0)
print(f"[rt] output framebuffer glo={screen.glo} size={screen.size} (0 = window)")
last = glfw.get_time()
acc_t, fps_n, fps_t = 0.0, 0, glfw.get_time()

while not glfw.window_should_close(win):
    now = glfw.get_time()
    dt, last = now - last, now
    if st.playing:
        acc_t += dt * STEPS_PER_SEC
        n_steps, acc_t = int(acc_t), acc_t - int(acc_t)
        advance(st, n_steps)
    if st.orbit:
        st.angle += ORBIT_SPEED * dt
    st.dt = float(np.clip(dt, 1 / 240, 1 / 20))
    st.time += st.dt
    st.rate = float(spikes[int(st.t) % T].mean()) * 20.0
    lum = upload_colors(st, pipe)
    if st.mode == 4:
        painter.frame(lum, st.rate, st.dt, st.time, screen)
    else:
        compute_attractors(st, lum)
        scene, bloom = render_frame(ctx, pipe, st, fbw, fbh)
        composite(ctx, pipe, st, scene, bloom, screen)
    glfw.swap_buffers(win)
    glfw.poll_events()
    fps_n += 1
    if now - fps_t > 2.0:
        print(
            f"[rt] {fps_n / (now - fps_t):5.1f} fps | step {int(st.t) % T}/{T} | "
            f"{'playing' if st.playing else 'paused'}"
        )
        fps_n, fps_t = 0, now

glfw.terminate()
