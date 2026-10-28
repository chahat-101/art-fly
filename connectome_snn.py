import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- config
SERVER = "neuprint.janelia.org"
DATASET = "hemibrain:v1.2.1"
ROI = "AL(R)"  # right antennal lobe: olfactory, ~2-3k neurons, clear sensory input
CACHE = "connectome_cache.npz"
OUTDIR = "out"

T_STEPS = 300  # simulation timesteps per rollout
TBPTT_K = 25  # detach the graph every K steps (truncated BPTT)
DT = 1.0  # ms
V_REST, V_RESET = 0.0, 0.0
REFRAC = 2  # refractory steps

TARGET_RATE = 0.02  # mean spike probability per neuron per step (~20 Hz at 1 ms)
TARGET_CHI = 0.25  # chi synchrony: ~1/sqrt(N) is fully async, 1.0 is lockstep.
# The AL is genuinely glomerular, so some collective structure is real biology,
# not a failure - asking for 0.15 here chases a floor the anatomy won't go below.
STIM_RATE = 0.25  # per-input-neuron Poisson drive under its own envelope
TARGET_BRANCH = 1.0  # critical branching ratio (avalanche-like dynamics)

# loss weights (all terms are normalized, so these are directly comparable)
W_RATE, W_DEAD, W_CHI, W_BRANCH, W_RUNAWAY = 1.0, 0.5, 0.6, 0.4, 3.0
CURRICULUM_START, CURRICULUM_END = 15, 45  # ramp chi/branch in over this epoch range

EPOCHS, LR, WARMUP = 150, 0.01, 5
EPOCHS = int(os.environ.get("EPOCHS", "150"))  # overridable for quick pipeline smoke-tests
SEED = int(os.environ.get("FLY_SEED", "0"))

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTDIR, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------------------------------------- 1. fetch / cache connectome
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    rows, cols, vals = z["rows"], z["cols"], z["vals"]
    signs, pos, types = z["signs"], z["pos"], z["types"]
    print(f"[data] loaded cached connectome: {len(signs)} neurons, {len(vals)} synapses")
else:
    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

    client = Client(SERVER, dataset=DATASET, token=os.environ["NEUPRINT_TOKEN"])
    print(f"[data] connected to {DATASET}")

    crit = NC(rois=[ROI], status="Traced", cropped=False)
    neurons, _ = fetch_neurons(crit)
    neurons = neurons.drop_duplicates("bodyId").reset_index(drop=True)
    print(f"[data] {len(neurons)} traced neurons in {ROI}")

    _, conns = fetch_adjacencies(
        NC(bodyId=neurons.bodyId.tolist()),
        NC(bodyId=neurons.bodyId.tolist()),
        rois=[ROI],
    )
    conns = conns.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    conns = conns[conns.weight >= 3]  # drop very weak / noisy edges
    print(f"[data] {len(conns)} synaptic connections (weight >= 3)")

    keep = pd.Index(sorted(set(conns.bodyId_pre) | set(conns.bodyId_post)))
    neurons = neurons[neurons.bodyId.isin(keep)].reset_index(drop=True)
    idx = {b: i for i, b in enumerate(neurons.bodyId)}
    rows = conns.bodyId_post.map(idx).to_numpy()  # post = destination row
    cols = conns.bodyId_pre.map(idx).to_numpy()  # pre  = source column
    vals = conns.weight.to_numpy().astype(np.float32)

    # biological sign: GABA and glutamate are inhibitory in the fly
    nt = neurons.get("predictedNt", pd.Series([""] * len(neurons))).fillna("").str.lower()
    if (nt == "").all() and "ntType" in neurons.columns:
        nt = neurons["ntType"].fillna("").str.lower()
    signs = np.where(nt.str.contains("gaba|glutamate|inhibit", regex=True), -1.0, 1.0).astype(np.float32)

    if (signs < 0).sum() == 0:
        rng = np.random.RandomState(SEED)
        signs = np.ones(len(signs), np.float32)
        signs[rng.choice(len(signs), size=int(0.2*len(signs)), replace=False)] = -1.0
        print("[net ] no NT labels – forced 20 % random inhibition")

    # real soma coordinates drive the artwork's spatial layout
    pos = np.stack(
        [
            np.array(p if isinstance(p, (list, tuple)) else [np.nan] * 3, dtype=np.float32)
            for p in neurons.get("somaLocation", [None] * len(neurons))
        ]
    )
    nan = np.isnan(pos).any(1)
    pos[nan] = np.nanmean(pos[~nan], axis=0) if (~nan).any() else 0.0
    pos += np.random.RandomState(SEED).normal(0, 1e-3, pos.shape).astype(np.float32)

    types = neurons.get("type", pd.Series([""] * len(neurons))).fillna("").to_numpy().astype(str)
    np.savez(CACHE, rows=rows, cols=cols, vals=vals, signs=signs, pos=pos, types=types)
    print(f"[data] cached -> {CACHE}")

N = len(signs)

# -------------------------------------------------- 2. sparse weight matrix (fixed biology)
w = np.log1p(vals) * signs[cols]  # log-compress so a few huge connections can't dominate
W = sp.csr_matrix((w, (rows, cols)), shape=(N, N), dtype=np.float32)
W = W.multiply(1.0 / (np.abs(W).sum(1) + 1e-6))  # row-normalize incoming drive
W = sp.csr_matrix(W, dtype=np.float32)

# split into excitatory and inhibitory parts so their gains can be tuned
# INDEPENDENTLY. With one shared gain the firing-rate constraint consumes the
# only degree of freedom and E/I balance - the actual control parameter for
# asynchronous, critical dynamics - is unreachable.
torch.sparse.check_sparse_tensor_invariants.disable()


def _to_sparse(M):
    c = M.tocoo()
    return (
        torch.sparse_coo_tensor(
            np.vstack([c.row, c.col]), c.data.astype(np.float32), (N, N)
        )
        .coalesce()
        .to(DEV)
    )


W_exc_sp = W.copy()
W_exc_sp.data = np.maximum(W_exc_sp.data, 0)
W_exc_sp.eliminate_zeros()
W_inh_sp = W.copy()
W_inh_sp.data = np.minimum(W_inh_sp.data, 0)
W_inh_sp.eliminate_zeros()
W_exc_t, W_inh_t = _to_sparse(W_exc_sp), _to_sparse(W_inh_sp)

def _repair_positions(p, Wcsr, seed=SEED):
    """Neurons whose soma lies outside the imaged volume have no somaLocation
    and were all dumped onto one mean point - on screen that is the single
    blazing white blob. Spread them by harmonic interpolation on the graph:
    each missing neuron sits at the weighted mean of its neighbours, iterated,
    so ORNs land next to the glomeruli they actually innervate."""
    p = p.copy()
    _, inv, cnt = np.unique(np.round(p, 1), axis=0, return_inverse=True, return_counts=True)
    miss = cnt[inv] > max(3, int(0.02 * len(p)))
    n = int(miss.sum())
    if n == 0 or (~miss).sum() < 3:
        return p, 0
    rs = np.random.RandomState(seed + 1)
    A = Wcsr.copy()
    A.data = np.abs(A.data)
    A = (A + A.T).tocsr()
    scale = p[~miss].std(0).mean() + 1e-6
    p[miss] = p[~miss].mean(0) + rs.normal(0, 0.5 * scale, (n, 3))
    deg = np.asarray(A.sum(1)).ravel()
    mi = np.where(miss)[0]
    for _ in range(80):
        nb = (A @ p)[mi] / np.maximum(deg[mi], 1e-9)[:, None]
        ok = deg[mi] > 1e-9
        p[mi[ok]] = 0.7 * nb[ok] + 0.3 * p[mi[ok]]
    p[miss] += rs.normal(0, 0.04 * scale, (n, 3))
    return p, n


pos, _n_repaired = _repair_positions(pos, W)
if _n_repaired:
    print(f"[net ] re-laid out {_n_repaired} neurons with no soma location via graph harmonic fill")

frac_inh = float((signs < 0).mean())
print(f"[net ] {N} neurons | {W.nnz} synapses | {frac_inh:.1%} inhibitory | device={DEV}")

# ------------------------------------------------------------------- 3. stimulus (sensory)
is_input = np.array([t.startswith("ORN") or t.startswith("HRN") for t in types])
if is_input.sum() < 5:  # fallback: highest out-degree neurons
    outdeg = np.asarray((W != 0).sum(0)).ravel()
    is_input = np.zeros(N, bool)
    is_input[np.argsort(-outdeg)[: max(10, N // 50)]] = True
input_np = np.where(is_input)[0]
input_idx = torch.tensor(input_np, device=DEV)
print(f"[net ] {len(input_idx)} sensory input neurons")

# --- which neurons can EVER fire? anything not reachable from the inputs is hopeless,
#     so excluding them stops the dead-neuron loss from chasing an impossible target
_reach = np.zeros(N, dtype=bool)
_reach[input_np] = True
_Gb = (W != 0).astype(np.float32).tocsr()  # Gb[post, pre]
for _ in range(64):
    _new = _reach | ((_Gb @ _reach.astype(np.float32)) > 0)
    if _new.sum() == _reach.sum():
        break
    _reach = _new
reach_mask = torch.tensor(_reach.astype(np.float32), device=DEV)
n_reach = int(_reach.sum())
print(f"[net ] {n_reach}/{N} neurons reachable from inputs ({1 - n_reach / N:.1%} excluded from dead-loss)")


def make_stim(seed):
    """Poisson drive with a PER-NEURON envelope phase.

    A single shared envelope makes every input neuron fire in lockstep, which
    injects synchrony the network cannot undo - chi then sits at a floor set by
    the stimulus and no dynamics parameter can move it. Independent phases mean
    measured synchrony reflects the network, not the drive.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_in = len(input_np)
    phase = torch.rand(n_in, generator=g) * 2 * np.pi
    t = torch.linspace(0, 4 * np.pi, T_STEPS)[:, None]
    env = 0.5 + 0.5 * torch.sin(t + phase[None, :])
    r = torch.rand(T_STEPS, n_in, generator=g)
    return (r < (STIM_RATE * env)).float().to(DEV)

def make_odor_stim(seed, t_steps, n_odors=6, episode=500, duty=0.6):
    """Odour episodes: each odour drives a fixed subset of glomeruli, then a
    quiet gap. Different communities take turns, so the output has phases."""
    g = np.random.RandomState(seed)
    glom = np.unique(types[input_np], return_inverse=True)[1].ravel()
    if glom.max() < 3:                                   # no type labels: fake 20 groups
        glom = g.randint(0, 20, len(input_np))
    odors = [g.rand(glom.max() + 1) < 0.25 for _ in range(n_odors)]
    rate = np.full((t_steps, len(input_np)), 0.03*STIM_RATE, np.float32)   # background
    for s in range(0, t_steps, episode):
        on = slice(s, min(s + int(episode*duty), t_steps))
        rate[on, odors[g.randint(n_odors)][glom]] = STIM_RATE
    return torch.tensor(g.rand(t_steps, len(input_np)) < rate,
                        dtype=torch.float32, device=DEV)

stim_eval = make_stim(SEED)  # fixed stimulus for comparable diagnostics

# ------------------------------------------------ 4. learnable dynamics (topology stays fixed)
# initialized slightly ABOVE target rate: a mildly over-active net has strong surrogate
# gradients, whereas a silent one gives almost none and training stalls
log_tau = torch.tensor(np.log(20.0), device=DEV, requires_grad=True)  # membrane time constant
thresh = torch.tensor(0.12, device=DEV, requires_grad=True)  # spike threshold
in_gain = torch.tensor(2.5, device=DEV, requires_grad=True)  # sensory drive gain
log_w_exc = torch.tensor(np.log(5.0), device=DEV, requires_grad=True)  # excitatory gain
log_w_inh = torch.tensor(np.log(5.0), device=DEV, requires_grad=True)  # inhibitory gain
log_noise = torch.tensor(np.log(0.01), device=DEV, requires_grad=True)  # membrane noise
params = [log_tau, thresh, in_gain, log_w_exc, log_w_inh, log_noise]


class SurrGradSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward."""

    scale = 10.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        return grad_out / (SurrGradSpike.scale * v.abs() + 1.0) ** 2


spike_fn = SurrGradSpike.apply


def simulate(stim, t_steps=T_STEPS, detach=False, tbptt=TBPTT_K):
    """Run the LIF network over the fixed connectome. Returns [T, N] spikes.

    `tbptt` truncates the backprop graph every K steps. Without it, gradients
    through a 300-step recurrent spiking chain are effectively noise.
    """
    tau = log_tau.exp().clamp(2.0, 200.0)
    we = log_w_exc.exp().clamp(0.1, 25.0)
    wi = log_w_inh.exp().clamp(0.1, 25.0)
    nz = log_noise.exp().clamp(0.0, 0.2)
    v = torch.zeros(N, device=DEV)
    ref = torch.zeros(N, device=DEV)
    prev = torch.zeros(N, device=DEV)
    out = []
    for t in range(t_steps):
        if (not detach) and tbptt and t > 0 and t % tbptt == 0:
            v, ref, prev = v.detach(), ref.detach(), prev.detach()
        pu = prev.unsqueeze(1)
        i_rec = (
            torch.sparse.mm(W_exc_t, pu).squeeze(1) * we
            + torch.sparse.mm(W_inh_t, pu).squeeze(1) * wi
        )
        i_ext = torch.zeros(N, device=DEV).index_put_(
            (input_idx,), stim[t % stim.shape[0]] * in_gain
        )
        v = v + (DT / tau) * (V_REST - v + i_rec + i_ext)
        v = v + nz * torch.randn(N, device=DEV)  # decorrelating membrane noise
        v = torch.where(ref > 0, torch.full_like(v, V_RESET), v)  # refractory clamp
        s = spike_fn(v - thresh)
        v = v * (1 - s) + V_RESET * s
        ref = torch.clamp(ref - 1, min=0) + s * REFRAC
        prev = s.detach() if detach else s
        out.append(s)
    return torch.stack(out)


def metrics(spikes):
    """rate, dead, chi (synchrony), branch (branching ratio) - all differentiable."""
    per_neuron = spikes.mean(0)
    pop = spikes.mean(1)
    rate = per_neuron.mean().clamp(min=1e-7)

    # dead: only count neurons that are structurally capable of firing
    dead_soft = torch.exp(-per_neuron / (0.25 * TARGET_RATE))
    dead = (dead_soft * reach_mask).sum() / reach_mask.sum()

    # chi synchrony (Golomb-Rinzel): population variance over mean single-neuron
    # variance. ~1/sqrt(N) when asynchronous, 1.0 in lockstep. Crucially this is
    # RATE-INVARIANT, so it cannot be minimized by simply firing more.
    chi = torch.sqrt(pop.var() / (spikes.var(0).mean() + 1e-9))

    # branching ratio as a least-squares slope through the origin
    a, b = pop[:-1], pop[1:]
    branch = (a * b).sum() / ((a * a).sum() + 1e-9)
    return rate, dead, chi, branch


# ------------------------------------------------------------ 5. homeostatic training loop
opt = torch.optim.AdamW(params, lr=LR)


def lr_at(ep):
    if ep <= WARMUP:
        return LR * ep / max(WARMUP, 1)
    p = (ep - WARMUP) / max(EPOCHS - WARMUP, 1)
    return LR * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * p)))


history = []
best = {"score": float("inf")}

print(f"\n[train] target_rate={TARGET_RATE} target_chi={TARGET_CHI} target_branch={TARGET_BRANCH}")
print(f"{'epoch':>6} {'loss':>8} {'rate':>8} {'dead':>7} {'chi':>7} {'branch':>8} {'ramp':>6} {'lr':>9}")

for epoch in range(1, EPOCHS + 1):
    lr_now = lr_at(epoch)
    for pg in opt.param_groups:
        pg["lr"] = lr_now

    opt.zero_grad()
    spikes = simulate(make_stim(SEED + epoch))  # fresh stimulus each epoch
    rate, dead, chi, branch = metrics(spikes)

    # curriculum: get the firing rate under control before asking for structure,
    # otherwise the terms fight and the optimizer just oscillates
    ramp = float(np.clip((epoch - CURRICULUM_START) / (CURRICULUM_END - CURRICULUM_START), 0, 1))

    l_rate = W_RATE * torch.log(rate / TARGET_RATE) ** 2  # symmetric in log space
    l_dead = W_DEAD * dead
    l_chi = W_CHI * ramp * (F.relu(chi - TARGET_CHI) / TARGET_CHI) ** 2
    l_branch = W_BRANCH * ramp * ((branch - TARGET_BRANCH) / 0.3) ** 2
    l_runaway = W_RUNAWAY * F.relu(rate - 5 * TARGET_RATE) ** 2 / TARGET_RATE**2
    loss = l_rate + l_dead + l_chi + l_branch + l_runaway

    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    with torch.no_grad():  # keep parameters physically sane
        thresh.clamp_(0.05, 2.0)
        in_gain.clamp_(0.0, 10.0)

    # curriculum-independent score so checkpoints are comparable across epochs
    score = float(
        np.log(max(rate.item(), 1e-7) / TARGET_RATE) ** 2
        + (max(chi.item() - TARGET_CHI, 0) / TARGET_CHI) ** 2
        + ((branch.item() - TARGET_BRANCH) / 0.3) ** 2
        + dead.item()
    )
    if score < best["score"]:
        best = {
            "score": score,
            "epoch": epoch,
            "vals": [p.detach().clone() for p in params],
        }

    history.append(
        (epoch, loss.item(), rate.item(), dead.item(), chi.item(), branch.item(), lr_now)
    )
    if epoch % 5 == 0 or epoch == 1:
        print(
            f"{epoch:6d} {loss.item():8.4f} {rate.item():8.4f} {dead.item():7.3f} "
            f"{chi.item():7.4f} {branch.item():8.4f} {ramp:6.2f} {lr_now:9.2e}"
        )

# the objective is non-convex and mildly oscillatory; keep the best point, not the last
with torch.no_grad():
    for p, v in zip(params, best["vals"]):
        p.copy_(v)
print(f"\n[train] restored best epoch {best['epoch']} (score {best['score']:.4f})")
print(
    f"[train] tau={float(log_tau.detach().exp().clamp(2.0, 200.0)):.1f}ms "
    f"thresh={float(thresh):.3f} in_gain={float(in_gain):.3f} "
    f"w_exc={float(log_w_exc.detach().exp()):.2f} w_inh={float(log_w_inh.detach().exp()):.2f} "
    f"E/I={float((log_w_exc - log_w_inh).detach().exp()):.3f} "
    f"noise={float(log_noise.detach().exp()):.4f}"
)

# ------------------------------------------------------------------- 6. final rollout + save
ROLLOUT = 6000   # ~100 s of playback at 60 steps/s
with torch.no_grad():
    final_t = simulate(make_odor_stim(SEED, ROLLOUT), t_steps=ROLLOUT, detach=True)
    r_f, d_f, c_f, b_f = metrics(final_t)
    final = final_t.cpu().numpy()
np.save(f"{OUTDIR}/spikes.npy", final)
np.save(f"{OUTDIR}/positions.npy", pos)
print(
    f"[out ] spikes {final.shape} | rate {r_f.item():.4f} | chi {c_f.item():.4f} | "
    f"branch {b_f.item():.4f} | active {(final.sum(0) > 0).mean():.1%}"
)

h = np.array(history)
fig, ax = plt.subplots(1, 5, figsize=(23, 3.4))
for a, (col, title, tgt) in zip(
    ax,
    [
        (1, "loss", None),
        (2, "mean firing rate", TARGET_RATE),
        (4, "chi (synchrony)", TARGET_CHI),
        (5, "branching ratio", TARGET_BRANCH),
        (6, "learning rate", None),
    ],
):
    a.plot(h[:, 0], h[:, col], color="#B5651D")
    if tgt is not None:
        a.axhline(tgt, ls="--", c="#2A2118", lw=1)
    a.set_title(title)
    a.set_xlabel("epoch")
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/training.png", dpi=140)
plt.close()

sub = final[:, np.argsort(-final.sum(0))[:400]]
t_i, n_i = np.nonzero(sub)
plt.figure(figsize=(13, 5), facecolor="#F2EBDD")
plt.scatter(t_i, n_i, s=0.45, c="#2A2118", marker="|", linewidths=0.6)
plt.xlabel("timestep")
plt.ylabel("neuron (by activity)")
plt.title(f"{ROI} spike raster")
plt.gca().set_facecolor("#F2EBDD")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/raster.png", dpi=150)
plt.close()

# avalanche analysis
pop_final = final.sum(axis=1)
aval_thresh = pop_final.mean() * 0.5
above = pop_final > aval_thresh
avalanche_sizes, in_aval, acc = [], False, 0.0
for t in range(len(above)):
    if above[t]:
        in_aval, acc = True, acc + pop_final[t]
    elif in_aval:
        avalanche_sizes.append(acc)
        acc, in_aval = 0.0, False
if in_aval:
    avalanche_sizes.append(acc)
avalanche_sizes = np.array(avalanche_sizes)
print(f"[diag] {len(avalanche_sizes)} avalanches (thresh={aval_thresh:.1f})")

if len(avalanche_sizes) > 2:
    s_min, s_max = avalanche_sizes.min(), avalanche_sizes.max()
    bins = (
        np.logspace(np.log10(max(s_min, 0.5)), np.log10(s_max + 1), 30)
        if s_max > s_min
        else np.logspace(0, 1, 10)
    )
    counts, edges = np.histogram(avalanche_sizes, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    m = counts > 0
    fig, a = plt.subplots(figsize=(6, 5))
    a.scatter(centers[m], counts[m], s=28, color="#2A2118", zorder=3)
    if m.sum() > 1:
        rx = centers[m]
        a.plot(rx, counts[m][0] * (rx / rx[0]) ** -1.5, "--", c="#B5651D", label="slope -1.5")
        a.legend(frameon=False)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("avalanche size (total spikes)")
    a.set_ylabel("count")
    a.set_title(f"{ROI} avalanche sizes (sigma={b_f.item():.3f})")
    a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/avalanches.png", dpi=140)
    plt.close()

# ------------------------------------------------------- 7. signal-to-art rendering frames
from scipy.ndimage import gaussian_filter
import colorsys

# ========================= RENDERER CONFIG =========================
CANVAS_SIZE = 1440

PALETTE_HEX = [
    "#00E5FF", "#FF006E", "#7B2FBE", "#00C853", "#FFB300", "#FF5722",
    "#E040FB", "#00E676", "#2979FF", "#FF1744", "#C6FF00", "#D500F9",
]
EXC_HUE_SHIFT, INH_HUE_SHIFT = 0.04, -0.06
DECAY_R, DECAY_G, DECAY_B = 0.89, 0.88, 0.87  # chromatic aberration in the trail
# Spread tightened from .90/.87/.84: that width pushed the TYPICAL frame to an
# R/B ratio near 4.9 (hard amber). At .89/.88/.87 with emission scaling the
# average is hue-neutral and only the fading tails stay slightly warm.
# Different per-channel decay means different STEADY-STATE accumulation
# (ss_c = rate/(1-decay_c)), so a white neuron drifts to amber as the trail
# fills. Scaling emission by (1-decay_c) makes equilibrium hue-neutral, leaving
# the aberration visible only in fading tails - which is the part we wanted.

BLOOM_WEIGHTS = [0.5, 0.3, 0.2]
BLOOM_SIGMA_BASE = 10.0
EXPOSURE_PERCENTILE, EXPOSURE_MIN, EXPOSURE_KEY = 99.0, 0.05, 0.55
EXPOSURE_CLAMP = (0.05, 2.5)  # hard bounds so quiet frames can't flare
EXPOSURE_SMOOTH = 0.85  # temporal EMA on exposure -> no flicker

GRADE_LIFT = np.array([0.00, -0.01, 0.03], dtype=np.float32)
GRADE_GAMMA = np.array([0.98, 0.97, 1.02], dtype=np.float32)
GRADE_GAIN = np.array([1.00, 0.99, 1.04], dtype=np.float32)
VIGNETTE_STRENGTH, GRAIN_STRENGTH = 0.45, 0.015

ORBIT_SPEED, ORBIT_TILT, FOV_PIXELS, DEPTH_EXAGGERATION = 0.015, 0.25, 900.0, 2.5
KERNEL_RADIUS_BASE, MIN_SPLAT_RADIUS, MAX_SPLAT_RADIUS = 14, 4, 24
BASE_SPLAT_INTENSITY = 1.6  # lowered: additive overlap was clipping dense regions
DEPTH_DESAT, DEPTH_COOL_SHIFT = 0.35, 0.07

MAX_EDGES, SCAFFOLD_ALPHA, SCAFFOLD_SAMPLES = 30_000, 0.03, 8
MAX_PULSES, MAX_EDGES_PER_SPIKE = 20_000, 6
PULSE_BASE_SPEED, PULSE_BRIGHTNESS, PULSE_BLUR = 8.0, 1.8, 2.2
JITTER_AMPLITUDE = 0.4
BG_COLOR = np.array([0.04, 0.03, 0.06], dtype=np.float32)
# ===================================================================


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _consolidate_communities(ids, k_keep=12):
    """Keep the k biggest communities; fold every leftover community into the kept
    one it is most strongly wired to (summed |W| flow). Keeps G sane so hues stay
    distinct and per-community ink signals are smooth instead of binary flickers."""
    ids = np.asarray(ids, np.int32).copy()
    u, sizes = np.unique(ids, return_counts=True)
    if len(u) > k_keep:
        big = u[np.argsort(-sizes)][:k_keep]
        D = (ids[:, None] == big[None, :]).astype(np.float32)      # N x k_keep
        flows = np.abs(W) @ D                                       # per-neuron flow into each kept group
        assign = big[flows.argmax(1)]
        ids = np.where(np.isin(ids, big), ids, assign)
    u, sizes = np.unique(ids, return_counts=True)
    order = u[np.argsort(-sizes)]                                   # relabel biggest = 0
    lut = np.full(ids.max() + 1, -1, np.int32)
    for r, c in enumerate(order):
        lut[c] = r
    return lut[ids], len(u)


# community detection with a BOUNDED resolution search. The old unbounded
# escalation kept splitting until the graph was 647 singletons, which made
# every artwork hue rank-tied mush and its ink signals binary flickers.
# We stop at the first resolution whose largest group is < 35% of neurons
# with <= 24 communities; then consolidation folds the rest.
community_ids, n_comm, res = None, 0, 1.0
try:
    import igraph as ig

    Wa = W.copy()
    Wa.data = np.abs(Wa.data)
    Ws = ((Wa + Wa.T) / 2).tocoo()
    up = Ws.row < Ws.col
    g_ig = ig.Graph(n=N, edges=list(zip(Ws.row[up].tolist(), Ws.col[up].tolist())), directed=False)
    g_ig.es["weight"] = Ws.data[up].tolist()

    def _leiden(r):
        return np.array(
            g_ig.community_leiden(weights="weight", resolution=r, n_iterations=3).membership,
            dtype=np.int32,
        )

    picked = None
    for res in (0.5, 1.0, 1.75, 3.0, 5.0):          # bounded: no runaway escalation
        ids = _leiden(res)
        mx = float(np.bincount(ids).max()) / N
        nc = int(ids.max()) + 1
        if 2 <= nc <= 24:
            picked = ids
            if mx < 0.35:
                break
    if picked is None:                               # last resort: the most balanced bounded set
        print("[render] bounded Leiden search never balanced; merging coarsest")
        picked = _leiden(0.5)
    community_ids, _n0 = _consolidate_communities(picked, k_keep=12)
    n_comm = int(community_ids.max()) + 1
    _sizes = np.sort(np.bincount(community_ids))[::-1]
    print(
        f"[render] Leiden: {n_comm} communities (from {_n0}) at resolution {res:g} | "
        f"largest holds {_sizes[0] / N:.0%} | top-5 sizes {_sizes[:5].tolist()}"
    )
except Exception as e:
    from scipy.sparse.linalg import eigsh
    from scipy.cluster.vq import kmeans2

    print(f"[render] igraph unavailable ({type(e).__name__}); using spectral clustering")
    Wa = W.copy()
    Wa.data = np.abs(Wa.data)
    Ws = ((Wa + Wa.T) / 2).tocsr()
    deg = np.asarray(Ws.sum(1)).ravel()
    deg[deg < 1e-10] = 1e-10
    Dis = sp.diags(1.0 / np.sqrt(deg))
    k = min(len(PALETTE_HEX), max(4, N // 80))
    _, ev = eigsh((sp.eye(N) - Dis @ Ws @ Dis).tocsc(), k=k, which="SM")
    ev = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + 1e-10)
    _, community_ids = kmeans2(ev.astype(np.float32), k, minit="points", seed=SEED)
    community_ids = community_ids.astype(np.int32)
    n_comm = k

palette_rgb = np.array([_hex_to_rgb(h) for h in PALETTE_HEX])
# map by SIZE RANK: community_ids are arbitrary labels, and with more
# communities than palette entries, modulo can hand the two biggest
# communities the same hue. Rank them so the dominant groups on screen are
# guaranteed maximally distinct colours.
_rank = {c: r for r, c in enumerate(np.argsort(-np.bincount(community_ids)))}
neuron_colors = np.zeros((N, 3), dtype=np.float32)
for _i in range(N):
    base = palette_rgb[_rank[community_ids[_i]] % len(palette_rgb)]
    hh, ss, vv = colorsys.rgb_to_hsv(*map(float, base))
    hh = (hh + (EXC_HUE_SHIFT if signs[_i] > 0 else INH_HUE_SHIFT)) % 1.0
    neuron_colors[_i] = np.array(colorsys.hsv_to_rgb(hh, ss, vv), dtype=np.float32)

# --- 3D setup ---
rel_3d = (pos - pos.mean(0)).astype(np.float32)
ar = np.ptp(rel_3d, axis=0)
fa = int(np.argmin(ar))
if ar[fa] < 0.3 * ar.max():
    rel_3d[:, fa] *= DEPTH_EXAGGERATION
rel_3d = (rel_3d / (np.std(rel_3d) + 1e-6)) * (CANVAS_SIZE * 0.22)
camera_distance = 3.0 * float(np.max(np.linalg.norm(rel_3d, axis=1)))

# auto-frame: choose FOV so the projected cloud fills ~72% of the canvas
# instead of huddling in the middle of a mostly-empty frame
_rmax = 0.0
for _a in (0.0, np.pi / 3, 2 * np.pi / 3):
    _ca, _sa = np.cos(_a), np.sin(_a)
    _xr = rel_3d[:, 0] * _ca - rel_3d[:, 2] * _sa
    _zr = rel_3d[:, 0] * _sa + rel_3d[:, 2] * _ca
    _ct, _st = np.cos(ORBIT_TILT), np.sin(ORBIT_TILT)
    _yt = rel_3d[:, 1] * _ct - _zr * _st
    _zt = rel_3d[:, 1] * _st + _zr * _ct
    _zc = np.maximum(_zt + camera_distance, camera_distance * 0.1)
    _rmax = max(_rmax, float(np.percentile(np.sqrt(_xr**2 + _yt**2) / _zc, 99.0)))
FOV_PIXELS = (0.72 * CANVAS_SIZE / 2) / max(_rmax, 1e-6)
print(f"[render] auto-framed FOV = {FOV_PIXELS:.0f}px")

_rj = np.random.RandomState(SEED + 42)
jitter_x = _rj.uniform(-JITTER_AMPLITUDE, JITTER_AMPLITUDE, N).astype(np.float32)
jitter_y = _rj.uniform(-JITTER_AMPLITUDE, JITTER_AMPLITUDE, N).astype(np.float32)

kernels_lut = {}
for _r in range(MIN_SPLAT_RADIUS, MAX_SPLAT_RADIUS + 1):
    _g = np.arange(-_r, _r + 1, dtype=np.float32)
    _kx, _ky = np.meshgrid(_g, _g)
    _k = np.exp(-(_kx**2 + _ky**2) / (2 * (_r / 2.5) ** 2)).astype(np.float32)
    kernels_lut[_r] = _k / _k.max()

# --- edges ---
cr = W.tocoo()
e_pre, e_post, e_str = cr.col.copy(), cr.row.copy(), np.abs(cr.data)
if len(e_pre) > MAX_EDGES:
    top = np.argsort(-e_str)[:MAX_EDGES]
    e_pre, e_post, e_str = e_pre[top], e_post[top], e_str[top]
n_edges = len(e_pre)
edge_scaffold_color = neuron_colors[e_pre] * ((e_str / (e_str.max() + 1e-6)) * SCAFFOLD_ALPHA)[:, None]
edge_3d_pre, edge_3d_post = rel_3d[e_pre], rel_3d[e_post]
edge_3d_len = np.linalg.norm(edge_3d_post - edge_3d_pre, axis=1)

# outgoing edges per neuron, capped so a dense hub can't flood the pulse pool
order = np.argsort(e_pre, kind="stable")
e_pre_sorted = e_pre[order]
starts = np.searchsorted(e_pre_sorted, np.arange(N), "left")
ends = np.searchsorted(e_pre_sorted, np.arange(N), "right")
out_edges = [order[starts[i] : ends[i]][:MAX_EDGES_PER_SPIKE] for i in range(N)]
print(f"[render] {n_edges} scaffold edges | pulse pool {MAX_PULSES}")

# everything realtime_art.py needs, so both renderers share one look
np.savez(
    f"{OUTDIR}/render_assets.npz",
    colors=neuron_colors, e_pre=e_pre, e_post=e_post, e_str=e_str, signs=signs,
    community=community_ids,
)
print(f"[render] exported {OUTDIR}/render_assets.npz for realtime_art.py")

_vy, _vx = np.mgrid[0:CANVAS_SIZE, 0:CANVAS_SIZE].astype(np.float32)
_vx = _vx / (CANVAS_SIZE - 1) * 2 - 1
_vy = _vy / (CANVAS_SIZE - 1) * 2 - 1
vignette_mask = np.clip(1.0 - VIGNETTE_STRENGTH * (_vx**2 + _vy**2), 0, 1)[:, :, None]
rng_grain = np.random.RandomState(SEED + 7)

pulse_edge = np.zeros(MAX_PULSES, dtype=np.int32)
pulse_prog = np.zeros(MAX_PULSES, dtype=np.float32)
pulse_speed = np.zeros(MAX_PULSES, dtype=np.float32)
pulse_active = np.zeros(MAX_PULSES, dtype=bool)
pulse_head = 0


def project_points(p, angle, tilt):
    ca, sa = np.cos(angle), np.sin(angle)
    xr = p[:, 0] * ca - p[:, 2] * sa
    zr = p[:, 0] * sa + p[:, 2] * ca
    ct, st = np.cos(tilt), np.sin(tilt)
    yt = p[:, 1] * ct - zr * st
    zt = p[:, 1] * st + zr * ct
    zc = np.maximum(zt + camera_distance, camera_distance * 0.1)
    return FOV_PIXELS * xr / zc + CANVAS_SIZE / 2, FOV_PIXELS * yt / zc + CANVAS_SIZE / 2, zc


def scatter_bilinear(layer, xs, ys, cols):
    """Vectorized sub-pixel point scatter - no per-point Python loop."""
    ok = (xs >= 0) & (xs < CANVAS_SIZE - 1) & (ys >= 0) & (ys < CANVAS_SIZE - 1)
    if not ok.any():
        return
    xs, ys, cols = xs[ok], ys[ok], cols[ok]
    x0, y0 = np.floor(xs).astype(np.int32), np.floor(ys).astype(np.int32)
    fx, fy = (xs - x0)[:, None], (ys - y0)[:, None]
    for dx, dy, wgt in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        cw = cols * wgt
        for c in range(3):
            np.add.at(layer[:, :, c], (y0 + dy, x0 + dx), cw[:, c])


def splat_subpixel(canvas, cx, cy, kern, color):
    """Bilinear-weighted gaussian splat for soma (variable radius)."""
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    fx, fy = cx - ix, cy - iy
    Rk = kern.shape[0] // 2
    for ox, oy, wgt in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
        if wgt < 0.001:
            continue
        cxi, cyi = ix + ox, iy + oy
        y0, y1, x0, x1 = cyi - Rk, cyi + Rk + 1, cxi - Rk, cxi + Rk + 1
        ky0, kx0 = max(0, -y0), max(0, -x0)
        ky1 = kern.shape[0] - max(0, y1 - CANVAS_SIZE)
        kx1 = kern.shape[1] - max(0, x1 - CANVAS_SIZE)
        y0c, y1c, x0c, x1c = max(0, y0), min(CANVAS_SIZE, y1), max(0, x0), min(CANVAS_SIZE, x1)
        if y0c >= y1c or x0c >= x1c:
            continue
        canvas[y0c:y1c, x0c:x1c] += kern[ky0:ky1, kx0:kx1, None] * (color * wgt)[None, None, :]


def bloom_multiscale(canvas):
    res = canvas.copy()
    h, w = canvas.shape[:2]
    for i, bw in enumerate(BLOOM_WEIGHTS):
        f = 2 ** (i + 2)
        sh, sw = h // f, w // f
        if sh < 2 or sw < 2:
            continue
        ds = canvas[: sh * f, : sw * f].reshape(sh, f, sw, f, 3).mean((1, 3))
        for c in range(3):
            ds[:, :, c] = gaussian_filter(ds[:, :, c], sigma=BLOOM_SIGMA_BASE * (i + 1) / f)
        up = ds.repeat(f, 0).repeat(f, 1)
        for c in range(3):  # smooth the blocky nearest-neighbour upsample
            up[:, :, c] = gaussian_filter(up[:, :, c], sigma=f * 0.5)
        pad = np.zeros_like(canvas)
        pad[: up.shape[0], : up.shape[1]] = up
        res += bw * pad
    return res


def aces_tonemap(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def color_grade(img):
    g = img * GRADE_GAIN[None, None, :]
    g = np.power(np.maximum(g, 0), GRADE_GAMMA[None, None, :])
    g += GRADE_LIFT[None, None, :] * (1.0 - g)
    return np.clip(g, 0.0, 1.0)
