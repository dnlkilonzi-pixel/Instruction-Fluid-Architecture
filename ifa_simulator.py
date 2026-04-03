"""
Instruction Fluid Architecture (IFA) Simulator — v2
====================================================

Upgrades over v1
----------------
1. Multi-CPU field interference — N CPU particles move simultaneously in the
   shared field, creating computational interference and emergent paths.
2. Dynamic field mutation — CPUs rewrite the instruction field as they compute:
   high accumulators strengthen local transform weights, bind events carve
   memory scars, and movement density periodically reshapes flow vectors.
3. Programmable scalar potential Φ(x,y) — CPU movement follows −∇Φ instead of
   hard-coded attractor/repulsor vectors baked into the field.  Wells and hills
   are declared as data; the gradient is computed numerically each step.
4. Temporal layering Φ(x,y,t) — the instruction field oscillates in time.
   Each cell has a random phase offset, producing instruction echoes that
   fade in/out and reappear.
5. Semantic DSL instruction atoms — instruction vectors now hold weights for
   transform, bind, flow, split, collapse, amplify instead of raw add/mul/jump.
   These primitives describe computational physics rather than CPU opcodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRID_W, GRID_H = 50, 50     # field dimensions (cells)
N_STEPS        = 250         # number of simulation steps
N_CPUS         = 5           # number of concurrent CPU particles (Upgrade 1)
SAMPLE_RADIUS  = 3           # cell-neighbourhood radius for field sampling
DECAY_RATE     = 0.0015      # global per-step field decay rate

# ── Upgrade 5: Semantic DSL instruction atoms ─────────────────────────────────
# Each cell stores a vector of N_OPS weights, one per semantic atom.
IDX_TRANSFORM = 0   # reshape accumulator (arithmetic intensity)
IDX_BIND      = 1   # snapshot accumulator → flag register (jump checkpoint)
IDX_FLOW_X    = 2   # horizontal movement impulse
IDX_FLOW_Y    = 3   # vertical movement impulse
IDX_SPLIT     = 4   # deposit CPU energy into local field (divergence)
IDX_COLLAPSE  = 5   # absorb local field energy into accumulator (convergence)
IDX_AMPLIFY   = 6   # scale the magnitude of all other ops this step
IDX_DECAY     = 7   # per-cell decay rate modifier (static)
N_OPS         = 8

# Op firing thresholds
BIND_THRESHOLD     = 0.05
SPLIT_THRESHOLD    = 0.10
COLLAPSE_THRESHOLD = 0.10
AMPLIFY_THRESHOLD  = 0.08

# ── Upgrade 2: Field mutation parameters ──────────────────────────────────────
MUT_TRANSFORM_SCALE = 0.003  # accumulator magnitude → local transform boost
MUT_TRANSFORM_CAP   = 2.5    # ceiling on transform weight
MUT_SCAR_STRENGTH   = 0.06   # bind event → bind-weight reduction in vicinity
MUT_SCAR_RADIUS     = 2      # cell radius of memory scars
MUT_DENSITY_SCALE   = 0.0008 # density gradient → flow deflection (applied periodically)

# ── Upgrade 3: Potential field parameters ─────────────────────────────────────
POT_GRADIENT_WEIGHT = 0.25   # how strongly −∇Φ biases CPU movement

# ── Upgrade 4: Temporal echo parameters ───────────────────────────────────────
ECHO_AMPLITUDE = 0.20        # oscillation amplitude (fraction of base field)
ECHO_PERIOD    = 70          # steps per full oscillation cycle

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTION FIELD — construction
# ─────────────────────────────────────────────────────────────────────────────

def build_instruction_field(width: int, height: int) -> np.ndarray:
    """
    Build the base 2-D instruction field of shape (height, width, N_OPS).

    Two programmatic regions:
      Region A (left half)  — transform-heavy and collapse-weighted
                               → arithmetic / energy-collection zone.
      Region B (right half) — flow / bind / split / amplify
                               → movement and control-flow zone.
    A global rightward base drift simulates a program-counter advance.
    """
    rng   = np.random.default_rng(seed=42)
    field = np.zeros((height, width, N_OPS), dtype=float)
    half  = width // 2

    # Region A — arithmetic zone
    field[:, :half, IDX_TRANSFORM] = rng.uniform(0.4, 1.0, (height, half))
    field[:, :half, IDX_COLLAPSE]  = rng.uniform(0.05, 0.2, (height, half))

    # Region B — movement / control-flow zone
    field[:, half:, IDX_FLOW_X]  = rng.uniform(-0.4, 0.6, (height, width - half))
    field[:, half:, IDX_FLOW_Y]  = rng.uniform(-0.4, 0.6, (height, width - half))
    field[:, half:, IDX_BIND]    = rng.uniform(0.0, 0.22, (height, width - half))
    field[:, half:, IDX_SPLIT]   = rng.uniform(0.0, 0.18, (height, width - half))
    field[:, half:, IDX_AMPLIFY] = rng.uniform(0.0, 0.14, (height, width - half))

    # Per-cell decay rates scattered everywhere
    field[:, :, IDX_DECAY] = rng.uniform(0.0, 0.002, (height, width))

    # Base rightward flow throughout — analogous to a program-counter advance
    field[:, :, IDX_FLOW_X] += 0.13

    return field


def build_phase_field(width: int, height: int) -> np.ndarray:
    """
    Generate per-cell random phase offsets for the temporal echo (Upgrade 4).
    Shape: (height, width), values in [0, 2π).
    """
    rng = np.random.default_rng(seed=7)
    return rng.uniform(0.0, 2.0 * np.pi, (height, width))


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 4 — TEMPORAL LAYERING  field(x, y, t)
# ─────────────────────────────────────────────────────────────────────────────

def apply_temporal_echo(base_field: np.ndarray,
                        phase_field: np.ndarray,
                        t: int) -> np.ndarray:
    """
    Return the time-modulated effective field at timestep *t*.

        field_eff(x,y,t) = base_field(x,y) · (1 + A · sin(2π·t/T + φ(x,y)))

    Each cell has its own phase offset φ, so different regions of the field
    fade in and out at different times — "instruction echoes".
    The IDX_DECAY channel is kept static (decay rates do not oscillate).
    """
    osc = 1.0 + ECHO_AMPLITUDE * np.sin(
        2.0 * np.pi * t / ECHO_PERIOD + phase_field
    )                                                    # shape (H, W)
    eff = base_field * osc[:, :, np.newaxis]             # broadcast over N_OPS
    eff[:, :, IDX_DECAY] = base_field[:, :, IDX_DECAY]  # keep decay static
    return eff


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 3 — PROGRAMMABLE SCALAR POTENTIAL Φ(x, y)
# ─────────────────────────────────────────────────────────────────────────────

def build_potential_field(width:  int,
                          height: int,
                          wells:  list,
                          hills:  list) -> np.ndarray:
    """
    Construct the scalar potential Φ(x, y).

    wells : list of (x, y, strength)          — attractive gravity wells
    hills : list of (x, y, strength, radius)  — Gaussian repulsive hills

    CPUs follow −∇Φ: they descend into wells (attracted) and climb
    away from hills (repelled), without any vectors baked into the
    instruction field itself.
    """
    gy_idx, gx_idx = np.mgrid[0:height, 0:width].astype(float)
    phi = np.zeros((height, width))

    for wx, wy, wstrength in wells:
        dist  = np.hypot(gx_idx - wx, gy_idx - wy) + 1.0
        phi  -= wstrength / dist                           # well = negative potential

    for hx, hy, hstrength, hradius in hills:
        dist  = np.hypot(gx_idx - hx, gy_idx - hy)
        phi  += hstrength * np.exp(-0.5 * (dist / hradius) ** 2)  # Gaussian hill

    return phi


def potential_gradient(phi: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Numerical gradient of Φ at continuous position *pos* = (x, y).
    Returns (∂Φ/∂x, ∂Φ/∂y) via central finite differences.
    """
    height, width = phi.shape
    ix = int(np.clip(round(pos[0]), 1, width  - 2))
    iy = int(np.clip(round(pos[1]), 1, height - 2))
    grad_x = (phi[iy, ix + 1] - phi[iy, ix - 1]) / 2.0
    grad_y = (phi[iy + 1, ix] - phi[iy - 1, ix]) / 2.0
    return np.array([grad_x, grad_y])


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 2 — DYNAMIC FIELD MUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _cell_index(pos: np.ndarray, width: int, height: int):
    """Convert continuous position to integer cell indices (clamped)."""
    ix = int(np.clip(pos[0], 0, width  - 1))
    iy = int(np.clip(pos[1], 0, height - 1))
    return ix, iy


def mutate_field_transform(field: np.ndarray,
                           pos: np.ndarray,
                           accumulator: float) -> None:
    """
    High accumulator → strengthen IDX_TRANSFORM weight at the CPU's cell.
    Computation writes its own arithmetic intensity into the physics —
    the field grows more potent where CPUs have been busy calculating.
    """
    height, width = field.shape[:2]
    ix, iy = _cell_index(pos, width, height)
    delta  = min(MUT_TRANSFORM_SCALE * abs(accumulator), 0.08)
    field[iy, ix, IDX_TRANSFORM] = min(
        field[iy, ix, IDX_TRANSFORM] + delta, MUT_TRANSFORM_CAP
    )


def mutate_field_scar(field: np.ndarray, pos: np.ndarray) -> None:
    """
    Bind event → carve a memory scar by reducing IDX_BIND weight locally.
    Future CPUs passing this region are less likely to re-checkpoint here,
    forming an inhibitory memory trace analogous to synaptic depression.
    """
    height, width = field.shape[:2]
    ix, iy = _cell_index(pos, width, height)
    r      = MUT_SCAR_RADIUS
    iy_lo  = max(0,        iy - r)
    iy_hi  = min(height-1, iy + r)
    ix_lo  = max(0,        ix - r)
    ix_hi  = min(width -1, ix + r)
    field[iy_lo:iy_hi+1, ix_lo:ix_hi+1, IDX_BIND] -= MUT_SCAR_STRENGTH
    np.clip(field[:, :, IDX_BIND], 0.0, None, out=field[:, :, IDX_BIND])


def update_density_flow(field: np.ndarray, density_map: np.ndarray) -> None:
    """
    Movement density → reshape flow vectors away from congested cells.
    The field self-routes around heavily-visited regions, preventing
    the instruction highways from becoming over-saturated.
    """
    dy_d, dx_d = np.gradient(density_map.astype(float))
    norm = max(float(np.abs(dx_d).max()), float(np.abs(dy_d).max()), 1e-6)
    field[:, :, IDX_FLOW_X] -= MUT_DENSITY_SCALE * dx_d / norm
    field[:, :, IDX_FLOW_Y] -= MUT_DENSITY_SCALE * dy_d / norm


# ─────────────────────────────────────────────────────────────────────────────
# FIELD DECAY
# ─────────────────────────────────────────────────────────────────────────────

def apply_field_decay(field: np.ndarray) -> None:
    """
    Weaken the field each timestep.
    Each cell decays by DECAY_RATE + its own IDX_DECAY modifier.
    The IDX_DECAY channel itself is never modified.
    """
    local_decay = DECAY_RATE + field[:, :, IDX_DECAY]
    for i in range(N_OPS - 1):     # skip IDX_DECAY channel
        field[:, :, i] *= 1.0 - local_decay
    np.clip(field, 0.0, None, out=field)


# ─────────────────────────────────────────────────────────────────────────────
# UPGRADE 1 — CPU PARTICLE (one of N running concurrently)
# ─────────────────────────────────────────────────────────────────────────────

class CPUParticle:
    """
    Virtual CPU particle — one of N moving simultaneously in the shared field.

    New capabilities vs v1
    ----------------------
    * Reads the temporally-echoed field (Upgrade 4).
    * Movement = field flow ops + potential gradient −∇Φ (Upgrade 3).
    * Executes semantic DSL atoms: transform, bind, split, collapse,
      amplify (Upgrade 5).
    * Writes mutations back into the shared field after each step (Upgrade 2).
    """

    def __init__(self, x0: float, y0: float, cpu_id: int = 0) -> None:
        self.cpu_id    = cpu_id
        self.pos       = np.array([x0, y0], dtype=float)
        self.registers: dict = {
            "accumulator": 0.0,   # primary arithmetic value
            "counter":     0,     # step counter
            "flag":        0.0,   # bind/checkpoint snapshot register
            "energy":      1.0,   # vitality: rises with collapse, falls with split
        }

    # ── field sampling ───────────────────────────────────────────────────────

    def sample_field(self, eff_field: np.ndarray,
                     radius: int = SAMPLE_RADIUS) -> np.ndarray:
        """Inverse-distance-weighted neighbourhood sampling of the effective field."""
        height, width = eff_field.shape[:2]
        cx, cy        = self.pos
        impulse       = np.zeros(N_OPS)
        total_w       = 0.0

        ix_min = max(0,        int(cx) - radius)
        ix_max = min(width -1, int(cx) + radius)
        iy_min = max(0,        int(cy) - radius)
        iy_max = min(height-1, int(cy) + radius)

        for gy in range(iy_min, iy_max + 1):
            for gx in range(ix_min, ix_max + 1):
                dx   = gx - cx
                dy   = gy - cy
                dist = np.hypot(dx, dy) + 1e-6
                if dist - 1e-6 > radius:
                    continue
                w        = 1.0 / dist
                impulse += w * eff_field[gy, gx]
                total_w += w

        if total_w > 0.0:
            impulse /= total_w
        return impulse

    # ── single timestep ──────────────────────────────────────────────────────

    def step(self, base_field: np.ndarray, phase_field: np.ndarray,
             phi: np.ndarray, density_map: np.ndarray, t: int) -> None:
        """
        Execute one simulation timestep (all 5 upgrades integrated).

        Pipeline
        --------
        1.  Apply temporal echo to get the effective field at time t.
        2.  Sample the effective field (inverse-distance weighted).
        3.  Compute amplify factor from the amplify atom.
        4.  Execute DSL atoms:
              transform — grows accumulator from local field intensity
              bind      — snapshots accumulator → flag; carves memory scar
              split     — deposits energy into local field cells
              collapse  — absorbs local field energy into accumulator
        5.  Mutate base field based on accumulator level (Upgrade 2).
        6.  Compute movement = field flow + potential gradient descent.
        7.  Update position; clamp to grid bounds.
        8.  Record cell visit in density map.
        """
        height, width = base_field.shape[:2]

        # 1. Temporal echo (Upgrade 4)
        eff_field = apply_temporal_echo(base_field, phase_field, t)

        # 2. Sample
        impulse = self.sample_field(eff_field)

        # 3. Amplify atom — boosts magnitude of all other ops this step
        amplify = (1.0 + impulse[IDX_AMPLIFY]
                   if impulse[IDX_AMPLIFY] > AMPLIFY_THRESHOLD else 1.0)

        # 4. DSL atoms ─────────────────────────────────────────────────────────

        # transform: reshape accumulator (analogous to arithmetic computation)
        self.registers["accumulator"] += amplify * impulse[IDX_TRANSFORM]

        # bind: checkpoint accumulator → flag; carve memory scar
        if impulse[IDX_BIND] > BIND_THRESHOLD:
            self.registers["flag"]        = self.registers["accumulator"]
            self.registers["accumulator"] = 0.0
            mutate_field_scar(base_field, self.pos)             # Upgrade 2

        # split: deposit CPU energy into local field (divergence)
        if impulse[IDX_SPLIT] > SPLIT_THRESHOLD:
            ix, iy = _cell_index(self.pos, width, height)
            deposit = min(0.025 * amplify, 0.1)
            base_field[iy, ix, IDX_TRANSFORM] = min(
                base_field[iy, ix, IDX_TRANSFORM] + deposit, MUT_TRANSFORM_CAP
            )
            self.registers["energy"] = max(self.registers["energy"] - 0.02, 0.1)

        # collapse: absorb local field energy into accumulator (convergence)
        if impulse[IDX_COLLAPSE] > COLLAPSE_THRESHOLD:
            ix, iy = _cell_index(self.pos, width, height)
            local_energy = base_field[iy, ix, IDX_TRANSFORM]
            self.registers["accumulator"] += 0.4 * local_energy * amplify
            self.registers["energy"] = min(self.registers["energy"] + 0.02, 2.0)

        self.registers["counter"] += 1

        # 5. Accumulator-driven field strengthening (Upgrade 2)
        mutate_field_transform(base_field, self.pos,
                               self.registers["accumulator"])

        # 6. Movement = flow atoms + −∇Φ (Upgrade 3)
        flow_x = impulse[IDX_FLOW_X]
        flow_y = impulse[IDX_FLOW_Y]
        grad   = potential_gradient(phi, self.pos)
        move_x = flow_x - POT_GRADIENT_WEIGHT * grad[0]
        move_y = flow_y - POT_GRADIENT_WEIGHT * grad[1]

        # 7. Update position
        self.pos[0] = np.clip(self.pos[0] + move_x, 0.0, float(width  - 1))
        self.pos[1] = np.clip(self.pos[1] + move_y, 0.0, float(height - 1))

        # 8. Density map
        ix, iy = _cell_index(self.pos, width, height)
        density_map[iy, ix] += 1


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(n_steps: int = N_STEPS, n_cpus: int = N_CPUS):
    """
    Build all fields, spawn N CPUs, and run the simulation loop.

    Returns
    -------
    initial_field  : base instruction field before any mutation/decay
    final_field    : field after the full simulation
    phase_field    : per-cell temporal phase offsets (static)
    phi            : scalar potential field Φ(x,y) (static)
    trajectories   : list of (n_steps+1, 2) arrays — one per CPU
    reg_histories  : list of register-history dicts — one per CPU
    density_map    : (H, W) visit-count heatmap
    wells          : potential-well specifications
    hills          : potential-hill specifications
    """
    # --- build fields ---
    base_field  = build_instruction_field(GRID_W, GRID_H)
    phase_field = build_phase_field(GRID_W, GRID_H)

    # Upgrade 3: programmable potential field
    wells = [(46, 10, 8.0), (46, 40, 8.0)]        # two attractive wells
    hills = [(25, 25, 3.5, 5.0)]                   # one central Gaussian hill
    phi   = build_potential_field(GRID_W, GRID_H, wells, hills)

    initial_field = base_field.copy()
    density_map   = np.zeros((GRID_H, GRID_W))

    # Upgrade 1: N CPUs spread across the arithmetic (left) region
    start_positions = [
        (3.0,  8.0),
        (3.0, 18.0),
        (3.0, 25.0),
        (3.0, 35.0),
        (3.0, 44.0),
    ][:n_cpus]

    cpus          = [CPUParticle(x0, y0, i)
                     for i, (x0, y0) in enumerate(start_positions)]
    trajectories  = [[cpu.pos.copy()] for cpu in cpus]
    reg_histories = [{k: [float(v)] for k, v in cpu.registers.items()}
                     for cpu in cpus]

    # --- main simulation loop ---
    for t in range(n_steps):
        # All CPUs step through the shared field (Upgrade 1: interference)
        for i, cpu in enumerate(cpus):
            cpu.step(base_field, phase_field, phi, density_map, t)
            trajectories[i].append(cpu.pos.copy())
            for k, v in cpu.registers.items():
                reg_histories[i][k].append(float(v))

        # Upgrade 2: density-driven flow reshaping every 25 steps
        if t > 0 and t % 25 == 0:
            update_density_flow(base_field, density_map)

        apply_field_decay(base_field)

    trajectories = [np.array(tr) for tr in trajectories]
    return (initial_field, base_field, phase_field, phi,
            trajectories, reg_histories, density_map, wells, hills)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

_CPU_COLORS = ["cyan", "lime", "yellow", "magenta", "orange"]


def _field_intensity(field: np.ndarray) -> np.ndarray:
    """Sum all non-decay op weights into a scalar intensity map."""
    return field[:, :, :IDX_DECAY].sum(axis=2)


def visualise(initial_field, final_field, phase_field, phi,
              trajectories, reg_histories, density_map, wells, hills,
              save_path: str = "ifa_simulation.png") -> None:
    """
    Eight-panel visualisation covering all 5 upgrades:

      (a) gs[0,0:2]  Initial instruction field — DSL op-weight heatmap with CPU starts
      (b) gs[0,2]    Scalar potential Φ(x,y)  — wells (▲) and hills (■) marked
      (c) gs[1,0]    All CPU trajectories     — overlaid on the post-simulation field
      (d) gs[1,1]    Visit density            — emergent instruction highways
      (e) gs[1,2]    Temporal echo snapshot   — field at t = ECHO_PERIOD//2
      (f) gs[2,0]    Field mutation delta     — how CPUs rewrote the physics
      (g) gs[2,1]    Accumulator time series  — arithmetic emergence for all CPUs
      (h) gs[2,2]    Flag register            — bind/checkpoint events for all CPUs
    """
    fig       = plt.figure(figsize=(20, 14))
    grid_spec = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    init_intensity  = _field_intensity(initial_field)
    final_intensity = _field_intensity(final_field)

    # ── (a) Initial instruction field ─────────────────────────────────────────
    ax_a = fig.add_subplot(grid_spec[0, 0:2])
    im_a = ax_a.imshow(init_intensity, origin="lower", cmap="plasma",
                        extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im_a, ax=ax_a, label="Op Weight Sum")
    for i, tr in enumerate(trajectories):
        ax_a.plot(tr[0, 0], tr[0, 1], "o",
                  color=_CPU_COLORS[i % len(_CPU_COLORS)], ms=7, zorder=3,
                  label=f"CPU {i}")
    ax_a.set_title(
        "Initial Instruction Field  "
        "(DSL: transform · bind · flow · split · collapse · amplify)"
    )
    ax_a.set_xlabel("X"); ax_a.set_ylabel("Y")
    ax_a.legend(loc="upper right", fontsize=7, ncol=3)

    # ── (b) Scalar potential Φ ────────────────────────────────────────────────
    ax_b = fig.add_subplot(grid_spec[0, 2])
    im_b = ax_b.imshow(phi, origin="lower", cmap="RdYlGn_r",
                        extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im_b, ax=ax_b, label="Φ")
    for wx, wy, _ in wells:
        ax_b.plot(wx, wy, "g^", ms=9, label="Well")
    for hx, hy, _, _ in hills:
        ax_b.plot(hx, hy, "rs", ms=9, label="Hill")
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    ax_b.legend(dict(zip(labels_b, handles_b)).values(),
                dict(zip(labels_b, handles_b)).keys(), fontsize=7)
    ax_b.set_title("Scalar Potential Φ(x,y)\n(CPUs follow −∇Φ)")
    ax_b.set_xlabel("X"); ax_b.set_ylabel("Y")

    # ── (c) CPU trajectories ──────────────────────────────────────────────────
    ax_c = fig.add_subplot(grid_spec[1, 0])
    ax_c.imshow(final_intensity, origin="lower", cmap="plasma", alpha=0.4,
                extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    for i, tr in enumerate(trajectories):
        c = _CPU_COLORS[i % len(_CPU_COLORS)]
        ax_c.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.1, alpha=0.85,
                  label=f"CPU {i}")
        ax_c.plot(tr[0,  0], tr[0,  1], "o", color=c, ms=5, zorder=4)
        ax_c.plot(tr[-1, 0], tr[-1, 1], "*", color=c, ms=9, zorder=4)
    ax_c.set_title("Multi-CPU Trajectories\n(○ start  ★ end — interference visible)")
    ax_c.set_xlabel("X"); ax_c.set_ylabel("Y")
    ax_c.legend(loc="upper right", fontsize=7, ncol=2)

    # ── (d) Visit density ─────────────────────────────────────────────────────
    ax_d = fig.add_subplot(grid_spec[1, 1])
    im_d = ax_d.imshow(density_map, origin="lower", cmap="hot",
                        extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im_d, ax=ax_d, label="Visit Count")
    ax_d.set_title("Visit Density\n(emergent instruction highways)")
    ax_d.set_xlabel("X"); ax_d.set_ylabel("Y")

    # ── (e) Temporal echo snapshot ────────────────────────────────────────────
    ax_e = fig.add_subplot(grid_spec[1, 2])
    echo_t      = ECHO_PERIOD // 2
    echo_intens = _field_intensity(
        apply_temporal_echo(initial_field, phase_field, echo_t)
    )
    im_e = ax_e.imshow(echo_intens, origin="lower", cmap="plasma",
                        extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im_e, ax=ax_e, label="Op Weight Sum")
    ax_e.set_title(
        f"Temporal Echo  (t = {echo_t})\ninstruction echoes shift field intensity"
    )
    ax_e.set_xlabel("X"); ax_e.set_ylabel("Y")

    # ── (f) Field mutation delta ───────────────────────────────────────────────
    ax_f = fig.add_subplot(grid_spec[2, 0])
    delta = final_intensity - init_intensity
    vmax  = max(float(np.abs(delta).max()), 1e-6)
    im_f  = ax_f.imshow(delta, origin="lower", cmap="bwr",
                         vmin=-vmax, vmax=vmax,
                         extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im_f, ax=ax_f, label="Δ Op Weight")
    ax_f.set_title(
        "Field Mutation Delta  (final − initial)\nred = strengthened  blue = weakened"
    )
    ax_f.set_xlabel("X"); ax_f.set_ylabel("Y")

    # ── (g) Accumulator time series ───────────────────────────────────────────
    ax_g = fig.add_subplot(grid_spec[2, 1])
    for i, rh in enumerate(reg_histories):
        ax_g.plot(rh["accumulator"],
                  color=_CPU_COLORS[i % len(_CPU_COLORS)],
                  lw=1.0, alpha=0.9, label=f"CPU {i}")
    ax_g.set_title("Accumulator  (all CPUs)\narithmetic emergence via transform + collapse")
    ax_g.set_xlabel("Step"); ax_g.set_ylabel("Value")
    ax_g.legend(fontsize=7, ncol=2)

    # ── (h) Flag register time series ─────────────────────────────────────────
    ax_h = fig.add_subplot(grid_spec[2, 2])
    for i, rh in enumerate(reg_histories):
        ax_h.plot(rh["flag"],
                  color=_CPU_COLORS[i % len(_CPU_COLORS)],
                  lw=1.0, alpha=0.9, label=f"CPU {i}")
    ax_h.set_title("Flag Register  (all CPUs)\nbind checkpoints — computation memory")
    ax_h.set_xlabel("Step"); ax_h.set_ylabel("Value")
    ax_h.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Instruction Fluid Architecture (IFA) Simulator — v2\n"
        "Multi-CPU · Programmable Potential · Temporal Echoes · "
        "Field Mutation · Semantic DSL",
        fontsize=13, y=1.01
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualisation saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Instruction Fluid Architecture (IFA) Simulator — v2 ===")
    print(f"Grid: {GRID_W}×{GRID_H}  |  Steps: {N_STEPS}  |  "
          f"CPUs: {N_CPUS}  |  Sample radius: {SAMPLE_RADIUS}")

    (initial_field, final_field, phase_field, phi,
     trajectories, reg_histories,
     density_map, wells, hills) = run_simulation()

    print("\n--- Per-CPU summary ---")
    for i, (tr, rh) in enumerate(zip(trajectories, reg_histories)):
        flags    = rh["flag"]
        n_binds  = sum(1 for j in range(1, len(flags)) if flags[j] != flags[j-1])
        max_acc  = max(rh["accumulator"])
        print(f"  CPU {i}: start={tr[0]}  end={tr[-1].round(1)}"
              f"  max_acc={max_acc:.2f}  binds={n_binds}")

    total_visits = int(density_map.sum())
    print(f"\nTotal field visits : {total_visits}")
    print(f"Peak density cell  : {int(density_map.max())} visits")

    visualise(initial_field, final_field, phase_field, phi,
              trajectories, reg_histories, density_map, wells, hills)
