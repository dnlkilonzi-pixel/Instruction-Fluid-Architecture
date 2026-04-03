"""
Instruction Fluid Architecture (IFA) Simulator
===============================================

A research prototype demonstrating that computation can emerge from
continuous field interaction rather than discrete instruction sequencing.

Architecture overview
---------------------
* A 2-D *instruction field* (NumPy grid) stores per-cell vectors of
  operation weights (add, multiply, move-x, move-y, jump, decay).
* A *CPU particle* floats through the field and, at every timestep,
  samples nearby cells, aggregates the weighted vectors into an
  *execution impulse*, applies the impulse to its registers, and
  updates its position.
* Special field features (attractors, repulsors, decay) produce
  emergent trajectories and register dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GRID_W, GRID_H = 50, 50    # field dimensions (cells)
N_STEPS        = 250        # number of simulation steps
SAMPLE_RADIUS  = 3          # cell-neighbourhood radius for field sampling
DECAY_RATE     = 0.002      # global field magnitude decay per step
ATTRACTOR_STR  = 2.5        # pull strength added to movement vectors
REPULSOR_STR   = 1.8        # push strength added to movement vectors
REPULSOR_RADIUS = 9.0       # effective radius of a repulsor zone

MUL_THRESHOLD    = 0.1      # minimum multiply impulse before multiplication fires
MUL_SCALE_FACTOR = 0.1      # how strongly the multiply impulse scales the accumulator
JUMP_THRESHOLD   = 0.10     # minimum jump impulse that triggers a register checkpoint

# Indices into the per-cell instruction vector
IDX_ADD   = 0   # weight applied to accumulator addition
IDX_MUL   = 1   # weight applied to accumulator multiplication
IDX_MOVX  = 2   # horizontal movement impulse
IDX_MOVY  = 3   # vertical movement impulse
IDX_JUMP  = 4   # "jump" weight — triggers register checkpoint / reset
IDX_DECAY = 5   # local decay-rate modifier (not applied to itself)
N_OPS     = 6   # total number of instruction components

# ─────────────────────────────────────────────────────────────────────────────
# INSTRUCTION FIELD — construction and modification
# ─────────────────────────────────────────────────────────────────────────────

def build_instruction_field(width: int, height: int) -> np.ndarray:
    """
    Initialise a 2-D instruction field of shape (height, width, N_OPS).

    Two programmatic regions are created:
      * Region A (left half)  — arithmetic-heavy: strong add/multiply weights.
      * Region B (right half) — movement-heavy: strong move-x/move-y and
                                some jump weight.
    Scattered local decay modifiers are seeded across the entire field.
    """
    rng   = np.random.default_rng(seed=42)
    field = np.zeros((height, width, N_OPS), dtype=float)

    half = width // 2

    # Region A — arithmetic zone
    field[:, :half, IDX_ADD] = rng.uniform(0.3, 1.0, (height, half))
    field[:, :half, IDX_MUL] = rng.uniform(0.1, 0.5, (height, half))

    # Region B — movement / control-flow zone
    field[:, half:, IDX_MOVX] = rng.uniform(-0.5, 0.5, (height, width - half))
    field[:, half:, IDX_MOVY] = rng.uniform(-0.5, 0.5, (height, width - half))
    field[:, half:, IDX_JUMP] = rng.uniform(0.0,  0.3, (height, width - half))

    # Local decay modifiers scattered everywhere
    field[:, :, IDX_DECAY] = rng.uniform(0.0, 0.003, (height, width))

    # Global rightward flow — gives the CPU a base drift analogous to
    # a program counter advancing through instruction memory.
    field[:, :, IDX_MOVX] += 0.15

    return field


def add_attractor(field: np.ndarray, ax: int, ay: int,
                  strength: float = ATTRACTOR_STR) -> None:
    """
    Encode an attractor at grid coordinate (ax, ay).

    Every cell gains movement vector components that point *toward*
    (ax, ay), weighted by inverse distance.  This biases the CPU
    particle to drift in the attractor's direction.
    """
    height, width = field.shape[:2]
    gy_idx, gx_idx = np.mgrid[0:height, 0:width]   # shape (H, W) each

    dx   = ax - gx_idx                              # direction to attractor
    dy   = ay - gy_idx
    dist = np.hypot(dx, dy) + 1e-6
    w    = strength / (dist + 1.0)

    field[:, :, IDX_MOVX] += w * (dx / dist)
    field[:, :, IDX_MOVY] += w * (dy / dist)


def add_repulsor(field: np.ndarray, rx: int, ry: int,
                 strength: float = REPULSOR_STR,
                 radius:   float = REPULSOR_RADIUS) -> None:
    """
    Encode a repulsor at grid coordinate (rx, ry).

    Cells within *radius* gain movement vectors pointing *away* from
    (rx, ry).  The push is stronger the closer the cell is.
    """
    height, width = field.shape[:2]
    gy_idx, gx_idx = np.mgrid[0:height, 0:width]

    dx   = gx_idx - rx
    dy   = gy_idx - ry
    dist = np.hypot(dx, dy) + 1e-6
    mask = dist < radius                             # only nearby cells

    w = np.where(mask, strength * (1.0 - dist / radius), 0.0)
    field[:, :, IDX_MOVX] += w * (dx / dist)
    field[:, :, IDX_MOVY] += w * (dy / dist)


def apply_field_decay(field: np.ndarray) -> None:
    """
    Weaken the field every timestep.

    Each cell decays by the global DECAY_RATE *plus* its own IDX_DECAY
    modifier.  All operation components (except IDX_DECAY itself) are
    multiplied by (1 – decay).  Values never go below zero.
    """
    local_decay = DECAY_RATE + field[:, :, IDX_DECAY]   # shape (H, W)

    for i in range(N_OPS - 1):    # leave IDX_DECAY column untouched
        field[:, :, i] *= 1.0 - local_decay

    np.clip(field, 0.0, None, out=field)

# ─────────────────────────────────────────────────────────────────────────────
# CPU PARTICLE — state, sampling, and execution
# ─────────────────────────────────────────────────────────────────────────────

class CPUParticle:
    """
    A virtual CPU that drifts through the instruction field.

    Attributes
    ----------
    pos : np.ndarray, shape (2,)
        Continuous float position [x, y] within the field grid.
    registers : dict
        Named state registers that evolve through field interaction.
    """

    def __init__(self, x0: float, y0: float) -> None:
        self.pos = np.array([x0, y0], dtype=float)
        self.registers: dict = {
            "accumulator": 0.0,   # primary arithmetic value
            "counter":     0,     # counts executed steps
            "flag":        0.0,   # snapshot register (jump target analogue)
        }

    # ── field sampling ───────────────────────────────────────────────────────

    def sample_field(self, field: np.ndarray,
                     radius: int = SAMPLE_RADIUS) -> np.ndarray:
        """
        Collect instruction vectors from cells within *radius* of the CPU.

        Uses inverse-distance weighting so that immediately adjacent cells
        dominate the aggregate impulse.  Returns a single normalised vector
        of length N_OPS representing the blended *execution impulse*.
        """
        height, width = field.shape[:2]
        cx, cy = self.pos

        impulse      = np.zeros(N_OPS)
        total_weight = 0.0

        ix_min = max(0, int(cx) - radius)
        ix_max = min(width  - 1, int(cx) + radius)
        iy_min = max(0, int(cy) - radius)
        iy_max = min(height - 1, int(cy) + radius)

        for gy in range(iy_min, iy_max + 1):
            for gx in range(ix_min, ix_max + 1):
                dx   = gx - cx
                dy   = gy - cy
                dist = np.hypot(dx, dy) + 1e-6
                if dist - 1e-6 > radius:
                    continue
                w             = 1.0 / dist
                impulse      += w * field[gy, gx]
                total_weight += w

        if total_weight > 0.0:
            impulse /= total_weight

        return impulse

    # ── single timestep ──────────────────────────────────────────────────────

    def step(self, field: np.ndarray) -> None:
        """
        Execute one simulation timestep.

        Pipeline
        --------
        1. Sample the instruction field at the current position.
        2. Apply arithmetic components (add / multiply) to the accumulator.
        3. Apply jump component: if the impulse exceeds a threshold,
           snapshot the accumulator into the flag register and reset it
           (analogous to a conditional jump in classical architectures).
        4. Move the CPU by the movement components of the impulse.
        5. Clamp position to the grid bounds.
        """
        height, width = field.shape[:2]
        impulse = self.sample_field(field)

        # --- arithmetic operations ---
        add_imp = impulse[IDX_ADD]
        mul_imp = impulse[IDX_MUL]
        jump_w  = impulse[IDX_JUMP]

        self.registers["accumulator"] += add_imp
        if mul_imp > MUL_THRESHOLD:                # multiplication threshold
            self.registers["accumulator"] *= (1.0 + mul_imp * MUL_SCALE_FACTOR)

        # Jump: checkpoint accumulator into flag and clear it
        if jump_w > JUMP_THRESHOLD:
            self.registers["flag"]        = self.registers["accumulator"]
            self.registers["accumulator"] = 0.0

        self.registers["counter"] += 1

        # --- movement ---
        move_x = impulse[IDX_MOVX]
        move_y = impulse[IDX_MOVY]

        self.pos[0] = np.clip(self.pos[0] + move_x, 0.0, float(width  - 1))
        self.pos[1] = np.clip(self.pos[1] + move_y, 0.0, float(height - 1))

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(n_steps: int = N_STEPS):
    """
    Assemble the field, place the CPU particle, and iterate for *n_steps*.

    Returns
    -------
    initial_field : np.ndarray
        Snapshot of the field before any decay (used for visualisation).
    final_field   : np.ndarray
        Field state after the full simulation.
    trajectory    : np.ndarray, shape (n_steps+1, 2)
        CPU x/y position recorded at every step (including initial).
    reg_history   : dict of lists
        Register value recorded at every step for each register.
    attractors    : list of (int, int)
        Coordinates of attractor points.
    repulsors     : list of (int, int)
        Coordinates of repulsor points.
    """
    # --- build field ---
    field = build_instruction_field(GRID_W, GRID_H)

    # --- embed attractors ---
    attractors = [(46, 10)]
    for ax, ay in attractors:
        add_attractor(field, ax, ay)

    # --- embed repulsor below the CPU path to deflect trajectory upward ---
    repulsors = [(20, 20)]
    for rx, ry in repulsors:
        add_repulsor(field, rx, ry)

    # --- snapshot before decay ---
    initial_field = field.copy()

    # --- place CPU in the upper-left arithmetic region ---
    cpu = CPUParticle(x0=3.0, y0=30.0)

    # --- logging ---
    trajectory  = [cpu.pos.copy()]
    reg_history = {k: [float(v)] for k, v in cpu.registers.items()}

    # --- main simulation loop ---
    for _ in range(n_steps):
        cpu.step(field)
        apply_field_decay(field)

        trajectory.append(cpu.pos.copy())
        for k, v in cpu.registers.items():
            reg_history[k].append(float(v))

    trajectory = np.array(trajectory)   # shape (n_steps+1, 2)
    return initial_field, field, trajectory, reg_history, attractors, repulsors

# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _field_intensity(field: np.ndarray) -> np.ndarray:
    """Sum all non-decay operation weights into a scalar intensity map."""
    return field[:, :, :IDX_DECAY].sum(axis=2)


def visualise(initial_field, final_field, trajectory, reg_history,
              attractors, repulsors, save_path: str = "ifa_simulation.png"):
    """
    Render four panels:
      (a) Initial instruction field heatmap — shows the programmed regions.
      (b) CPU trajectory overlaid on the field — demonstrates emergent path.
      (c) Accumulator register over time     — arithmetic emergence.
      (d) Flag register over time            — jump / checkpoint events.
    """
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    intensity = _field_intensity(initial_field)

    # ── (a) Initial field heatmap ────────────────────────────────────────────
    ax_field = fig.add_subplot(gs[0, :2])
    im = ax_field.imshow(intensity, origin="lower", cmap="plasma",
                         extent=[0, GRID_W, 0, GRID_H], aspect="auto")
    plt.colorbar(im, ax=ax_field, label="Instruction Magnitude")
    ax_field.set_title("Initial Instruction Field (total operation weight per cell)")
    ax_field.set_xlabel("X")
    ax_field.set_ylabel("Y")

    for ax_pt, ay_pt in attractors:
        ax_field.plot(ax_pt, ay_pt, "g^", ms=10, label="Attractor")
    for rx, ry in repulsors:
        ax_field.plot(rx, ry, "rs", ms=10, label="Repulsor")
    # deduplicate legend labels
    handles, labels = ax_field.get_legend_handles_labels()
    ax_field.legend(dict(zip(labels, handles)).values(),
                    dict(zip(labels, handles)).keys(),
                    loc="upper left")

    # ── (b) CPU trajectory ───────────────────────────────────────────────────
    ax_traj = fig.add_subplot(gs[1, :2])
    ax_traj.imshow(intensity, origin="lower", cmap="plasma", alpha=0.4,
                   extent=[0, GRID_W, 0, GRID_H], aspect="auto")

    steps = np.arange(len(trajectory))
    sc = ax_traj.scatter(trajectory[:, 0], trajectory[:, 1],
                         c=steps, cmap="cool", s=8, zorder=3)
    plt.colorbar(sc, ax=ax_traj, label="Timestep")

    ax_traj.plot(trajectory[0,  0], trajectory[0,  1], "wo", ms=8,
                 label="Start", zorder=4)
    ax_traj.plot(trajectory[-1, 0], trajectory[-1, 1], "w*", ms=12,
                 label="End",   zorder=4)

    ax_traj.set_title("CPU Particle Trajectory Through Instruction Field")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.legend(loc="upper left")

    # ── (c) Accumulator register ─────────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[0, 2])
    ax_acc.plot(reg_history["accumulator"], color="royalblue")
    ax_acc.set_title("Accumulator Over Time")
    ax_acc.set_xlabel("Step")
    ax_acc.set_ylabel("Value")

    # ── (d) Flag register ────────────────────────────────────────────────────
    ax_flag = fig.add_subplot(gs[1, 2])
    ax_flag.plot(reg_history["flag"], color="darkorange")
    ax_flag.set_title("Flag Register Over Time\n(captures jump checkpoints)")
    ax_flag.set_xlabel("Step")
    ax_flag.set_ylabel("Value")

    fig.suptitle("Instruction Fluid Architecture (IFA) Simulator",
                 fontsize=14, y=1.01)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualisation saved to {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Instruction Fluid Architecture (IFA) Simulator ===")
    print(f"Grid: {GRID_W}×{GRID_H}  |  Steps: {N_STEPS}  |  "
          f"Sample radius: {SAMPLE_RADIUS}")

    (initial_field, final_field,
     trajectory, reg_history,
     attractors, repulsors) = run_simulation()

    print(f"\nCPU start position : {trajectory[0]}")
    print(f"CPU final position : {trajectory[-1]}")
    print(f"Final accumulator  : {reg_history['accumulator'][-1]:.4f}")
    print(f"Final flag         : {reg_history['flag'][-1]:.4f}")
    print(f"Total jumps fired  : "
          f"{sum(1 for i in range(1, len(reg_history['flag'])) if reg_history['flag'][i] != reg_history['flag'][i-1])}")

    visualise(initial_field, final_field, trajectory, reg_history,
              attractors, repulsors)
