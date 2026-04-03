"""
IFA Minimal Core Engine — v3
============================

Reference implementation of the Instruction Fluid Architecture as defined
in IFA_SPEC_v1.md.

This module contains *only* the four elements of the formal system:

  1. IFAField          — the field tensor Φ  (§2.1, §5)
  2. Particle          — the CPU particle (P, S)  (§2.2, §4)
  3. sample()          — the sampling operator σ  (§3)  [exposed via Particle]
  4. update()          — the execution rule  (§4)  [exposed via Particle]

One measurement layer is built in (§7.1):

  5. EnergyLedger      — per-step energy injection vs decay ratio

No visualisation, no v2 experimental artefacts, no debug helpers.
The v2 simulator (ifa_simulator.py) and analysis module (ifa_analysis.py)
are the correct places for those concerns.

Usage
-----
    from ifa_core import IFAField, Particle, EnergyLedger, run_core

    field     = IFAField(width=50, height=50)
    phi       = field.make_potential(wells=[(46,10,8.0),(46,40,8.0)],
                                     hills=[(25,25,3.5,5.0)])
    particles = [Particle(3.0, y) for y in (8, 18, 25, 35, 44)]
    ledger    = EnergyLedger()
    run_core(field, particles, phi, steps=250, ledger=ledger)
    ledger.summary()
"""

from __future__ import annotations

import math
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# OP-ATOM INDICES  (§2.1)
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = 0   # arithmetic intensity
BIND      = 1   # checkpoint / scar trigger
FLOW_X    = 2   # horizontal movement impulse
FLOW_Y    = 3   # vertical movement impulse
SPLIT     = 4   # energy divergence — CPU deposits into field
COLLAPSE  = 5   # energy convergence — CPU absorbs from field
AMPLIFY   = 6   # scale factor for all atoms this step
DECAY_CH  = 7   # per-cell intrinsic decay rate (static)
N_OPS     = 8

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (§4, §5, §7)
# ─────────────────────────────────────────────────────────────────────────────

# Firing thresholds
T_BIND     = 0.05
T_SPLIT    = 0.10
T_COLLAPSE = 0.10
T_AMPLIFY  = 0.08

# Field mutation
INJECT_SCALE     = 0.003  # γ_mut in §5.1
INJECT_STEP_CAP  = 0.08   # per-step injection ceiling
TRANSFORM_CAP    = 2.5    # cap in §5.1 and §5.2 (SPLIT deposit)
SCAR_STRENGTH    = 0.06   # s_strength in §5.2
SCAR_RADIUS      = 2      # r_scar in §5.2

# Field decay  (§5.3)
GLOBAL_DECAY     = 0.0015   # γ_0

# Numerical stability guard for energy ratio denominators
ENERGY_EPSILON   = 1e-12

# Temporal echo  (§2.3)
ECHO_AMPLITUDE   = 0.20     # A
ECHO_PERIOD      = 70       # T

# Movement  (§4 step 6)
POT_WEIGHT       = 0.25     # κ

# Default sample radius  (§3)
SAMPLE_RADIUS    = 3


# ─────────────────────────────────────────────────────────────────────────────
# 1. IFAField — field tensor Φ  (§2.1, §5)
# ─────────────────────────────────────────────────────────────────────────────

class IFAField:
    """
    The instruction field Φ : ℤ² × {0…K−1} → ℝ≥0.

    Holds the mutable op-weight tensor and exposes three mutation operators
    that correspond exactly to §5 of IFA_SPEC_v1:
      - inject(pos, acc)        — §5.1 transform strengthening
      - scar(pos)               — §5.2 bind memory trace
      - decay()                 — §5.3 global field weakening

    The DECAY_CH channel is set at construction and never modified.

    Parameters
    ----------
    width, height : int
        Grid dimensions.  Default instruction layout: left half is
        transform/collapse-weighted (arithmetic zone), right half is
        flow/bind/split/amplify-weighted (control-flow zone).
    seed : int
        Random seed for deterministic initialisation.
    """

    def __init__(self, width: int = 50, height: int = 50, seed: int = 42) -> None:
        self.width   = width
        self.height  = height
        rng          = np.random.default_rng(seed=seed)
        self.data    = np.zeros((height, width, N_OPS), dtype=float)
        half         = width // 2

        # Arithmetic zone (left half)
        self.data[:, :half, TRANSFORM] = rng.uniform(0.4, 1.0,  (height, half))
        self.data[:, :half, COLLAPSE]  = rng.uniform(0.05, 0.2, (height, half))

        # Control-flow zone (right half)
        self.data[:, half:, FLOW_X]  = rng.uniform(-0.4,  0.6,  (height, width - half))
        self.data[:, half:, FLOW_Y]  = rng.uniform(-0.4,  0.6,  (height, width - half))
        self.data[:, half:, BIND]    = rng.uniform( 0.0,  0.22, (height, width - half))
        self.data[:, half:, SPLIT]   = rng.uniform( 0.0,  0.18, (height, width - half))
        self.data[:, half:, AMPLIFY] = rng.uniform( 0.0,  0.14, (height, width - half))

        # Static per-cell decay rates
        self.data[:, :, DECAY_CH] = rng.uniform(0.0, 0.002, (height, width))

        # Base rightward flow (program-counter advance analogue)
        self.data[:, :, FLOW_X] += 0.13

    # ── §2.3  Temporal echo ──────────────────────────────────────────────────

    def effective(self, phase: np.ndarray, t: int) -> np.ndarray:
        """
        Return the temporally-echoed effective field at step t.

            Φ_eff(x,y,t) = Φ(x,y) · [1 + A·sin(2πt/T + φ(x,y))]   k ≠ DECAY_CH

        Phase shape: (height, width).
        """
        osc = 1.0 + ECHO_AMPLITUDE * np.sin(
            2.0 * np.pi * t / ECHO_PERIOD + phase
        )                                            # (H, W)
        eff = self.data * osc[:, :, np.newaxis]      # broadcast over N_OPS
        eff[:, :, DECAY_CH] = self.data[:, :, DECAY_CH]
        return eff

    # ── §5  Field mutation rules ─────────────────────────────────────────────

    def inject(self, ix: int, iy: int, acc: float) -> None:
        """§5.1 — Accumulator-driven transform strengthening at cell (ix, iy)."""
        delta = min(INJECT_SCALE * abs(acc), INJECT_STEP_CAP)
        self.data[iy, ix, TRANSFORM] = min(
            self.data[iy, ix, TRANSFORM] + delta, TRANSFORM_CAP
        )

    def scar(self, ix: int, iy: int) -> None:
        """§5.2 — Bind event: carve inhibitory scar around cell (ix, iy)."""
        iy_lo = max(0,             iy - SCAR_RADIUS)
        iy_hi = min(self.height-1, iy + SCAR_RADIUS)
        ix_lo = max(0,             ix - SCAR_RADIUS)
        ix_hi = min(self.width -1, ix + SCAR_RADIUS)
        self.data[iy_lo:iy_hi+1, ix_lo:ix_hi+1, BIND] -= SCAR_STRENGTH
        np.clip(self.data[:, :, BIND], 0.0, None, out=self.data[:, :, BIND])

    def decay(self) -> None:
        """§5.3 — Per-step global field weakening (all channels except DECAY_CH)."""
        gamma = GLOBAL_DECAY + self.data[:, :, DECAY_CH]    # (H, W)
        factor = 1.0 - gamma                                  # (H, W)
        for k in range(N_OPS - 1):                            # skip DECAY_CH
            self.data[:, :, k] *= factor
        np.clip(self.data, 0.0, None, out=self.data)

    # ── Potential field factory  (§2.4) ──────────────────────────────────────

    def make_potential(self,
                       wells: list[tuple],
                       hills: list[tuple]) -> np.ndarray:
        """
        Build and return a static scalar potential Ψ(x,y) — shape (H, W).

        wells : [(x, y, strength), ...]
        hills : [(x, y, strength, radius), ...]
        """
        gy, gx = np.mgrid[0:self.height, 0:self.width].astype(float)
        psi    = np.zeros((self.height, self.width))
        for wx, wy, ws in wells:
            psi -= ws / (np.hypot(gx - wx, gy - wy) + 1.0)
        for hx, hy, hs, hr in hills:
            dist = np.hypot(gx - hx, gy - hy)
            psi += hs * np.exp(-0.5 * (dist / hr) ** 2)
        return psi

    # ── Phase field factory ──────────────────────────────────────────────────

    def make_phase(self, seed: int = 7) -> np.ndarray:
        """Return per-cell phase offsets φ(x,y) ~ Uniform[0, 2π) — shape (H,W)."""
        return np.random.default_rng(seed=seed).uniform(
            0.0, 2.0 * math.pi, (self.height, self.width)
        )

    # ── Energy (§6.1) ────────────────────────────────────────────────────────

    def energy(self) -> float:
        """E(Φ) = Σ_{x,y,k≠DECAY} Φ(x,y)[k]  (§6.1)."""
        return float(self.data[:, :, :DECAY_CH].sum())


# ─────────────────────────────────────────────────────────────────────────────
# 3 + 4.  Sampling operator σ  and  Particle (P, S)  (§3, §4)
# ─────────────────────────────────────────────────────────────────────────────

class Particle:
    """
    CPU particle — the formal (P, S) pair from §2.2.

    Public methods match the execution rule pipeline (§4) exactly.
    All state is stored in the particle; the field is passed in per-call
    so that multi-particle scenarios share one field object.
    """

    def __init__(self, x0: float, y0: float) -> None:
        self.pos     = np.array([x0, y0], dtype=float)
        self.acc     = 0.0     # accumulator
        self.flag    = 0.0     # bind checkpoint register
        self.energy  = 1.0    # CPU vitality
        self.counter = 0       # step count

    # ── §3  Sampling operator ─────────────────────────────────────────────────

    def sample(self, eff: np.ndarray, radius: int = SAMPLE_RADIUS) -> np.ndarray:
        """
        σ(P, Φ_eff, r) — inverse-distance-weighted neighbourhood sample.

        Returns a K-vector of op-atom weights.
        """
        H, W   = eff.shape[:2]
        cx, cy = self.pos
        imp    = np.zeros(N_OPS)
        total  = 0.0

        for gy in range(max(0, int(cy) - radius),
                        min(H, int(cy) + radius + 1)):
            for gx in range(max(0, int(cx) - radius),
                            min(W, int(cx) + radius + 1)):
                d = math.hypot(gx - cx, gy - cy) + 1e-6
                if d - 1e-6 > radius:
                    continue
                w      = 1.0 / d
                imp   += w * eff[gy, gx]
                total += w

        if total > 0.0:
            imp /= total
        return imp

    # ── §4  Execution rule ────────────────────────────────────────────────────

    def update(self,
               field: IFAField,
               phi:   np.ndarray,
               phase: np.ndarray,
               t:     int) -> None:
        """
        Execute one simulation timestep (§4 pipeline, steps 1–6).

        Mutates both `self` (CPU state P, S) and `field.data` (Φ).

        Parameters
        ----------
        field : IFAField   — the shared mutable instruction field
        phi   : np.ndarray — static potential Ψ, shape (H, W)
        phase : np.ndarray — static echo phases φ, shape (H, W)
        t     : int        — current timestep (for temporal echo)
        """
        H, W = field.height, field.width

        # §4 step 1 — effective field
        eff = field.effective(phase, t)

        # §4 step 2 — sample
        ι = self.sample(eff)

        # §4 step 3 — amplify factor
        α = 1.0 + ι[AMPLIFY] if ι[AMPLIFY] > T_AMPLIFY else 1.0

        # §4 step 4 — semantic atoms
        ix = int(np.clip(self.pos[0], 0, W - 1))
        iy = int(np.clip(self.pos[1], 0, H - 1))

        # TRANSFORM
        self.acc += α * ι[TRANSFORM]

        # BIND
        if ι[BIND] > T_BIND:
            self.flag = self.acc
            self.acc  = 0.0
            field.scar(ix, iy)                        # §5.2

        # SPLIT
        if ι[SPLIT] > T_SPLIT:
            deposit = min(0.025 * α, 0.1)
            field.data[iy, ix, TRANSFORM] = min(
                field.data[iy, ix, TRANSFORM] + deposit, TRANSFORM_CAP
            )
            self.energy = max(self.energy - 0.02, 0.1)

        # COLLAPSE
        if ι[COLLAPSE] > T_COLLAPSE:
            self.acc    += 0.4 * field.data[iy, ix, TRANSFORM] * α
            self.energy  = min(self.energy + 0.02, 2.0)

        self.counter += 1

        # §4 step 5 — inject
        field.inject(ix, iy, self.acc)                # §5.1

        # §4 step 6 — movement
        grad   = _potential_grad(phi, self.pos)
        self.pos[0] = np.clip(
            self.pos[0] + ι[FLOW_X] - POT_WEIGHT * grad[0], 0.0, float(W - 1)
        )
        self.pos[1] = np.clip(
            self.pos[1] + ι[FLOW_Y] - POT_WEIGHT * grad[1], 0.0, float(H - 1)
        )


def _potential_grad(phi: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Central finite-difference gradient of Ψ at continuous position pos."""
    H, W = phi.shape
    ix   = int(np.clip(round(pos[0]), 1, W - 2))
    iy   = int(np.clip(round(pos[1]), 1, H - 2))
    return np.array([
        (phi[iy, ix + 1] - phi[iy, ix - 1]) / 2.0,
        (phi[iy + 1, ix] - phi[iy - 1, ix]) / 2.0,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 5. EnergyLedger — measurement layer  (§7.1)
# ─────────────────────────────────────────────────────────────────────────────

class EnergyLedger:
    """
    Per-step energy injection vs decay ratio  (§7.1).

    At each step the ledger records:
      - r(t)  = E_after_injection / E_before_injection  (injection ratio)
      - d(t)  = E_after_decay     / E_after_injection   (decay ratio)

    These two ratios fully characterise the energy budget at each step:
      E(t+1) = E(t) · r(t) · d(t)

    Stability criterion (§7.1):
      A run is energy-stable if the time-averaged injection ratio
          r̄ = mean(r) < 1 / (1 − γ_eff)
      i.e., the injection gain does not outrun the decay loss.

    Usage
    -----
        ledger = EnergyLedger()
        # inside loop:
        E_before = field.energy()
        for p in particles: p.update(field, phi, phase, t)
        E_mid    = field.energy()
        field.decay()
        E_after  = field.energy()
        ledger.record(E_before, E_mid, E_after)
        # after loop:
        ledger.summary()
    """

    def __init__(self) -> None:
        self.injection_ratios: list[float] = []   # r(t)
        self.decay_ratios:     list[float] = []   # d(t)
        self.energy_series:    list[float] = []   # E(t) after full step

    def record(self, E_before: float, E_mid: float, E_after: float) -> None:
        """
        Record one timestep.

        Parameters
        ----------
        E_before : field energy before CPUs step
        E_mid    : field energy after CPUs step, before decay
        E_after  : field energy after decay
        """
        r = E_mid   / E_before if E_before > ENERGY_EPSILON else 1.0
        d = E_after / E_mid    if E_mid    > ENERGY_EPSILON else 1.0
        self.injection_ratios.append(r)
        self.decay_ratios.append(d)
        self.energy_series.append(E_after)

    def summary(self) -> None:
        """Print a concise stability report to stdout."""
        if not self.injection_ratios:
            print("EnergyLedger: no data recorded.")
            return

        r_arr = np.array(self.injection_ratios)
        d_arr = np.array(self.decay_ratios)
        E_arr = np.array(self.energy_series)

        r_mean = float(r_arr.mean())
        d_mean = float(d_arr.mean())
        threshold = 1.0 / (1.0 - GLOBAL_DECAY)
        stable = r_mean < threshold

        print("─" * 52)
        print("EnergyLedger  (§7.1 — injection vs decay ratio)")
        print("─" * 52)
        print(f"  Steps recorded          : {len(r_arr)}")
        print(f"  Initial energy E(0)     : {E_arr[0] if len(E_arr) > 0 else 'N/A':.3f}")
        print(f"  Final energy   E(T)     : {E_arr[-1]:.3f}")
        print(f"  Pure-decay pred E(T)    : "
              f"{E_arr[0] * (1.0 - GLOBAL_DECAY)**len(r_arr):.3f}")
        print()
        print(f"  Injection ratio r̄       : {r_mean:.6f}")
        print(f"  Stability threshold     : {threshold:.6f}")
        print(f"  System is              : {'STABLE ✓' if stable else 'DIVERGENT ✗'}")
        print()
        print(f"  Decay ratio d̄          : {d_mean:.6f}")
        print(f"  Net budget r̄·d̄         : {r_mean * d_mean:.6f}  "
              f"({'loss' if r_mean * d_mean < 1 else 'gain'})")
        print()
        print(f"  r(t) min / max          : {r_arr.min():.6f} / {r_arr.max():.6f}")
        print(f"  r(t) std                : {r_arr.std():.6f}")
        print(f"  Steps with r > 1 (gain) : "
              f"{int((r_arr > 1.0).sum())} / {len(r_arr)}")
        print("─" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_core(field:     IFAField,
             particles: list[Particle],
             phi:       np.ndarray,
             steps:     int = 250,
             ledger:    EnergyLedger | None = None) -> None:
    """
    Run the minimal IFA simulation loop.

    Corresponds to Algorithm 1 in IFA_SPEC_v1 §4, with EnergyLedger
    instrumentation when provided.

    Parameters
    ----------
    field     : IFAField       — shared mutable instruction field
    particles : list[Particle] — one or more CPU particles
    phi       : np.ndarray     — static potential Ψ, shape (H, W)
    steps     : int            — number of timesteps to run
    ledger    : EnergyLedger   — optional measurement; pass None to skip
    """
    phase = field.make_phase()

    for t in range(steps):
        E_before = field.energy() if ledger is not None else 0.0

        for p in particles:
            p.update(field, phi, phase, t)

        E_mid = field.energy() if ledger is not None else 0.0
        field.decay()
        E_after = field.energy() if ledger is not None else 0.0

        if ledger is not None:
            ledger.record(E_before, E_mid, E_after)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== IFA Minimal Core Engine — v3 ===\n")

    field = IFAField(width=50, height=50, seed=42)
    phi   = field.make_potential(
        wells=[(46, 10, 8.0), (46, 40, 8.0)],
        hills=[(25, 25, 3.5, 5.0)],
    )
    particles = [Particle(3.0, float(y)) for y in (8, 18, 25, 35, 44)]
    ledger    = EnergyLedger()

    print(f"Grid     : {field.width}×{field.height}")
    print(f"Particles: {len(particles)}")
    print(f"Steps    : 250")
    print(f"E(0)     : {field.energy():.3f}\n")

    run_core(field, particles, phi, steps=250, ledger=ledger)

    print("--- Per-particle summary ---")
    for i, p in enumerate(particles):
        print(f"  P{i}: pos=({p.pos[0]:.1f}, {p.pos[1]:.1f})"
              f"  acc={p.acc:.3f}  flag={p.flag:.3f}"
              f"  energy={p.energy:.3f}  steps={p.counter}")
    print()

    ledger.summary()
