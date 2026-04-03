"""
IFA Field Primitive Library, Composition API, and Program Compiler
==================================================================

This module defines the programmatic layer that sits on top of the
IFA Minimal Core Engine (ifa_core.py).  It provides:

  1. Field Primitive Library  — five canonical field building blocks
  2. Field Composition API    — compose() stamps primitives onto a field
  3. Field Compiler           — compile_program() turns an instruction list
                                into a ready-to-run IFAField + potential Ψ
  4. Three Canonical Programs — parity_checker, binary_counter,
                                signal_propagator
  5. ProgramObserver          — convergence, trajectory clustering,
                                energy utilisation metrics

Usage (quick start)
-------------------
    from ifa_field_compiler import compile_program, ProgramObserver, PROGRAMS
    from ifa_core import Particle, run_core

    field, phi, start_pos = compile_program(PROGRAMS["parity_checker"])
    particles = [Particle(x, y) for x, y in start_pos]
    observer  = ProgramObserver(field, particles)
    run_core(field, particles, phi, steps=300, ledger=observer.ledger)
    observer.finalise(field, particles)
    observer.report()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np

from ifa_core import (
    IFAField, Particle, EnergyLedger, run_core,
    TRANSFORM, BIND, FLOW_X, FLOW_Y, SPLIT, COLLAPSE, AMPLIFY, DECAY_CH,
    N_OPS, TRANSFORM_CAP, ENERGY_EPSILON,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FIELD PRIMITIVE LIBRARY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldPrimitive:
    """
    A callable that writes op-weights into a field patch.

    name  : human-readable label used in compiler output
    apply : function(field_data: np.ndarray) -> None
            Writes directly into field.data in-place.
    meta  : arbitrary metadata (position, radius, etc.) for introspection
    """
    name:  str
    apply: Callable[[np.ndarray], None]
    meta:  dict = dc_field(default_factory=dict)

    def __call__(self, field_data: np.ndarray) -> None:
        self.apply(field_data)


def well(x: float, y: float, strength: float = 8.0,
         radius: float = 6.0) -> FieldPrimitive:
    """
    Attractor Well — pulls CPUs toward (x, y).

    Writes a strong negative-potential region by setting FLOW_X / FLOW_Y
    to point inward and placing a moderate COLLAPSE weight so that CPUs
    near the well absorb local field energy (convergence / focus).

    Parameters
    ----------
    x, y     : well centre (grid coordinates)
    strength : magnitude of the inward flow impulse
    radius   : cell radius over which the well acts
    """
    def _apply(data: np.ndarray) -> None:
        H, W = data.shape[:2]
        for gy in range(H):
            for gx in range(W):
                dx   = x - gx
                dy   = y - gy
                dist = math.hypot(dx, dy) + 1e-6
                if dist > radius:
                    continue
                # Inward flow — decays with distance²
                weight = strength / (dist ** 2 + 1.0)
                data[gy, gx, FLOW_X]   += weight * dx / dist
                data[gy, gx, FLOW_Y]   += weight * dy / dist
                data[gy, gx, COLLAPSE] += 0.15 * (1.0 - dist / radius)

    return FieldPrimitive(
        name="well",
        apply=_apply,
        meta={"x": x, "y": y, "strength": strength, "radius": radius},
    )


def flow_channel(start: tuple[float, float],
                 end:   tuple[float, float],
                 width: float = 3.0,
                 speed: float = 0.5) -> FieldPrimitive:
    """
    Flow Channel — directional vector bias along a line segment.

    Writes FLOW_X / FLOW_Y pointing from *start* toward *end* for all
    cells within *width* cells of the line segment.  Used to build
    movement corridors and sequencing paths.

    Parameters
    ----------
    start, end : (x, y) endpoints of the channel
    width      : half-width of the corridor in cells
    speed      : magnitude of the flow impulse
    """
    sx, sy = start
    ex, ey = end
    seg_dx = ex - sx
    seg_dy = ey - sy
    seg_len = math.hypot(seg_dx, seg_dy) + 1e-6
    # unit direction
    ux = seg_dx / seg_len
    uy = seg_dy / seg_len

    def _apply(data: np.ndarray) -> None:
        H, W = data.shape[:2]
        for gy in range(H):
            for gx in range(W):
                # distance from gx,gy to the line segment
                t = ((gx - sx) * seg_dx + (gy - sy) * seg_dy) / (seg_len ** 2)
                t = max(0.0, min(1.0, t))
                closest_x = sx + t * seg_dx
                closest_y = sy + t * seg_dy
                dist = math.hypot(gx - closest_x, gy - closest_y)
                if dist > width:
                    continue
                fade = 1.0 - dist / width
                data[gy, gx, FLOW_X] += speed * ux * fade
                data[gy, gx, FLOW_Y] += speed * uy * fade

    return FieldPrimitive(
        name="flow_channel",
        apply=_apply,
        meta={"start": start, "end": end, "width": width, "speed": speed},
    )


def transform_zone(cx: float, cy: float,
                   radius: float = 5.0,
                   intensity: float = 0.8) -> FieldPrimitive:
    """
    Transform Zone — high TRANSFORM weight region.

    CPUs passing through receive large accumulator increments — analogous
    to performing arithmetic.  Also raises COLLAPSE slightly so that CPUs
    can absorb the accumulated energy.

    Parameters
    ----------
    cx, cy    : zone centre
    radius    : cell radius
    intensity : TRANSFORM weight written (capped at TRANSFORM_CAP)
    """
    def _apply(data: np.ndarray) -> None:
        H, W = data.shape[:2]
        for gy in range(H):
            for gx in range(W):
                dist = math.hypot(gx - cx, gy - cy)
                if dist > radius:
                    continue
                fade = 1.0 - dist / radius
                new_val = intensity * fade
                data[gy, gx, TRANSFORM] = min(
                    data[gy, gx, TRANSFORM] + new_val, TRANSFORM_CAP
                )
                data[gy, gx, COLLAPSE] += 0.1 * fade

    return FieldPrimitive(
        name="transform_zone",
        apply=_apply,
        meta={"cx": cx, "cy": cy, "radius": radius, "intensity": intensity},
    )


def bind_gate(cx: float, cy: float,
              radius: float = 4.0,
              threshold: float = 0.6) -> FieldPrimitive:
    """
    Bind Gate — threshold-triggered checkpoint region.

    Writes a high BIND weight disk.  When a CPU's sampled BIND impulse
    exceeds T_BIND (0.05), it snapshots its accumulator → flag and resets.
    A strong BIND zone reliably forces this checkpoint.

    Parameters
    ----------
    cx, cy    : gate centre
    radius    : gate radius
    threshold : BIND weight to write (should be >> T_BIND = 0.05)
    """
    def _apply(data: np.ndarray) -> None:
        H, W = data.shape[:2]
        for gy in range(H):
            for gx in range(W):
                dist = math.hypot(gx - cx, gy - cy)
                if dist > radius:
                    continue
                fade = 1.0 - dist / radius
                data[gy, gx, BIND] += threshold * fade

    return FieldPrimitive(
        name="bind_gate",
        apply=_apply,
        meta={"cx": cx, "cy": cy, "radius": radius, "threshold": threshold},
    )


def split_collapse_region(cx: float, cy: float,
                          radius: float = 5.0,
                          mode: str = "split") -> FieldPrimitive:
    """
    Split / Collapse Region — branching or merging zone.

    mode="split"   : high SPLIT weight → CPUs deposit energy into the
                     field (divergence / branching signal).
    mode="collapse": high COLLAPSE weight → CPUs absorb local field
                     energy (convergence / merging signal).
    mode="both"    : balanced SPLIT + COLLAPSE → mixing zone.

    Parameters
    ----------
    cx, cy : region centre
    radius : cell radius
    mode   : "split", "collapse", or "both"
    """
    if mode not in ("split", "collapse", "both"):
        raise ValueError(f"mode must be 'split', 'collapse', or 'both'; got {mode!r}")

    def _apply(data: np.ndarray) -> None:
        H, W = data.shape[:2]
        for gy in range(H):
            for gx in range(W):
                dist = math.hypot(gx - cx, gy - cy)
                if dist > radius:
                    continue
                fade = 1.0 - dist / radius
                if mode in ("split", "both"):
                    data[gy, gx, SPLIT] += 0.25 * fade
                if mode in ("collapse", "both"):
                    data[gy, gx, COLLAPSE] += 0.25 * fade

    return FieldPrimitive(
        name=f"split_collapse_region[{mode}]",
        apply=_apply,
        meta={"cx": cx, "cy": cy, "radius": radius, "mode": mode},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FIELD COMPOSITION API
# ─────────────────────────────────────────────────────────────────────────────

def compose(field: IFAField, *primitives: FieldPrimitive) -> IFAField:
    """
    Stamp one or more FieldPrimitives onto *field* in order.

    Each primitive writes directly into field.data; later primitives
    accumulate on top of earlier ones.  Values are clipped to [0, ∞)
    after all primitives have been applied.

    Parameters
    ----------
    field      : IFAField to mutate in-place
    primitives : any number of FieldPrimitive objects

    Returns
    -------
    The same field object (mutated), for chaining.

    Example
    -------
        field = IFAField(width=60, height=60, seed=0)
        compose(
            field,
            well(x=50, y=30, strength=10.0),
            flow_channel((0,30), (50,30), width=4, speed=0.6),
            transform_zone(cx=20, cy=30, radius=6, intensity=0.9),
            bind_gate(cx=35, cy=30, radius=3, threshold=0.8),
        )
    """
    for prim in primitives:
        prim(field.data)
    np.clip(field.data, 0.0, None, out=field.data)
    return field


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIELD COMPILER
# ─────────────────────────────────────────────────────────────────────────────

# Instruction opcodes understood by the compiler
_OPCODE_WELL              = "WELL"
_OPCODE_FLOW              = "FLOW"
_OPCODE_TRANSFORM         = "TRANSFORM"
_OPCODE_BIND              = "BIND"
_OPCODE_SPLIT             = "SPLIT"
_OPCODE_COLLAPSE          = "COLLAPSE"
_OPCODE_START             = "START"     # declare a CPU start position
_OPCODE_TARGET_WELL       = "TARGET_WELL"  # add a well to the Ψ potential

@dataclass
class CompiledProgram:
    """
    Output of compile_program().

    field       : IFAField ready to run
    phi         : static potential Ψ, shape (H, W)
    start_pos   : list of (x, y) CPU start positions
    primitives  : ordered list of FieldPrimitive that were applied
    source      : original instruction list (for introspection)
    """
    field:      IFAField
    phi:        np.ndarray
    start_pos:  list[tuple[float, float]]
    primitives: list[FieldPrimitive]
    source:     list[tuple]


def compile_program(instructions: list[tuple],
                    width:  int = 60,
                    height: int = 60,
                    seed:   int = 0) -> CompiledProgram:
    """
    Compile a list of field instructions into a runnable IFA program.

    Instruction format
    ------------------
    Each instruction is a tuple whose first element is an opcode string
    and whose remaining elements are keyword arguments (as a dict) OR
    positional arguments:

        ("WELL",      {"x": 50, "y": 10, "strength": 8.0})
        ("FLOW",      {"start": (0, 25), "end": (50, 25), "speed": 0.5})
        ("TRANSFORM", {"cx": 20, "cy": 10, "radius": 6, "intensity": 0.9})
        ("BIND",      {"cx": 35, "cy": 10, "radius": 4, "threshold": 0.7})
        ("SPLIT",     {"cx": 15, "cy": 10, "radius": 4})
        ("COLLAPSE",  {"cx": 45, "cy": 10, "radius": 4})
        ("START",     {"x": 3.0, "y": 10.0})         # CPU start position
        ("TARGET_WELL", {"x": 50, "y": 10, "strength": 8.0})  # adds to Ψ

    Returns
    -------
    CompiledProgram

    Example
    -------
        prog = compile_program([
            ("TARGET_WELL", {"x": 55, "y": 30, "strength": 10.0}),
            ("FLOW",   {"start": (0,30), "end": (55,30)}),
            ("TRANSFORM", {"cx": 20, "cy": 30, "radius": 5, "intensity": 0.8}),
            ("BIND",   {"cx": 40, "cy": 30, "radius": 3}),
            ("START",  {"x": 3.0, "y": 30.0}),
        ])
        p   = Particle(*prog.start_pos[0])
        run_core(prog.field, [p], prog.phi, steps=200)
    """
    field      = IFAField(width=width, height=height, seed=seed)
    primitives: list[FieldPrimitive] = []
    start_pos:  list[tuple[float, float]] = []
    wells_psi:  list[tuple] = []    # (x, y, strength)
    hills_psi:  list[tuple] = []

    for instr in instructions:
        opcode = instr[0].upper()
        params: dict = instr[1] if len(instr) > 1 else {}

        if opcode == _OPCODE_WELL:
            prim = well(**params)
            primitives.append(prim)

        elif opcode == _OPCODE_FLOW:
            prim = flow_channel(**params)
            primitives.append(prim)

        elif opcode == _OPCODE_TRANSFORM:
            prim = transform_zone(**params)
            primitives.append(prim)

        elif opcode == _OPCODE_BIND:
            prim = bind_gate(**params)
            primitives.append(prim)

        elif opcode == _OPCODE_SPLIT:
            prim = split_collapse_region(mode="split", **params)
            primitives.append(prim)

        elif opcode == _OPCODE_COLLAPSE:
            prim = split_collapse_region(mode="collapse", **params)
            primitives.append(prim)

        elif opcode == _OPCODE_START:
            start_pos.append((float(params["x"]), float(params["y"])))

        elif opcode == _OPCODE_TARGET_WELL:
            wells_psi.append((params["x"], params["y"],
                              params.get("strength", 8.0)))

        else:
            raise ValueError(f"Unknown opcode: {opcode!r}")

    # Stamp all primitives onto the blank field
    compose(field, *primitives)

    # Build static potential Ψ from TARGET_WELL declarations
    phi = field.make_potential(wells=wells_psi, hills=[])

    return CompiledProgram(
        field=field,
        phi=phi,
        start_pos=start_pos if start_pos else [(3.0, float(height // 2))],
        primitives=primitives,
        source=instructions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THREE CANONICAL PROGRAMS
# ─────────────────────────────────────────────────────────────────────────────

# Each program is stored as a list of compiler instructions.
# Pass to compile_program() to instantiate.

PROGRAMS: dict[str, list[tuple]] = {}

# ── A. Parity Checker ──────────────────────────────────────────────────────
# Layout (width=60, height=40):
#
#   Transform zones at x=10,20 (read high=1 / low=0 cells)
#   Bind gates at x=15,25 (checkpoint after each bit read)
#   Flow channel along y=20 driving right
#   Well at x=55 (convergence / halt attractor)
#
# A CPU entering this field reads alternating transform zones,
# checkpoints its accumulator at each bind gate (simulating state
# transitions of the parity TM), and converges to the well when it
# passes x=50 (blank / end-of-tape).
PROGRAMS["parity_checker"] = [
    # Attractor well (halt state)
    ("TARGET_WELL", {"x": 55, "y": 20, "strength": 10.0}),
    # Main rightward flow corridor
    ("FLOW",      {"start": (0, 20), "end": (55, 20), "speed": 0.55}),
    # Bit-1 region  (high TRANSFORM = "1")
    ("TRANSFORM", {"cx": 10, "cy": 20, "radius": 4, "intensity": 0.9}),
    # Checkpoint after bit 1
    ("BIND",      {"cx": 15, "cy": 20, "radius": 3, "threshold": 0.7}),
    # Bit-0 region  (low TRANSFORM = "0" — inherits field background ≈ 0)
    # Checkpoint after bit 0
    ("BIND",      {"cx": 25, "cy": 20, "radius": 3, "threshold": 0.7}),
    # Bit-1 region repeated (tape: 1 0 1 → odd parity)
    ("TRANSFORM", {"cx": 35, "cy": 20, "radius": 4, "intensity": 0.9}),
    # Final checkpoint
    ("BIND",      {"cx": 40, "cy": 20, "radius": 3, "threshold": 0.7}),
    # End-of-tape: collapse into well
    ("COLLAPSE",  {"cx": 50, "cy": 20, "radius": 4}),
    # CPU start
    ("START",     {"x": 2.0, "y": 20.0}),
]

# ── B. Binary Counter ──────────────────────────────────────────────────────
# Layout (width=60, height=40):
#
#   A circular loop field that forces the CPU to orbit, accumulating
#   transform energy on each pass (incrementing a binary counter in
#   the accumulator register).  A bind gate at the loop apex
#   snapshots each count.  After N orbits the CPU spirals out to the
#   well attractor.
#
#   Orbit is built from four flow-channel segments forming a rectangle:
#     right → down → left → up → right…
#   A strong transform zone at the top-right corner adds +1 per orbit.
#   The well at the centre is weak (κ·POT_WEIGHT small) so it only
#   captures after enough orbits have weakened the orbit energy.
PROGRAMS["binary_counter"] = [
    # Weak central well (captures after ~8 orbits when orbit energy depletes)
    ("TARGET_WELL", {"x": 30, "y": 20, "strength": 3.0}),
    # Orbit: right along top
    ("FLOW", {"start": (10, 10), "end": (50, 10), "speed": 0.5}),
    # Orbit: down on right
    ("FLOW", {"start": (50, 10), "end": (50, 35), "speed": 0.5}),
    # Orbit: left along bottom
    ("FLOW", {"start": (50, 35), "end": (10, 35), "speed": 0.5}),
    # Orbit: up on left
    ("FLOW", {"start": (10, 35), "end": (10, 10), "speed": 0.5}),
    # Increment zone: top-right corner (+1 per orbit)
    ("TRANSFORM", {"cx": 50, "cy": 10, "radius": 5, "intensity": 0.85}),
    # Bind gate at apex: snapshot count
    ("BIND",      {"cx": 30, "cy": 10, "radius": 4, "threshold": 0.65}),
    # Split at bottom: deposit energy so field remembers orbit history
    ("SPLIT",     {"cx": 30, "cy": 35, "radius": 4}),
    # CPU start (top-left corner, just inside the orbit)
    ("START", {"x": 10.0, "y": 10.0}),
]

# ── C. Signal Propagator ──────────────────────────────────────────────────
# Layout (width=60, height=40):
#
#   A wave-like computation: a series of alternating transform + split
#   zones along the horizontal axis create a chain of energy pulses.
#   Each zone deposits into the field (SPLIT) and boosts the next zone's
#   transform weight via the inject mutation — a travelling-wave
#   computation pattern.
#
#   Three wells are placed at the end; the CPU that carries the most
#   accumulated signal will be captured by a specific well depending
#   on which zones it fired (deterministic wave routing).
PROGRAMS["signal_propagator"] = [
    # Three output wells (signal routing destinations)
    ("TARGET_WELL", {"x": 56, "y": 10, "strength":  8.0}),
    ("TARGET_WELL", {"x": 56, "y": 20, "strength":  8.0}),
    ("TARGET_WELL", {"x": 56, "y": 35, "strength":  8.0}),
    # Main flow — three lanes
    ("FLOW", {"start": (0, 10), "end": (55, 10), "speed": 0.5}),
    ("FLOW", {"start": (0, 20), "end": (55, 20), "speed": 0.5}),
    ("FLOW", {"start": (0, 35), "end": (55, 35), "speed": 0.5}),
    # Wave stage 1: inject pulse
    ("TRANSFORM", {"cx": 10, "cy": 20, "radius": 6, "intensity": 0.8}),
    ("SPLIT",     {"cx": 10, "cy": 20, "radius": 4}),
    # Wave stage 2: amplify
    ("TRANSFORM", {"cx": 25, "cy": 20, "radius": 6, "intensity": 0.9}),
    ("SPLIT",     {"cx": 25, "cy": 20, "radius": 4}),
    # Wave stage 3: route via bind gates to specific wells
    ("BIND",      {"cx": 40, "cy": 10, "radius": 3, "threshold": 0.6}),
    ("BIND",      {"cx": 40, "cy": 20, "radius": 3, "threshold": 0.6}),
    ("BIND",      {"cx": 40, "cy": 35, "radius": 3, "threshold": 0.6}),
    # Collapse before well to lock in the signal
    ("COLLAPSE",  {"cx": 50, "cy": 20, "radius": 5}),
    # Three CPUs — one per lane
    ("START", {"x": 2.0, "y": 10.0}),
    ("START", {"x": 2.0, "y": 20.0}),
    ("START", {"x": 2.0, "y": 35.0}),
]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PROGRAM OBSERVABILITY
# ─────────────────────────────────────────────────────────────────────────────

class ProgramObserver:
    """
    Observability layer for a compiled IFA program run.

    Records three metrics:

    (a) Convergence
        Did each particle reach a well within capture_radius cells?
        Result: per-particle bool + distance to nearest well.

    (b) Trajectory Clustering
        K-means (k=2) on the sequence of per-step positions for each
        particle.  Reports cluster inertia as a dispersion measure:
          low  inertia → tight, focused trajectory (converging)
          high inertia → spread, wandering trajectory (chaotic)

    (c) Energy Utilisation Efficiency
        η = (E_initial − E_final) / E_initial
        Combined with EnergyLedger injection ratio gives a picture of
        how much field energy was consumed by the computation.

    Usage
    -----
        observer = ProgramObserver(field, particles)
        run_core(field, particles, phi, steps=300, ledger=observer.ledger)
        observer.finalise(field, particles)
        observer.report()
    """

    def __init__(self, field: IFAField,
                 particles: list[Particle],
                 wells: list[tuple] | None = None,
                 capture_radius: float = 8.0) -> None:
        self.ledger         = EnergyLedger()
        self.capture_radius = capture_radius
        self.E_initial      = field.energy()
        self.E_final: float = 0.0

        # Extract well positions from the field's stored potential metadata
        # or accept an explicit list of (x, y, strength) tuples.
        self._wells = [(wx, wy) for wx, wy, *_ in (wells or [])]

        # Per-particle trajectory: list of (step × 2) arrays accumulated live
        self._traj: list[list[np.ndarray]] = [[] for _ in particles]

        # Track initial positions for hook
        self._n_particles = len(particles)

    def record_step(self, particles: list[Particle]) -> None:
        """
        Call once per simulation step (after all particles have been updated)
        to record their current positions.
        """
        for i, p in enumerate(particles):
            self._traj[i].append(p.pos.copy())

    def finalise(self, field: IFAField, particles: list[Particle]) -> None:
        """
        Compute final metrics.  Call once after run_core() returns.
        """
        self.E_final = field.energy()

        # ── (a) Convergence ───────────────────────────────────────────────
        self.converged: list[bool]  = []
        self.final_dist: list[float] = []
        for p in particles:
            if self._wells:
                dists = [math.hypot(p.pos[0]-wx, p.pos[1]-wy)
                         for wx, wy in self._wells]
                d_min = min(dists)
            else:
                d_min = float("inf")
            self.converged.append(d_min <= self.capture_radius)
            self.final_dist.append(d_min)

        # ── (b) Trajectory clustering (k-means, k=2) ─────────────────────
        self.cluster_inertia: list[float] = []
        for traj in self._traj:
            if len(traj) < 4:
                self.cluster_inertia.append(float("nan"))
                continue
            pts = np.array(traj)           # (T, 2)
            inertia = _kmeans_inertia(pts, k=2)
            self.cluster_inertia.append(inertia)

        # ── (c) Energy utilisation efficiency ────────────────────────────
        if self.E_initial > ENERGY_EPSILON:
            self.efficiency = (self.E_initial - self.E_final) / self.E_initial
        else:
            self.efficiency = 0.0

    def report(self) -> None:
        """Print a formatted observability report to stdout."""
        print("═" * 56)
        print("ProgramObserver — Run Report")
        print("═" * 56)
        print()

        # (a) Convergence
        print("(a) Convergence")
        print("─" * 40)
        if not self._wells:
            print("  No wells declared; convergence not measured.")
        else:
            for i, (conv, dist) in enumerate(
                    zip(self.converged, self.final_dist)):
                status = "CONVERGED ✓" if conv else "NOT CONVERGED"
                print(f"  P{i}: dist_to_well={dist:.2f}  [{status}]")
            n_conv = sum(self.converged)
            print(f"  → {n_conv}/{self._n_particles} particles converged")
        print()

        # (b) Trajectory clustering
        print("(b) Trajectory Clustering  (k=2 inertia)")
        print("─" * 40)
        for i, inert in enumerate(self.cluster_inertia):
            if math.isnan(inert):
                print(f"  P{i}: insufficient data")
            else:
                regime = ("focused" if inert < 50 else
                          "moderate" if inert < 200 else "diffuse")
                print(f"  P{i}: inertia={inert:.1f}  [{regime}]")
        print()

        # (c) Energy utilisation
        print("(c) Energy Utilisation")
        print("─" * 40)
        print(f"  E_initial : {self.E_initial:.3f}")
        print(f"  E_final   : {self.E_final:.3f}")
        print(f"  Efficiency η = (E₀−Eₜ)/E₀ : {self.efficiency:.4f}  "
              f"({self.efficiency*100:.1f}%)")
        print()

        # EnergyLedger summary
        self.ledger.summary()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _kmeans_inertia(pts: np.ndarray, k: int = 2, n_init: int = 5) -> float:
    """
    Lightweight k-means inertia computation (no sklearn dependency).

    Runs k-means from *n_init* random initialisations and returns the
    lowest inertia (sum of squared distances to nearest centroid).
    """
    best = float("inf")
    rng  = np.random.default_rng(seed=0)
    for _ in range(n_init):
        idx        = rng.choice(len(pts), size=k, replace=False)
        centroids  = pts[idx].astype(float)
        for _it in range(50):                              # max 50 iterations
            dists  = np.array([
                np.sum((pts - c) ** 2, axis=1) for c in centroids
            ])                                             # (k, T)
            labels = dists.argmin(axis=0)                  # (T,)
            new_c  = np.array([
                pts[labels == j].mean(axis=0) if (labels == j).any()
                else centroids[j]
                for j in range(k)
            ])
            if np.allclose(new_c, centroids, atol=1e-6):
                break
            centroids = new_c
        inertia = float(sum(
            np.sum((pts[labels == j] - centroids[j]) ** 2)
            for j in range(k)
            if (labels == j).any()
        ))
        best = min(best, inertia)
    return best


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — run all three canonical programs
# ─────────────────────────────────────────────────────────────────────────────

def _run_canonical(name: str, steps: int = 300) -> None:
    """Compile, run, and report one canonical program."""
    print(f"\n{'━'*56}")
    print(f"  Program: {name}  ({steps} steps)")
    print(f"{'━'*56}")

    prog      = compile_program(PROGRAMS[name])
    particles = [Particle(x, y) for x, y in prog.start_pos]

    # Extract well positions for convergence tracking
    wells_for_obs = [
        (instr[1]["x"], instr[1]["y"], instr[1].get("strength", 8.0))
        for instr in prog.source
        if instr[0].upper() == "TARGET_WELL"
    ]

    observer = ProgramObserver(
        prog.field, particles,
        wells=wells_for_obs,
        capture_radius=8.0,
    )

    # Patch run_core to also call observer.record_step each step
    phase = prog.field.make_phase()
    for t in range(steps):
        E_before = prog.field.energy()
        for p in particles:
            p.update(prog.field, prog.phi, phase, t)
        observer.record_step(particles)
        E_mid = prog.field.energy()
        prog.field.decay()
        E_after = prog.field.energy()
        observer.ledger.record(E_before, E_mid, E_after)

    observer.finalise(prog.field, particles)
    observer.report()


if __name__ == "__main__":
    print("=== IFA Field Compiler — Canonical Program Suite ===")
    for prog_name in ("parity_checker", "binary_counter", "signal_propagator"):
        _run_canonical(prog_name, steps=300)
