"""
IFA Inverse Compiler — Constraint-Based Programming
=====================================================

Standard IFA compilation flow:
    instructions (list of opcodes) → field primitives → Φ

Inverse compilation flow:
    desired outcome (ProgramGoal) → heuristic solver → Φ

This module implements a first-generation heuristic inverse compiler.
Instead of specifying what primitives to place, the programmer declares
*what the computation should achieve*, and the solver constructs a field
that satisfies those constraints.

Goal specification
------------------
    goal = ProgramGoal(
        target       = (55, 20),              # where particles should converge
        must_pass    = [(10, 20), (30, 20)],  # ordered waypoints
        avoid        = [(20, 20)],            # regions to steer away from
        operations   = ["transform", "bind"], # computations to perform
        start        = (2.0, 20.0),           # CPU start position
    )

Solver strategy (greedy placement)
-----------------------------------
  1. Place attractor well at `target` (draws particles to the goal)
  2. Build a flow-channel chain: start → must_pass[0] → … → target
     (creates a corridor that routes particles through waypoints)
  3. For each declared operation, place the corresponding primitive at
     the appropriate must_pass waypoint:
       "transform" → transform_zone at first waypoint
       "bind"      → bind_gate at second waypoint
       "split"     → split_collapse_region(mode="split") at waypoint
       "collapse"  → split_collapse_region(mode="collapse") at waypoint
  4. Add Gaussian repulsion hills in Ψ at `avoid` positions
     (the static potential steers particles away from those regions)
  5. Validate: compile → run → check whether particles converge to target

Usage
-----
    from ifa_inverse_compiler import ProgramGoal, inverse_compile, validate_goal

    goal = ProgramGoal(
        target    = (55, 20),
        must_pass = [(15, 20), (35, 20)],
        avoid     = [(25, 15)],
        operations = ["transform", "bind"],
        start     = (2.0, 20.0),
    )
    prog   = inverse_compile(goal)
    result = validate_goal(prog, goal, steps=300)
    print(result)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from ifa_core import Particle, ENERGY_EPSILON
from ifa_field_compiler import (
    compile_program, CompiledProgram, compose,
    well, flow_channel, transform_zone, bind_gate, split_collapse_region,
    FieldPrimitive,
)

# ─────────────────────────────────────────────────────────────────────────────
# GOAL SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgramGoal:
    """
    Declarative specification of what an IFA program should compute.

    Attributes
    ----------
    target     : (x, y) — final convergence point; the program "halts" here
    must_pass  : ordered list of (x, y) waypoints the CPU must visit in order
    avoid      : list of (x, y) positions that the CPU should stay away from
    operations : list of operation names in order of execution
                 Supported: "transform", "bind", "split", "collapse"
    start      : (x, y) CPU start position; None → left-edge midpoint
    width      : field width (default 60)
    height     : field height (default 40)
    """
    target:     tuple[float, float]
    must_pass:  list[tuple[float, float]] = dc_field(default_factory=list)
    avoid:      list[tuple[float, float]] = dc_field(default_factory=list)
    operations: list[str]                  = dc_field(default_factory=list)
    start:      tuple[float, float] | None = None
    width:      int                        = 60
    height:     int                        = 40


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_position(sx: float, sy: float,
                           tx: float, ty: float,
                           step: int, total: int
                           ) -> tuple[float, float]:
    """
    Return a point linearly interpolated from (sx, sy) to (tx, ty).

    step  : 1-based index of this point along the path
    total : total number of points being interpolated
    """
    t = step / max(total, 1)
    return sx + (tx - sx) * t, sy + (ty - sy) * t


# ─────────────────────────────────────────────────────────────────────────────
# INVERSE COMPILER
# ─────────────────────────────────────────────────────────────────────────────

def inverse_compile(goal: ProgramGoal, seed: int = 0) -> CompiledProgram:
    """
    Heuristic inverse compiler: ProgramGoal → CompiledProgram.

    Returns a CompiledProgram whose field satisfies the goal constraints
    as closely as possible using greedy primitive placement.
    """
    tx, ty = goal.target
    sx, sy = goal.start if goal.start else (2.0, float(goal.height // 2))

    # ── Step 1: Assemble instruction list ────────────────────────────────
    instructions: list[tuple] = []

    # Attractor well at target (both Ψ and field layer)
    instructions.append(("TARGET_WELL", {"x": tx, "y": ty, "strength": 10.0}))
    instructions.append(("WELL",        {"x": tx, "y": ty,
                                          "strength": 9.0, "radius": 8.0}))

    # ── Step 2: Flow-channel chain along waypoints ────────────────────────
    waypoints = [( sx, sy)] + list(goal.must_pass) + [(tx, ty)]
    for i in range(len(waypoints) - 1):
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        instructions.append((
            "FLOW",
            {"start": p0, "end": p1, "width": 4.0, "speed": 0.55},
        ))

    # ── Step 3: Place operation primitives at waypoints ───────────────────
    _op_to_opcode = {
        "transform": "TRANSFORM",
        "bind":      "BIND",
        "split":     "SPLIT",
        "collapse":  "COLLAPSE",
    }
    _op_defaults = {
        "TRANSFORM": {"radius": 4.0, "intensity": 0.85},
        "BIND":      {"radius": 3.5, "threshold": 0.70},
        "SPLIT":     {"radius": 4.0},
        "COLLAPSE":  {"radius": 4.0},
    }
    for i, op_name in enumerate(goal.operations):
        op_name_lower = op_name.lower()
        if op_name_lower not in _op_to_opcode:
            raise ValueError(
                f"Unsupported operation {op_name!r}. "
                f"Supported: {list(_op_to_opcode)}"
            )
        opcode = _op_to_opcode[op_name_lower]
        # Place at the i-th must_pass waypoint if available, else interpolated
        if i < len(goal.must_pass):
            wx, wy = goal.must_pass[i]
        else:
            wx, wy = _interpolate_position(sx, sy, tx, ty,
                                           i + 1, len(goal.operations))
        params = {"cx": wx, "cy": wy, **_op_defaults[opcode]}
        instructions.append((opcode, params))

    # ── Step 4: CPU start position ────────────────────────────────────────
    instructions.append(("START", {"x": sx, "y": sy}))

    # ── Step 5: Compile to field ──────────────────────────────────────────
    prog = compile_program(
        instructions,
        width=goal.width,
        height=goal.height,
        seed=seed,
    )

    # ── Step 6: Add Ψ repulsion hills at avoid positions ─────────────────
    if goal.avoid:
        gy_grid, gx_grid = np.mgrid[
            0:goal.height, 0:goal.width
        ].astype(float)
        for ax, ay in goal.avoid:
            dist = np.hypot(gx_grid - ax, gy_grid - ay)
            prog.phi += 6.0 * np.exp(-0.5 * (dist / 3.0) ** 2)

    return prog


# ─────────────────────────────────────────────────────────────────────────────
# SOLUTION VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of running a compiled goal program."""
    converged:        bool
    dist_to_target:   float
    waypoints_passed: list[bool]
    avoid_violations: list[float]   # min distance to each avoid point
    energy_used:      float         # η = (E₀ − Eₜ) / E₀

    def __str__(self) -> str:
        lines = [
            "ValidationResult",
            f"  Converged to target  : {'✓' if self.converged else '✗'}  "
            f"(dist={self.dist_to_target:.2f})",
            f"  Waypoints visited    : "
            f"{sum(self.waypoints_passed)}/{len(self.waypoints_passed)}  "
            f"{self.waypoints_passed}",
            f"  Avoid violations     : {self.avoid_violations}",
            f"  Energy consumed      : {self.energy_used*100:.1f}%",
        ]
        return "\n".join(lines)


def validate_goal(prog:           CompiledProgram,
                  goal:           ProgramGoal,
                  steps:          int   = 300,
                  capture_radius: float = 8.0,
                  waypoint_radius: float = 12.0) -> ValidationResult:
    """
    Run a compiled program and check whether it satisfies the goal constraints.

    Parameters
    ----------
    prog            : compiled program to validate
    goal            : goal specification to validate against
    steps           : simulation steps
    capture_radius  : distance threshold to count as "reached target"
    waypoint_radius : distance threshold to count as "passed waypoint"

    Returns
    -------
    ValidationResult
    """
    particles = [Particle(x, y) for x, y in prog.start_pos]
    phase     = prog.field.make_phase()
    E_initial = prog.field.energy()

    # Per-particle trajectory
    trajectories: list[list[np.ndarray]] = [[] for _ in particles]

    for t in range(steps):
        for pi, p in enumerate(particles):
            p.update(prog.field, prog.phi, phase, t)
            trajectories[pi].append(p.pos.copy())
        prog.field.decay()

    E_final = prog.field.energy()
    eta = ((E_initial - E_final) / E_initial
           if E_initial > ENERGY_EPSILON else 0.0)

    tx, ty = goal.target

    # Check each particle (use first particle for waypoint/avoid checks)
    p = particles[0]
    traj = np.array(trajectories[0]) if trajectories[0] else np.zeros((1, 2))

    # Convergence
    dist_target = math.hypot(p.pos[0] - tx, p.pos[1] - ty)
    converged   = dist_target <= capture_radius

    # Waypoints: did the trajectory come within waypoint_radius of each?
    waypoints_passed = []
    for wx, wy in goal.must_pass:
        dists = np.hypot(traj[:, 0] - wx, traj[:, 1] - wy)
        waypoints_passed.append(bool(dists.min() <= waypoint_radius))

    # Avoid violations: how close did the trajectory come?
    avoid_violations = []
    for ax, ay in goal.avoid:
        dists = np.hypot(traj[:, 0] - ax, traj[:, 1] - ay)
        avoid_violations.append(float(dists.min()))

    return ValidationResult(
        converged=converged,
        dist_to_target=dist_target,
        waypoints_passed=waypoints_passed,
        avoid_violations=avoid_violations,
        energy_used=eta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — demonstration
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_GOALS: list[tuple[str, ProgramGoal]] = [
    (
        "Linear computation (load → transform → bind → halt)",
        ProgramGoal(
            target     = (55, 20),
            must_pass  = [(15, 20), (35, 20)],
            avoid      = [(25, 10)],
            operations = ["transform", "bind"],
            start      = (2.0, 20.0),
        ),
    ),
    (
        "Diagonal routing (start bottom-left → target top-right)",
        ProgramGoal(
            target     = (55, 8),
            must_pass  = [(20, 25), (40, 15)],
            avoid      = [],
            operations = ["transform", "transform"],
            start      = (2.0, 35.0),
        ),
    ),
    (
        "Split-and-converge (branch then merge)",
        ProgramGoal(
            target     = (55, 20),
            must_pass  = [(20, 20)],
            avoid      = [],
            operations = ["split", "collapse"],
            start      = (2.0, 20.0),
        ),
    ),
]


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   IFA INVERSE COMPILER — Constraint-Based Programming           ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║   Input : ProgramGoal(target, must_pass, avoid, operations)     ║")
    print("║   Output: Φ(x,y) constructed from primitives via heuristic      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    for title, goal in _DEMO_GOALS:
        print(f"\n{'━'*68}")
        print(f"  Goal: {title}")
        print(f"  Spec: target={goal.target}  must_pass={goal.must_pass}")
        print(f"        avoid={goal.avoid}  operations={goal.operations}")
        print(f"        start={goal.start}")
        print(f"{'─'*68}")

        prog   = inverse_compile(goal)
        result = validate_goal(prog, goal, steps=300)

        n_prims = len(prog.primitives)
        print(f"  Solver placed {n_prims} primitives:")
        for prim in prog.primitives:
            print(f"    • {prim.name:30s}  meta={prim.meta}")
        print()
        print(result)

    print()
    print("━" * 68)
    print("  Inverse compiler successfully translates goal constraints")
    print("  into composed instruction fields without hand-coding primitives.")
