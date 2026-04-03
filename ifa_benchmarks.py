"""
IFA Benchmark Suite
===================

Standard test battery for the Instruction Fluid Architecture.

Five benchmarks, each producing three normalised scalar scores:

  success_rate      — fraction of particles / trials that converged
  energy_efficiency — η = (E₀ − Eₜ) / E₀  (field energy consumed)
  stability_margin  — 1 − r̄ / threshold   (>0 means STABLE)

Benchmark catalogue
-------------------
  1. parity             — parity-checker program, single run
  2. counter            — binary-counter program, single run
  3. routing            — signal-propagator program, three-particle run
  4. noise_robustness   — parity-checker under 20% corruption, 10 trials
  5. multi_program      — parity + signal_propagator composed into one field,
                          all particles run simultaneously

Usage
-----
    python ifa_benchmarks.py
    # or from another module:
    from ifa_benchmarks import run_all_benchmarks, print_benchmark_table
    results = run_all_benchmarks()
    print_benchmark_table(results)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ifa_core import (
    IFAField, Particle, EnergyLedger,
    ENERGY_EPSILON, GLOBAL_DECAY,
    TRANSFORM, BIND, FLOW_X, FLOW_Y, SPLIT, COLLAPSE, AMPLIFY, N_OPS,
)
from ifa_field_compiler import (
    PROGRAMS, compile_program, compose,
    well, flow_channel, transform_zone, bind_gate, split_collapse_region,
)
from ifa_advantage import corrupt_field

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

STEPS          = 300
CAPTURE_RADIUS = 8.0
TABLE_WIDTH    = 74


@dataclass
class BenchmarkResult:
    """Standardised output of one benchmark run."""
    name:              str
    success_rate:      float   # 0 – 1  (fraction that converged)
    energy_efficiency: float   # η = (E₀ − Eₜ) / E₀
    stability_margin:  float   # 1 − r̄ / threshold  (>0 = stable)
    notes:             str = ""


def _extract_wells(prog_source: list[tuple]) -> list[tuple[float, float]]:
    """Return list of (x, y) well positions from a program's source."""
    return [
        (instr[1]["x"], instr[1]["y"])
        for instr in prog_source
        if instr[0].upper() == "TARGET_WELL"
    ]


def _run_program(prog, steps: int = STEPS, capture_radius: float = CAPTURE_RADIUS
                 ) -> tuple[float, float, float]:
    """
    Run a compiled program and return (success_rate, energy_efficiency,
    stability_margin).
    """
    particles = [Particle(x, y) for x, y in prog.start_pos]
    wells     = _extract_wells(prog.source)
    ledger    = EnergyLedger()
    phase     = prog.field.make_phase()
    E_initial = prog.field.energy()

    for t in range(steps):
        E_before = prog.field.energy()
        for p in particles:
            p.update(prog.field, prog.phi, phase, t)
        E_mid = prog.field.energy()
        prog.field.decay()
        E_after = prog.field.energy()
        ledger.record(E_before, E_mid, E_after)

    E_final = prog.field.energy()

    # success_rate
    converged = 0
    for p in particles:
        if wells:
            d_min = min(math.hypot(p.pos[0] - wx, p.pos[1] - wy)
                        for wx, wy in wells)
        else:
            d_min = float("inf")
        if d_min <= capture_radius:
            converged += 1
    n = len(particles)
    success_rate = converged / n if n > 0 else 0.0

    # energy_efficiency
    eta = ((E_initial - E_final) / E_initial
           if E_initial > ENERGY_EPSILON else 0.0)

    # stability_margin
    if ledger.injection_ratios:
        r_bar     = float(np.mean(ledger.injection_ratios))
        threshold = 1.0 / (1.0 - GLOBAL_DECAY)
        margin    = 1.0 - r_bar / threshold
    else:
        margin = 0.0

    return success_rate, eta, margin


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKS 1–3: canonical programs (single run)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_parity() -> BenchmarkResult:
    """Benchmark 1 — Parity Checker."""
    prog = compile_program(PROGRAMS["parity_checker"])
    sr, eta, margin = _run_program(prog)
    return BenchmarkResult(
        name="parity",
        success_rate=sr,
        energy_efficiency=eta,
        stability_margin=margin,
        notes="single CPU, tape: 1-0-1, well at x=55",
    )


def run_benchmark_counter() -> BenchmarkResult:
    """Benchmark 2 — Binary Counter."""
    prog = compile_program(PROGRAMS["binary_counter"])
    sr, eta, margin = _run_program(prog)
    return BenchmarkResult(
        name="counter",
        success_rate=sr,
        energy_efficiency=eta,
        stability_margin=margin,
        notes="orbital accumulator, weak central well",
    )


def run_benchmark_routing() -> BenchmarkResult:
    """Benchmark 3 — Signal Propagator (3-lane routing)."""
    prog = compile_program(PROGRAMS["signal_propagator"])
    sr, eta, margin = _run_program(prog)
    return BenchmarkResult(
        name="routing",
        success_rate=sr,
        energy_efficiency=eta,
        stability_margin=margin,
        notes="3 particles, 3 wells; wave-routing",
    )


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 4: noise robustness (new)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_noise_robustness(
        corruption_fraction: float = 0.20,
        n_trials:            int   = 10,
) -> BenchmarkResult:
    """
    Benchmark 4 — Noise Robustness.

    Runs the signal_propagator program under 20% random field corruption
    for `n_trials` independent trials (different corruption masks).
    Reports the mean convergence rate across all trials.

    success_rate = mean fraction of particles converged across trials
    """
    rates   = []
    etas    = []
    margins = []

    for trial in range(n_trials):
        prog = compile_program(PROGRAMS["signal_propagator"])
        corrupt_field(prog.field, corruption_fraction, seed=trial)
        sr, eta, margin = _run_program(prog)
        rates.append(sr)
        etas.append(eta)
        margins.append(margin)

    return BenchmarkResult(
        name="noise_robustness",
        success_rate=float(np.mean(rates)),
        energy_efficiency=float(np.mean(etas)),
        stability_margin=float(np.mean(margins)),
        notes=(
            f"{corruption_fraction*100:.0f}% corruption, "
            f"{n_trials} trials, "
            f"σ(success)={float(np.std(rates)):.3f}"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK 5: multi-program interference (new)
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_multi_program(n_trials: int = 5) -> BenchmarkResult:
    """
    Benchmark 5 — Multi-Program Interference.

    Two compiled programs (parity_checker and signal_propagator) are
    composed onto the *same* IFAField.  All particles from both programs
    run simultaneously.  The benchmark measures:

      • Whether each group of particles reaches *its own* wells
      • Degree of interference (how much does sharing a field hurt?)

    success_rate = mean fraction of all particles that converged
                   to their respective group's wells across trials.

    Two trials are run:
      trial 0: both programs use the same field (interference scenario)
      trial 1: each program runs in isolation (baseline)

    stability_margin is the margin averaged over both programs' fields.
    """
    combined_rates  = []
    isolated_rates  = []
    etas            = []
    margins         = []

    for trial in range(n_trials):
        # ── Combined field ──────────────────────────────────────────────
        # Compile each program independently, then add their field tensors
        prog_a = compile_program(PROGRAMS["parity_checker"],  seed=trial)
        prog_b = compile_program(PROGRAMS["signal_propagator"], seed=trial + 100)

        # Build a fresh shared field and stamp both programs' primitives onto it
        shared_field = IFAField(width=60, height=60, seed=trial)
        compose(shared_field, *prog_a.primitives, *prog_b.primitives)

        # Combined Ψ (sum of both potentials)
        combined_phi = prog_a.phi + prog_b.phi

        # All particles: group A from parity, group B from signal_propagator
        particles_a = [Particle(x, y) for x, y in prog_a.start_pos]
        particles_b = [Particle(x, y) for x, y in prog_b.start_pos]
        all_particles = particles_a + particles_b

        wells_a = _extract_wells(prog_a.source)
        wells_b = _extract_wells(prog_b.source)

        ledger = EnergyLedger()
        phase  = shared_field.make_phase()
        E_init = shared_field.energy()

        for t in range(STEPS):
            E_before = shared_field.energy()
            for p in all_particles:
                p.update(shared_field, combined_phi, phase, t)
            E_mid = shared_field.energy()
            shared_field.decay()
            E_after = shared_field.energy()
            ledger.record(E_before, E_mid, E_after)

        E_final = shared_field.energy()

        # Convergence: group A to wells_a, group B to wells_b
        def _conv(particles, wells):
            c = 0
            for p in particles:
                if wells:
                    d = min(math.hypot(p.pos[0] - wx, p.pos[1] - wy)
                            for wx, wy in wells)
                else:
                    d = float("inf")
                if d <= CAPTURE_RADIUS:
                    c += 1
            return c / len(particles) if particles else 0.0

        rate_a = _conv(particles_a, wells_a)
        rate_b = _conv(particles_b, wells_b)
        combined_rate = (rate_a + rate_b) / 2.0
        combined_rates.append(combined_rate)

        eta = ((E_init - E_final) / E_init if E_init > ENERGY_EPSILON else 0.0)
        etas.append(eta)

        if ledger.injection_ratios:
            r_bar     = float(np.mean(ledger.injection_ratios))
            threshold = 1.0 / (1.0 - GLOBAL_DECAY)
            margins.append(1.0 - r_bar / threshold)

        # ── Isolated baselines ──────────────────────────────────────────
        sr_a, _, _ = _run_program(compile_program(PROGRAMS["parity_checker"],
                                                   seed=trial))
        sr_b, _, _ = _run_program(compile_program(PROGRAMS["signal_propagator"],
                                                   seed=trial + 100))
        isolated_rates.append((sr_a + sr_b) / 2.0)

    mean_combined  = float(np.mean(combined_rates))
    mean_isolated  = float(np.mean(isolated_rates))
    interference   = mean_isolated - mean_combined   # > 0 means some degradation
    notes = (
        f"combined={mean_combined:.3f} vs isolated={mean_isolated:.3f}; "
        f"interference={interference:.3f}"
    )

    return BenchmarkResult(
        name="multi_program",
        success_rate=mean_combined,
        energy_efficiency=float(np.mean(etas)),
        stability_margin=float(np.mean(margins)) if margins else 0.0,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks() -> list[BenchmarkResult]:
    """Run all 5 benchmarks in order and return results."""
    results = []
    steps = [
        ("parity",           run_benchmark_parity),
        ("counter",          run_benchmark_counter),
        ("routing",          run_benchmark_routing),
        ("noise_robustness", lambda: run_benchmark_noise_robustness()),
        ("multi_program",    lambda: run_benchmark_multi_program()),
    ]
    for name, fn in steps:
        print(f"  Running benchmark: {name}…", flush=True)
        results.append(fn())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TABLE PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print a formatted benchmark results table."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                     IFA BENCHMARK SUITE — RESULTS                      ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    hdr = (f"  {'Benchmark':<22}"
           f"  {'success_rate':>12}"
           f"  {'energy_eff':>10}"
           f"  {'stab_margin':>11}"
           f"  Notes")
    print(f"║{hdr:<{TABLE_WIDTH}}║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")

    for r in results:
        stab_flag = "✓" if r.stability_margin > 0 else "✗"
        row = (f"  {r.name:<22}"
               f"  {r.success_rate:>12.3f}"
               f"  {r.energy_efficiency:>10.3f}"
               f"  {r.stability_margin:>9.4f}{stab_flag}"
               f"  {r.notes}")
        # Truncate to fit terminal width
        if len(row) > TABLE_WIDTH:
            row = row[:TABLE_WIDTH - 3] + "…"
        print(f"║{row:<{TABLE_WIDTH}}║")

    print("╠══════════════════════════════════════════════════════════════════════════╣")

    # Summary aggregates
    mean_sr  = float(np.mean([r.success_rate      for r in results]))
    mean_eta = float(np.mean([r.energy_efficiency for r in results]))
    mean_sm  = float(np.mean([r.stability_margin  for r in results]))
    all_stable = all(r.stability_margin > 0 for r in results)

    summary = (f"  {'MEAN':<22}"
               f"  {mean_sr:>12.3f}"
               f"  {mean_eta:>10.3f}"
               f"  {mean_sm:>9.4f}{'✓' if all_stable else '✗'}")
    print(f"║{summary:<{TABLE_WIDTH}}║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()
    print("  Columns:")
    print("    success_rate   — fraction of particles / trials that converged to a well")
    print("    energy_eff     — η = (E₀ − Eₜ) / E₀  (fraction of field energy used)")
    print("    stab_margin    — 1 − r̄/threshold  (>0 means energy-stable)")
    print()
    all_stable_str = "ALL stable ✓" if all_stable else "some UNSTABLE ✗"
    print(f"  System stability  : {all_stable_str}")
    print(f"  Mean success rate : {mean_sr:.1%}")
    print(f"  Mean energy use   : {mean_eta:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              IFA BENCHMARK SUITE (5 benchmarks)                 ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print("║  1. parity            — single-CPU parity check                 ║")
    print("║  2. counter           — orbital binary counter                  ║")
    print("║  3. routing           — 3-lane wave routing                     ║")
    print("║  4. noise_robustness  — 20% corruption, 10 trials (NEW)         ║")
    print("║  5. multi_program     — 2 programs coexisting in one field (NEW)║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    print("  Running…")
    print()

    results = run_all_benchmarks()
    print_benchmark_table(results)
