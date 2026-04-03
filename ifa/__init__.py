"""
IFCF — Instruction Fluid Computational Field
=============================================

Top-level package for the IFA system.

Sub-packages
------------
  ifa.spec        ← formal specification documents
  ifa.core        ← IFA-Core runtime engine
  ifa.compiler    ← forward + inverse compiler (IPM)
  ifa.programs    ← canonical program library
  ifa.analysis    ← phase-space + trajectory analysis tools
  ifa.benchmarks  ← standard benchmark suite

All production code currently lives in the flat root-level modules
(ifa_core.py, ifa_field_compiler.py, …).  This package provides
convenient re-exports so that callers can use either the flat or the
package import style:

    from ifa_core import IFAField, Particle            # flat style
    from ifa.core import IFAField, Particle             # package style
"""

from ifa_core import IFAField, Particle, EnergyLedger, run_core
from ifa_field_compiler import (
    FieldPrimitive, compose,
    well, flow_channel, transform_zone, bind_gate, split_collapse_region,
    compile_program, CompiledProgram, ProgramObserver, PROGRAMS,
)
from ifa_inverse_compiler import ProgramGoal, inverse_compile, validate_goal
from ifa_benchmarks import run_all_benchmarks, print_benchmark_table

__all__ = [
    # core
    "IFAField", "Particle", "EnergyLedger", "run_core",
    # compiler / primitives
    "FieldPrimitive", "compose",
    "well", "flow_channel", "transform_zone", "bind_gate",
    "split_collapse_region",
    "compile_program", "CompiledProgram", "ProgramObserver", "PROGRAMS",
    # inverse compiler
    "ProgramGoal", "inverse_compile", "validate_goal",
    # benchmarks
    "run_all_benchmarks", "print_benchmark_table",
]
