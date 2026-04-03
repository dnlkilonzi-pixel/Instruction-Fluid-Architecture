"""ifa.compiler — Forward + inverse compiler (IPM).  Re-exports."""
from ifa_field_compiler import (
    FieldPrimitive, compose,
    well, flow_channel, transform_zone, bind_gate, split_collapse_region,
    compile_program, CompiledProgram, ProgramObserver,
)
from ifa_inverse_compiler import ProgramGoal, inverse_compile, validate_goal, ValidationResult

__all__ = [
    "FieldPrimitive", "compose",
    "well", "flow_channel", "transform_zone", "bind_gate",
    "split_collapse_region",
    "compile_program", "CompiledProgram", "ProgramObserver",
    "ProgramGoal", "inverse_compile", "validate_goal", "ValidationResult",
]
