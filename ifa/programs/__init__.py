"""ifa.programs — Canonical program library.  Re-exports PROGRAMS dict."""
from ifa_field_compiler import PROGRAMS, compile_program

__all__ = ["PROGRAMS", "compile_program"]
