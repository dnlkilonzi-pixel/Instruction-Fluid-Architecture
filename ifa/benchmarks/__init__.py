"""ifa.benchmarks — Standard IFA benchmark suite.  Re-exports."""
from ifa_benchmarks import (
    BenchmarkResult,
    run_benchmark_parity,
    run_benchmark_counter,
    run_benchmark_routing,
    run_benchmark_noise_robustness,
    run_benchmark_multi_program,
    run_all_benchmarks,
    print_benchmark_table,
)
from ifa_advantage import (
    corrupt_field,
    run_degradation_experiment,
    traditional_model,
    print_degradation_report,
)

__all__ = [
    "BenchmarkResult",
    "run_benchmark_parity", "run_benchmark_counter", "run_benchmark_routing",
    "run_benchmark_noise_robustness", "run_benchmark_multi_program",
    "run_all_benchmarks", "print_benchmark_table",
    "corrupt_field", "run_degradation_experiment",
    "traditional_model", "print_degradation_report",
]
