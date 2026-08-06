# PlaceMux - Phase 3, Task 2: Observability Deep-Dive, SLOs & Error Budgets

This task implements the telemetry, monitoring, and error budget calculations for the PlaceMux matching intelligence system.

## Components
1. **Traffic Simulator** (`src/traffic_simulator.py`): Generates synthetic inference logs. It can inject artificial performance degradation like latency spikes or constant prediction scores.
2. **SLO Monitor** (`src/slo_monitor.py`): Ingests telemetry logs and evaluates them against defined targets (99.9% availability, p95 latency < 200ms, minimum score variance).
3. **Error Budget** (`src/error_budget.py`): Calculates how much of our failure budget has been consumed by a given slice of traffic.
4. **Demo Orchestrator** (`src/demo_slo_breach.py`): Runs multiple simulated traffic scenarios to demonstrate the alerting system functioning end-to-end.

## How to Run Task 2
Run the one-click Windows launcher to see the demo and package the deliverables:
```
.\run_task02.bat
```

---

# PlaceMux - Phase 3, Task 3: Performance Profiling & Bottleneck Elimination

This task simulates and resolves a common ML performance bottleneck: switching from row-by-row DataFrame operations to vectorized NumPy operations to meet strict latency SLOs.

## Components
1. **Inference Engines** (`src/inference_engine.py`): Contains both the slow, unoptimized `UnoptimizedInferenceEngine` and the mathematically identical but vastly faster `OptimizedInferenceEngine`.
2. **Profiler** (`src/profiler.py`): Runs `cProfile` to explicitly identify the bottleneck.
3. **Benchmark** (`src/benchmark.py`): Compares latency and cost across 100,000 synthetic records, asserting that quality remains identical.
4. **Demo Orchestrator** (`src/demo_optimization.py`): End-to-end launcher that outputs the profiling top 20 bottlenecks and the before/after numbers.

## How to Run Task 3
```
.\run_task03.bat
```
