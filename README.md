# PlaceMux - Phase 3, Task 2: Observability Deep-Dive, SLOs & Error Budgets

This task implements the telemetry, monitoring, and error budget calculations for the PlaceMux matching intelligence system.

## Components
1. **Traffic Simulator** (`src/traffic_simulator.py`): Generates synthetic inference logs. It can inject artificial performance degradation like latency spikes or constant prediction scores.
2. **SLO Monitor** (`src/slo_monitor.py`): Ingests telemetry logs and evaluates them against defined targets (99.9% availability, p95 latency < 200ms, minimum score variance).
3. **Error Budget** (`src/error_budget.py`): Calculates how much of our failure budget has been consumed by a given slice of traffic.
4. **Demo Orchestrator** (`src/demo_slo_breach.py`): Runs multiple simulated traffic scenarios to demonstrate the alerting system functioning end-to-end.

## How to Run
Run the one-click Windows launcher to see the demo and package the deliverables:
```
.\run_task02.bat
```
