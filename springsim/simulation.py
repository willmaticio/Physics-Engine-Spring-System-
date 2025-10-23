from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

from .integrators import Integrator
from .system import System


@dataclass
class EnergySample:
    step: int
    time: float
    kinetic: float
    potential: float
    total: float


@dataclass
class EnergyLogger:
    samples: List[EnergySample] = field(default_factory=list)

    def log(self, step: int, time: float, system: System) -> None:
        self.samples.append(
            EnergySample(
                step=step,
                time=time,
                kinetic=system.kinetic_energy(),
                potential=system.potential_energy(),
                total=system.total_energy(),
            )
        )

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "step,time,kinetic,potential,total\n"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            for sample in self.samples:
                fh.write(f"{sample.step},{sample.time:.8f},{sample.kinetic:.8f},{sample.potential:.8f},{sample.total:.8f}\n")

    def total_energy_trend(self) -> np.ndarray:
        return np.array([sample.total for sample in self.samples], dtype=float)


@dataclass
class SimulationResult:
    times: List[float]
    positions: List[np.ndarray]
    velocities: List[np.ndarray]
    cpu_times: List[float]
    energy_logger: Optional[EnergyLogger] = None

    def max_total_displacement(self, system: System) -> float:
        max_disp = 0.0
        rest = system.rest_positions.reshape(-1, 2)
        for pos in self.positions:
            disp = pos.reshape(-1, 2) - rest
            max_disp = max(max_disp, float(np.max(np.linalg.norm(disp, axis=1))))
        return max_disp


def run_simulation(
    system: System,
    integrator: Integrator,
    dt: float,
    steps: int,
    *,
    start_time: float = 0.0,
    capture_stride: int = 1,
    energy_logger: Optional[EnergyLogger] = None,
    progress: Optional[Callable[[int, float, System], None]] = None,
) -> SimulationResult:
    if steps <= 0:
        raise ValueError("Number of steps must be positive.")
    if dt <= 0.0:
        raise ValueError("Time step must be positive.")
    capture_stride = max(1, capture_stride)

    times: List[float] = [start_time]
    positions: List[np.ndarray] = [system.positions.copy()]
    velocities: List[np.ndarray] = [system.velocities.copy()]
    cpu_times: List[float] = []

    current_time = start_time
    if energy_logger is not None:
        energy_logger.log(0, current_time, system)

    for step in range(1, steps + 1):
        start = _perf_counter()
        integrator.step(system, dt, current_time)
        elapsed = _perf_counter() - start
        current_time += dt
        cpu_times.append(elapsed)
        if energy_logger is not None:
            energy_logger.log(step, current_time, system)
        if step % capture_stride == 0 or step == steps:
            times.append(current_time)
            positions.append(system.positions.copy())
            velocities.append(system.velocities.copy())
        if progress is not None:
            progress(step, current_time, system)

    return SimulationResult(
        times=times,
        positions=positions,
        velocities=velocities,
        cpu_times=cpu_times,
        energy_logger=energy_logger,
    )


def _perf_counter() -> float:
    from time import perf_counter

    return perf_counter()


__all__ = ["EnergyLogger", "EnergySample", "SimulationResult", "run_simulation"]
