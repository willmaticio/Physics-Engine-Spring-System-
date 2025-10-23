from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .system import System


class Integrator(Protocol):
    """Protocol implemented by time integration schemes."""

    def step(self, system: System, dt: float, t: float) -> None:
        ...


@dataclass
class SemiImplicitEuler:
    """Symplectic Euler integrator."""

    def step(self, system: System, dt: float, t: float) -> None:
        system.prepare_step(t)
        a_n = system.accelerations.copy()
        v = system.velocities.copy()
        x = system.positions.copy()
        free = system.free_dof_mask
        v[free] += dt * a_n[free]
        x[free] += dt * v[free]
        system._v = v  # type: ignore[attr-defined]
        system._x = x  # type: ignore[attr-defined]
        system._a = a_n  # type: ignore[attr-defined]
        system.prepare_step(t + dt)
        system.finalize_step()


@dataclass
class NewmarkBeta:
    """Average-acceleration Newmark-beta method."""

    beta: float = 0.25
    gamma: float = 0.5

    def __post_init__(self) -> None:
        if self.beta <= 0 or self.gamma <= 0:
            raise ValueError("Newmark-beta parameters must be positive.")

    def step(self, system: System, dt: float, t: float) -> None:
        system.prepare_step(t)
        a_n = system.accelerations.copy()
        u_n = system.displacements.copy()
        v_n = system.velocities.copy()
        free_idx = system.free_dof_indices

        u_pred = u_n + dt * v_n + (0.5 - self.beta) * dt * dt * a_n
        v_pred = v_n + (1.0 - self.gamma) * dt * a_n

        M = system.M
        C = system.C
        K = system.K
        f_ext = system.external_forces(t + dt)

        K_eff = M + self.gamma * dt * C + self.beta * dt * dt * K
        rhs = f_ext - C @ v_pred - K @ u_pred

        if free_idx.size > 0:
            K_ff = K_eff[np.ix_(free_idx, free_idx)]
            rhs_ff = rhs[free_idx]
            a_new_free = np.linalg.solve(K_ff, rhs_ff)
        else:
            a_new_free = np.array([], dtype=float)

        a_new = np.zeros_like(a_n)
        a_new[free_idx] = a_new_free

        u_new = u_pred + self.beta * dt * dt * a_new
        v_new = v_pred + self.gamma * dt * a_new
        x_new = system.rest_positions + u_new

        system._a = a_new  # type: ignore[attr-defined]
        system._v = v_new  # type: ignore[attr-defined]
        system._x = x_new  # type: ignore[attr-defined]
        system.finalize_step()


__all__ = ["Integrator", "SemiImplicitEuler", "NewmarkBeta"]
