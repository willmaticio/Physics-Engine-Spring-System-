from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .types import Particle, Spring, Vector


ArrayLike = np.ndarray
WindFn = Callable[[float, Vector], Vector]


@dataclass
class RayleighDamping:
    alpha: float
    beta: float

    def validate(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Rayleigh damping parameters must be non-negative.")


class System:
    """Mass-spring system with global matrix assembly."""

    def __init__(
        self,
        particles: Sequence[Particle],
        springs: Sequence[Spring],
        rayleigh_alpha: float = 0.01,
        rayleigh_beta: float = 0.001,
        gravity: Optional[Vector] = None,
        wind: Optional[WindFn] = None,
    ) -> None:
        if not particles:
            raise ValueError("System requires at least one particle.")
        self._particles: list[Particle] = [p for p in particles]
        self._springs: list[Spring] = [s for s in springs]
        self._gravity = np.array([0.0, -9.81]) if gravity is None else np.asarray(gravity, dtype=float)
        if self._gravity.shape != (2,):
            raise ValueError("Gravity vector must be 2D.")
        self._wind = wind
        self._n_particles = len(self._particles)
        self._ndof = 2 * self._n_particles
        self._rest_positions = self._flatten_positions([p.position for p in self._particles])
        self._x = self._rest_positions.copy()
        self._v = self._flatten_positions([p.velocity for p in self._particles])
        self._a = np.zeros(self._ndof)
        self._fixed_dofs = self._build_fixed_dofs()
        self._mass_diag = self._assemble_mass_diag()
        self.M = np.diag(self._mass_diag)
        self.K = self._assemble_stiffness_matrix()
        self.C_structural = self._assemble_structural_damping_matrix()
        self._rayleigh = RayleighDamping(rayleigh_alpha, rayleigh_beta)
        self._rayleigh.validate()
        self.C = self._assemble_damping_matrix()
        self.apply_constraints()
        self._synchronize_particle_states()

    # ------------------------------------------------------------------
    # Properties and helpers
    @property
    def particles(self) -> Sequence[Particle]:
        return self._particles

    @property
    def springs(self) -> Sequence[Spring]:
        return self._springs

    @property
    def ndof(self) -> int:
        return self._ndof

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def rayleigh_parameters(self) -> RayleighDamping:
        return self._rayleigh

    @property
    def rest_positions(self) -> ArrayLike:
        return self._rest_positions

    @property
    def positions(self) -> ArrayLike:
        return self._x

    @property
    def velocities(self) -> ArrayLike:
        return self._v

    @property
    def accelerations(self) -> ArrayLike:
        return self._a

    @property
    def displacements(self) -> ArrayLike:
        return self._x - self._rest_positions

    @property
    def free_dof_mask(self) -> np.ndarray:
        return self._fixed_dofs == 0

    @property
    def fixed_dof_mask(self) -> np.ndarray:
        return self._fixed_dofs == 1

    @property
    def free_dof_indices(self) -> np.ndarray:
        return np.nonzero(self.free_dof_mask)[0]

    @staticmethod
    def _flatten_positions(vectors: Iterable[Vector]) -> ArrayLike:
        return np.concatenate([np.asarray(vec, dtype=float) for vec in vectors])

    def _build_fixed_dofs(self) -> np.ndarray:
        mask = np.zeros(self._ndof, dtype=int)
        for index, particle in enumerate(self._particles):
            if particle.fixed:
                mask[2 * index : 2 * index + 2] = 1
        return mask

    # ------------------------------------------------------------------
    # Assembly
    def _assemble_mass_diag(self) -> ArrayLike:
        diag = np.zeros(self._ndof, dtype=float)
        for i, particle in enumerate(self._particles):
            diag[2 * i : 2 * i + 2] = particle.mass
        return diag

    def _assemble_stiffness_matrix(self) -> ArrayLike:
        K = np.zeros((self._ndof, self._ndof), dtype=float)
        for spring in self._springs:
            ii = spring.i
            jj = spring.j
            idx_i = slice(2 * ii, 2 * ii + 2)
            idx_j = slice(2 * jj, 2 * jj + 2)
            xi0 = self._rest_positions[idx_i]
            xj0 = self._rest_positions[idx_j]
            direction = xj0 - xi0
            length = np.linalg.norm(direction)
            if length <= 0.0:
                raise ValueError(f"Spring ({ii}, {jj}) has zero rest length vector.")
            n = direction / length
            k_local = spring.stiffness * np.outer(n, n)
            K[idx_i, idx_i] += k_local
            K[idx_j, idx_j] += k_local
            K[idx_i, idx_j] -= k_local
            K[idx_j, idx_i] -= k_local
        return K

    def _assemble_structural_damping_matrix(self) -> ArrayLike:
        C = np.zeros((self._ndof, self._ndof), dtype=float)
        for spring in self._springs:
            if spring.damping is None or spring.damping == 0.0:
                continue
            ii = spring.i
            jj = spring.j
            idx_i = slice(2 * ii, 2 * ii + 2)
            idx_j = slice(2 * jj, 2 * jj + 2)
            xi0 = self._rest_positions[idx_i]
            xj0 = self._rest_positions[idx_j]
            direction = xj0 - xi0
            length = np.linalg.norm(direction)
            if length <= 0.0:
                continue
            n = direction / length
            c_local = spring.damping * np.outer(n, n)
            C[idx_i, idx_i] += c_local
            C[idx_j, idx_j] += c_local
            C[idx_i, idx_j] -= c_local
            C[idx_j, idx_i] -= c_local
        return C

    def _assemble_damping_matrix(self) -> ArrayLike:
        alpha = self._rayleigh.alpha
        beta = self._rayleigh.beta
        return self.C_structural + alpha * self.M + beta * self.K

    def update_rayleigh_damping(self, alpha: float, beta: float) -> None:
        self._rayleigh = RayleighDamping(alpha, beta)
        self._rayleigh.validate()
        self.C = self._assemble_damping_matrix()

    # ------------------------------------------------------------------
    # Dynamics
    def external_forces(self, t: float) -> ArrayLike:
        forces = np.zeros(self._ndof, dtype=float)
        for i, particle in enumerate(self._particles):
            if particle.fixed:
                continue
            force = particle.mass * self._gravity
            if self._wind is not None:
                force = force + np.asarray(self._wind(t, self._x[2 * i : 2 * i + 2]), dtype=float)
            forces[2 * i : 2 * i + 2] = force
        return forces

    def compute_acceleration(self, t: float) -> ArrayLike:
        rhs = self.external_forces(t) - self.C @ self._v - self.K @ self.displacements
        acc = np.zeros_like(rhs)
        free = self.free_dof_mask
        acc[free] = rhs[free] / self._mass_diag[free]
        return acc

    def apply_constraints(self) -> None:
        # Enforce fixed displacement/velocity
        mask = self.fixed_dof_mask.astype(bool)
        self._x[mask] = self._rest_positions[mask]
        self._v[mask] = 0.0
        self._a[mask] = 0.0

    def set_state(self, positions: ArrayLike, velocities: ArrayLike) -> None:
        if positions.shape != (self._ndof,) or velocities.shape != (self._ndof,):
            raise ValueError("State vectors must have length equal to system DOFs.")
        self._x = positions.copy()
        self._v = velocities.copy()
        self.apply_constraints()
        self._a = self.compute_acceleration(0.0)
        self._synchronize_particle_states()

    def _synchronize_particle_states(self) -> None:
        for i, particle in enumerate(self._particles):
            particle.position = self._x[2 * i : 2 * i + 2].copy()
            particle.velocity = self._v[2 * i : 2 * i + 2].copy()

    # ------------------------------------------------------------------
    # Energy diagnostics
    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self._mass_diag * self._v**2))

    def potential_energy(self) -> float:
        disp = self.displacements
        return 0.5 * float(disp @ (self.K @ disp))

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def max_displacement(self) -> float:
        disp = self.displacements.reshape(-1, 2)
        norms = np.linalg.norm(disp, axis=1)
        return float(np.max(norms))

    # ------------------------------------------------------------------
    # Time stepping interface
    def prepare_step(self, t: float) -> None:
        self._a = self.compute_acceleration(t)

    def finalize_step(self) -> None:
        self.apply_constraints()
        self._synchronize_particle_states()


__all__ = ["System", "RayleighDamping"]
