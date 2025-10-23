from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .system import System
from .types import Particle, Spring


def build_chain(
    n: int,
    *,
    mass: float,
    stiffness: float,
    spacing: float = 0.5,
    spring_damping: Optional[float] = None,
    rayleigh_alpha: float = 0.01,
    rayleigh_beta: float = 0.001,
) -> System:
    if n < 2:
        raise ValueError("Chain requires at least two particles.")
    particles: List[Particle] = []
    for i in range(n):
        position = np.array([i * spacing, 0.0], dtype=float)
        particle = Particle(pid=i, mass=mass, fixed=(i == 0), position=position, velocity=np.zeros(2))
        particles.append(particle)

    springs: List[Spring] = []
    for i in range(n - 1):
        rest_length = spacing
        springs.append(
            Spring(i=i, j=i + 1, stiffness=stiffness, rest_length=rest_length, damping=spring_damping)
        )

    return System(
        particles=particles,
        springs=springs,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
    )


def build_cloth(
    nx: int,
    ny: int,
    *,
    mass: float,
    stiffness: float,
    spacing: float = 0.3,
    spring_damping: Optional[float] = None,
    include_shear: bool = True,
    rayleigh_alpha: float = 0.02,
    rayleigh_beta: float = 0.002,
) -> System:
    if nx < 2 or ny < 2:
        raise ValueError("Cloth grid requires at least 2x2 particles.")
    particles: List[Particle] = []
    springs: List[Spring] = []

    def particle_index(ix: int, iy: int) -> int:
        return iy * nx + ix

    for iy in range(ny):
        for ix in range(nx):
            pid = particle_index(ix, iy)
            position = np.array([ix * spacing, -iy * spacing], dtype=float)
            fixed = iy == 0
            particles.append(Particle(pid=pid, mass=mass, fixed=fixed, position=position, velocity=np.zeros(2)))

    def add_spring(ix1: int, iy1: int, ix2: int, iy2: int, factor: float = 1.0) -> None:
        i = particle_index(ix1, iy1)
        j = particle_index(ix2, iy2)
        delta = particles[j].position - particles[i].position
        rest_length = float(np.linalg.norm(delta))
        springs.append(
            Spring(
                i=i,
                j=j,
                stiffness=stiffness * factor,
                rest_length=rest_length,
                damping=spring_damping,
            )
        )

    # Structural springs
    for iy in range(ny):
        for ix in range(nx):
            if ix + 1 < nx:
                add_spring(ix, iy, ix + 1, iy)
            if iy + 1 < ny:
                add_spring(ix, iy, ix, iy + 1)

    if include_shear:
        shear_factor = 0.75
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                add_spring(ix, iy, ix + 1, iy + 1, factor=shear_factor)
                add_spring(ix + 1, iy, ix, iy + 1, factor=shear_factor)

    return System(
        particles=particles,
        springs=springs,
        rayleigh_alpha=rayleigh_alpha,
        rayleigh_beta=rayleigh_beta,
    )


__all__ = ["build_chain", "build_cloth"]
