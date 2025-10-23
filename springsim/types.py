from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


Vector = np.ndarray


@dataclass(slots=True)
class Particle:
    """A point mass with two translational degrees of freedom."""

    pid: int
    mass: float
    fixed: bool
    position: Vector = field(repr=False)
    velocity: Vector = field(repr=False)

    def __post_init__(self) -> None:
        if self.position.shape != (2,):
            raise ValueError(f"Particle {self.pid}: position must be shape (2,), got {self.position.shape}")
        if self.velocity.shape != (2,):
            raise ValueError(f"Particle {self.pid}: velocity must be shape (2,), got {self.velocity.shape}")
        if self.mass <= 0.0:
            raise ValueError(f"Particle {self.pid}: mass must be positive.")


@dataclass(slots=True)
class Spring:
    """Linear spring (and optional dashpot) coupling two particles."""

    i: int
    j: int
    stiffness: float
    rest_length: float
    damping: Optional[float] = None

    def __post_init__(self) -> None:
        if self.i == self.j:
            raise ValueError("Spring endpoints must differ.")
        if self.stiffness <= 0.0:
            raise ValueError("Spring stiffness must be positive.")
        if self.rest_length <= 0.0:
            raise ValueError("Spring rest length must be positive.")
        if self.damping is not None and self.damping < 0.0:
            raise ValueError("Spring damping coefficient must be non-negative.")
