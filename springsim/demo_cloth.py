from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import EnergyLogger, NewmarkBeta, SemiImplicitEuler, run_simulation
from .builders import build_cloth
from .viz import animate_system, plot_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a damped cloth under gravity.")
    parser.add_argument("--nx", type=int, default=15, help="Number of particles along X.")
    parser.add_argument("--ny", type=int, default=10, help="Number of particles along Y.")
    parser.add_argument("--k", type=float, default=1200.0, help="Spring stiffness.")
    parser.add_argument("--mass", type=float, default=0.05, help="Particle mass.")
    parser.add_argument("--dt", type=float, default=0.002, help="Time step size.")
    parser.add_argument("--steps", type=int, default=4000, help="Number of integration steps.")
    parser.add_argument("--alpha", type=float, default=0.02, help="Rayleigh mass damping coefficient.")
    parser.add_argument("--beta", type=float, default=0.002, help="Rayleigh stiffness damping coefficient.")
    parser.add_argument("--damping", type=float, default=0.5, help="Dashpot damping per spring.")
    parser.add_argument("--spacing", type=float, default=0.3, help="Rest spacing between lattice nodes.")
    parser.add_argument(
        "--integrator",
        choices=["euler", "newmark"],
        default="newmark",
        help="Integrator to use for time stepping.",
    )
    parser.add_argument("--no-shear", action="store_true", help="Disable diagonal shear springs.")
    parser.add_argument("--out", type=Path, default=Path("./out/cloth"), help="Output directory.")
    parser.add_argument("--frame-stride", type=int, default=0, help="Capture every N-th frame (auto if 0).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    system = build_cloth(
        args.nx,
        args.ny,
        mass=args.mass,
        stiffness=args.k,
        spacing=args.spacing,
        spring_damping=args.damping,
        include_shear=not args.no_shear,
        rayleigh_alpha=args.alpha,
        rayleigh_beta=args.beta,
    )

    integrator = SemiImplicitEuler() if args.integrator == "euler" else NewmarkBeta()
    energy_logger = EnergyLogger()
    stride = args.frame_stride if args.frame_stride > 0 else max(1, args.steps // 250)

    plot_configuration(system, system.positions, out_dir / "initial.png", title="Cloth - Initial")

    result = run_simulation(
        system,
        integrator,
        dt=args.dt,
        steps=args.steps,
        capture_stride=stride,
        energy_logger=energy_logger,
    )

    plot_configuration(system, system.positions, out_dir / "final.png", title="Cloth - Final")

    animation_target = out_dir / ("animation.mp4" if args.integrator == "newmark" else "animation.gif")
    animation_path = animate_system(system, result.positions, animation_target)
    energy_logger.to_csv(out_dir / "energy.csv")

    total = energy_logger.total_energy_trend()
    max_disp = result.max_total_displacement(system)
    avg_cpu = float(np.mean(result.cpu_times)) if result.cpu_times else 0.0

    print(f"Simulation finished in {args.steps} steps (dt={args.dt}).")
    print(f"Output directory: {out_dir}")
    print(f"Animation saved to: {animation_path}")
    print(f"Max displacement: {max_disp:.4f} m")
    print(f"Energy (initial -> final): {total[0]:.4f} -> {total[-1]:.4f}")
    print(f"Energy range: [{total.min():.4f}, {total.max():.4f}]")
    print(f"Average CPU time per step: {avg_cpu * 1000.0:.3f} ms")


if __name__ == "__main__":
    main()
