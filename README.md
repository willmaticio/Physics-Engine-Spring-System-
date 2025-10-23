# Springsim – Mass–Spring Physics Engine

An educational mass–spring physics engine that assembles global mass (`M`), damping (`C`), and stiffness (`K`) matrices and advances them with matrix-based time integrators. The repository focuses on clarity and linear-algebra-centric implementation rather than game-specific optimisations.

## Features
- `Particle`/`Spring` dataclasses with 2D translational degrees of freedom backed by NumPy arrays.
- Global matrices:
  - Lumped diagonal mass matrix `M`.
  - Rayleigh plus dashpot damping matrix `C`.
  - Linearised stiffness matrix `K` assembled from the spring network.
- Integrators:
  - Semi-implicit (symplectic) Euler.
  - Average-acceleration Newmark–β (β=0.25, γ=0.5) scheme.
- Diagnostics: kinetic, potential, and total energy with CSV logging.
- Matplotlib visualisation for static frames and animations (MP4/GIF fallback or PNG frame sequence).
- CLI demos for a hanging chain and a cloth patch.

## Getting Started
1. Open a terminal and change into the repository root:
   ```bash
   cd "/Users/willmatic/Desktop/nexus_project/Physics Engine (Spring System)"
   ```
2. Create and populate a local virtual environment (internet connection required for NumPy/Matplotlib/SciPy downloads):
   ```bash
   ./scripts/setup_env.sh           # creates ./.venv by default
   source .venv/bin/activate        # activate the environment
   ```
   If you prefer a custom location, pass it as an argument: `./scripts/setup_env.sh ~/venvs/springsim`.
3. Once activated, run the demo modules (or use the console scripts listed in `pyproject.toml`):
   ```bash
   # Hanging chain
   python -m springsim.demo_chain --n 10 --k 1000 --mass 0.1 --dt 0.002 --steps 5000 --alpha 0.01 --beta 0.001 --out ./out/chain

   # Cloth patch
   python -m springsim.demo_cloth --nx 15 --ny 10 --k 1200 --mass 0.05 --dt 0.002 --steps 4000 --alpha 0.02 --beta 0.002 --out ./out/cloth
   ```
4. After the simulations finish, deactivate the session with `deactivate`. Re-activate later via `source .venv/bin/activate`.

### Expected Outputs
Each demo populates its output directory with:
- `initial.png` / `final.png` — snapshots of the configuration at the beginning and end of the run (blue circles are particles, red squares mark fixed nodes, springs are drawn as light gray segments).
- `animation.mp4` — a time-lapse of the motion. If FFmpeg/ImageMagick writers are unavailable, it falls back to `animation.gif` or a sequence of PNG frames.
- `energy.csv` — diagnostic log with columns `(step, time, kinetic, potential, total)` enabling post-hoc plotting of energy drift.
- Terminal summary including max displacement, initial/final energy, energy range, and mean CPU time per integration step.

Example terminal excerpt:
```
Simulation finished in 5000 steps (dt=0.002).
Output directory: /.../out/chain
Animation saved to: /.../out/chain/animation.mp4
Max displacement: 0.3182 m
Energy (initial -> final): 2.1456 -> 2.0123
Energy range: [1.9871, 2.1456]
Average CPU time per step: 1.742 ms
```

### Switching integrators
Pass `--integrator euler` to use the semi-implicit Euler scheme instead of the Newmark–β default. Euler is faster but more dissipative and less stable for stiff systems.

## Mathematical & Computational Foundations

### Linear Algebra in the Engine
- **Mass Matrix (`M`)** — Assembled as a lumped diagonal matrix where each particle contributes its scalar mass to the associated `x` and `y` degrees of freedom (`springsim/system.py:129`). Storing `M` in matrix form enables compact expressions for kinetic energy (`T = ½ vᵀMv`) and dynamic equilibrium.
- **Stiffness Matrix (`K`)** — Built by superimposing the outer product `k · n nᵀ` of unit spring directions over the global degrees of freedom (`springsim/system.py:135`). This mirrors finite-element assembly and exposes the system to spectral analysis, modal decomposition, and matrix-based solvers.
- **Damping Matrix (`C`)** — Combines Rayleigh damping (`αM + βK`) with optional dashpot contributions, yielding a generalized linear drag operator that couples velocities across the mesh (`springsim/system.py:179`).
- **Newmark-β Integration** — Treats the second-order ODE system as `K_eff a = rhs` where `K_eff = M + γΔt C + βΔt² K`; solving this linear system each step uses NumPy’s dense linear algebra (`springsim/integrators.py:63`). This parallels implicit time integrators in structural dynamics and CFD.
- **Energy Diagnostics** — Kinetic/potential energy evaluations rely on matrix-vector multiplications, reinforcing how quadratic forms encapsulate mechanical invariants (`springsim/simulation.py:26`).

### Computer Science Connections
- **Data Abstractions** — `Particle` and `Spring` dataclasses encapsulate state and parameters, while `System` orchestrates assembly, showing object-oriented design applied to physical simulation.
- **Algorithmic Design** — The integrator protocol decouples time-stepping strategies from system dynamics, demonstrating the Strategy pattern and enabling extensibility.
- **Performance Considerations** — Leveraging NumPy (vectorized operations) highlights the importance of linear algebra libraries in high-performance computing compared to naive Python loops.
- **Software Engineering Practices** — Modular layout, CLI interfaces, automated environment setup, and diagnostic logging illustrate reproducibility and maintainability in scientific software.

## Repository Layout
```
springsim/
  __init__.py
  builders.py        # Helper constructors for chain/cloth systems.
  integrators.py     # Semi-implicit Euler and Newmark–β implementations.
  simulation.py      # Energy logging and simulation orchestration.
  system.py          # Matrix assembly, state management, diagnostics.
  types.py           # Dataclasses for particles and springs.
  viz.py             # Plotting utilities and animation helpers.
  demo_chain.py      # CLI demo for a hanging chain.
  demo_cloth.py      # CLI demo for a cloth patch.
```

## Numerical Stability Notes
- The system uses a linear spring stiffness matrix built from the rest configuration. For large deformations, increase damping or reduce the time step.
- The Newmark–β integrator with β=0.25 and γ=0.5 is unconditionally stable for linear problems, but the dashpot/Rayleigh damping coefficients strongly influence oscillations.
- Recommended CFL-style guideline: ensure `dt` satisfies `dt < π * sqrt(m/k)` for the stiffest spring when using semi-implicit Euler.
- If the animation shows high-frequency chatter, increase `--alpha`/`--beta` or the per-spring dashpot (`--damping`).

## Extending
- Add custom external forces by providing a wind callback when constructing a `System`.
- Build new geometries with the helper functions in `springsim.builders` or assemble your own particle/spring lists.
- Swap or extend integrators by implementing the `Integrator` protocol.

## License
MIT (implicit – adjust to your preference).
