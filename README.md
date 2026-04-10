# PDM4AR – Exercise 13: Satellite Docking

Solutions for Exercise 13 of *Planning and Decision-Making for Autonomous Robots (PDM4AR)* at ETH Zurich.
The exercise tackles autonomous satellite docking via trajectory optimization under nonlinear dynamics and hard constraints.

Full problem description, scoring, and setup instructions are on the [course website](https://pdm4ar.github.io/exercises/13-satellite_docking.html).

---

## Repository Structure

All student code lives under `src/pdm4ar/exercises/`. The rest of the repository is the course-provided scaffold.

| Module | Responsibility |
|---|---|
| `agent.py` | Entry point; interfaces with the `dg_commons` simulator |
| `planner.py` | SCP-based trajectory optimizer |
| `constraints.py` | State, control, and docking cone constraints |
| `dynamics.py` | Linearization of the Clohessy–Wiltshire / nonlinear spacecraft model |

---

## Exercise Overview

The goal is to drive a chaser satellite from an initial state to a target docking port while:

- Respecting thrust limits and fuel budget (control constraints)
- Staying within a docking cone at approach
- Avoiding collision with the target body
- Satisfying terminal state constraints (position, velocity, attitude)

The planner uses **Sequential Convex Programming (SCP)**: the nonlinear problem is solved as a sequence of convex sub-problems, each linearized around the previous iterate.

---

## SCP Algorithm

The optimization loop proceeds as follows:

1. Warm-start with a straight-line reference trajectory.
2. Discretize the continuous-time dynamics (zero-order hold).
3. Linearize around the current reference using a first-order Taylor expansion.
4. Solve the resulting CVXPY problem (SOCP / QP depending on the cost formulation).
5. Compute the improvement ratio ρ = (actual reduction) / (predicted reduction).
6. Accept or reject the iterate and update the trust-region radius accordingly.
7. Terminate on convergence, trust-region collapse, or iteration budget exhaustion.

Solvers used: `Clarabel` (default) with `ECOS` as fallback.

---

## Implementation Notes

- **DCP compliance:** all constraints and the objective satisfy CVXPY's disciplined convex programming rules.
- **Numerical safety:** CVXPY variable values are extracted with `.value` before any Python-level logical checks or array operations.
- **Trust-region arithmetic:** ratio and radius updates operate purely on NumPy arrays, never on live CVXPY objects.
- **Simulator interface:** `on_episode_init()` triggers the full SCP solve; the planner returns a `VehicleInputs` command sequence compatible with `dg_commons`.

---

## Running the Simulation

```bash
# Build and launch via Docker (recommended)
make build
make run

# Or directly with Poetry
poetry install
poetry run python -m pdm4ar.exercises_def.ex13.runner
```

The simulator evaluates the returned trajectory on collision avoidance, terminal error, fuel consumption, and computation time.
