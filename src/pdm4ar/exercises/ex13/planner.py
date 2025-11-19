import ast
from dataclasses import dataclass, field
from typing import Union
from typing import Optional
import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
import numpy as np
from pdm4ar.exercises.ex13 import satellite
from pdm4ar.exercises.ex13.discretization import *
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "ECOS"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100  # max algorithm iterations

    # SCVX parameters (Add paper reference)
    lambda_nu: float = 1e5  # slack variable weight
    weight_p: NDArray = field(default_factory=lambda: 10 * np.array([[1.0]]).reshape((1, -1)))  # weight for final time

    tr_radius: float = 5  # initial trust region radius
    min_tr_radius: float = 1e-4  # min trust region radius
    max_tr_radius: float = 100  # max trust region radius
    rho_0: float = 0.0  # trust region 0
    rho_1: float = 0.25  # trust region 1
    rho_2: float = 0.9  # trust region 2
    alpha: float = 2.0  # div factor trust region update
    beta: float = 3.2  # mult factor trust region update

    # tolerances
    pos_tol: float = 0.5
    dir_tol: float = 0.1
    vel_tol: float = 0.2

    # Set max time
    p_max: float = 100

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-5  # Stopping criteria constant


class SatellitePlanner:
    """
    Feel free to change anything in this class.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters
    problem: Optional[cvx.Problem]

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Check dynamics implementation (pass this test before going further. It is not part of the final evaluation, so you can comment it out later)
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed.")
        else:
            print("Dynamics check passed.")

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        # for SCvx it would follow a logic similar to:

        # initial guess interpolation
        # while stopping criterion not satisfied
        #     convexify
        #     discretize
        #     solve convex sub problem
        #     update trust region
        #     update stopping criterion
        # DEBUG: tiny cvxpy sanity check

        # Convert init and goal state to arrays
        init_vec = np.array([init_state.x, init_state.y, init_state.psi, init_state.vx, init_state.vy, init_state.dpsi])
        goal_vec = np.array([goal_state.x, goal_state.y, goal_state.psi, goal_state.vx, goal_state.vy, goal_state.dpsi])
        self.problem_parameters["init_vec"].value = init_vec
        self.problem_parameters["goal_vec"].value = goal_vec

        # Assign values to problem parameters
        self._set_goal()

        # Initial reference
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()
        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

        for i in range(self.params.max_iterations):
            self._convexification()
            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
                break

            if self._check_convergence():
                X_star = self.variables["X"].value
                U_star = self.variables["U"].value
                p_star = self.variables["p"].value

                if X_star is not None and U_star is not None and p_star is not None:
                    self.X_bar = X_star.copy()
                    self.U_bar = U_star.copy()
                    self.p_bar = p_star.copy()
                break

            self._update_trust_region()

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        P = self.problem_parameters
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        init_vec = P["init_vec"].value
        goal_vec = P["goal_vec"].value

        X_bar = np.zeros((n_x, K))
        U_bar = np.zeros((n_u, K))
        p_bar = np.zeros(n_p)
        # Linear interpolation
        for k in range(K):
            tau = k / (K - 1)
            X_bar[:, k] = (1 - tau) * init_vec + tau * goal_vec
        # Initial guess for time
        p_bar[0] = 10

        return X_bar, U_bar, p_bar

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        P = self.problem_parameters
        # Set trust region radius
        P["eta_tr"].value = self.params.tr_radius

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        num_obstacles = len(self.planets) + len(self.asteroids)

        variables = {
            "X": cvx.Variable((n_x, K)),
            "U": cvx.Variable((n_u, K)),
            "p": cvx.Variable(n_p),
            # slack, virtual control
            "nu": cvx.Variable((n_x, K - 1)),
            "nu_s": cvx.Variable((num_obstacles, K)),
            "nu_ic": cvx.Variable(n_x),
            "nu_tc": cvx.Variable(n_x),
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K
        num_obstacles = len(self.planets) + len(self.asteroids)

        problem_parameters = {
            # vector form of initial and goal states
            "init_vec": cvx.Parameter(n_x),
            "goal_vec": cvx.Parameter(n_x),
            # linearized dynamics parameters
            "A_bar": [cvx.Parameter((n_x, n_x)) for _ in range(K - 1)],
            "B_minus_bar": [cvx.Parameter((n_x, n_u)) for _ in range(K - 1)],
            "B_plus_bar": [cvx.Parameter((n_x, n_u)) for _ in range(K - 1)],
            "F_bar": [cvx.Parameter((n_x, n_p)) for _ in range(K - 1)],
            "r_bar": [cvx.Parameter(n_x) for _ in range(K - 1)],
            # linearized obstacles parameters
            "C_coll": [[cvx.Parameter((1, n_x)) for _ in range(num_obstacles)] for _ in range(K)],
            "G_coll": [[cvx.Parameter((1, n_p)) for _ in range(num_obstacles)] for _ in range(K)],
            "r_coll": [[cvx.Parameter() for _ in range(num_obstacles)] for _ in range(K)],
            # trust region radius
            "eta_tr": cvx.Parameter(nonneg=True),
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p
        K = self.params.K
        pos_tol = self.params.pos_tol
        dir_tol = self.params.dir_tol
        vel_tol = self.params.vel_tol
        p_max = self.params.p_max

        P = self.problem_parameters
        goal = P["goal_vec"]
        eta_tr = P["eta_tr"]

        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]
        nu_s = self.variables["nu_s"]
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]

        num_obstacles = len(self.planets) + len(self.asteroids)

        # dynamic constraints
        dynamic_constraints = []
        for k in range(K - 1):
            dynamic_constraints.append(
                X[:, k + 1]
                == P["A_bar"][k] @ X[:, k]
                + P["B_minus_bar"][k] @ U[:, k]
                + P["B_plus_bar"][k] @ U[:, k + 1]
                + P["F_bar"][k] @ p
                + P["r_bar"][k]
                + nu[:, k]
            )

        # obstacles constraints
        obstacles_constraints = []
        for k in range(K):
            for j in range(num_obstacles):
                obstacles_constraints.append(
                    P["C_coll"][k][j] @ X[:, k] + P["G_coll"][k][j] @ p + P["r_coll"][k][j] <= nu_s[j, k]
                )

        # trust region constraints
        tr_constraints = []
        for k in range(K):
            dx = X[:, k] - self.X_bar[:, k]
            du = U[:, k] - self.U_bar[:, k]
            dp = p - self.p_bar
            tr_constraints.append(cvx.norm(dx, 2) + cvx.norm(du, 2) + cvx.norm(dp, 2) <= eta_tr)

        # general constraints
        gen_constraints = [
            # initial state
            X[:, 0] - P["init_vec"] == nu_ic,
            # final state
            # pose
            cvx.abs(X[0, K - 1] - goal[0]) <= pos_tol + nu_tc[0],
            cvx.abs(X[1, K - 1] - goal[1]) <= pos_tol + nu_tc[1],
            cvx.abs(X[2, K - 1] - goal[2]) <= dir_tol + nu_tc[2],
            # velocity
            cvx.abs(X[3, K - 1] - goal[3]) <= vel_tol + nu_tc[3],
            cvx.abs(X[4, K - 1] - goal[4]) <= vel_tol + nu_tc[4],
            cvx.abs(X[5, K - 1] - goal[5]) <= vel_tol + nu_tc[5],
            # control inputs at start and goal
            U[:, 0] == 0,
            U[:, K - 1] == 0,
            # control inputs within limits
            U >= self.sp.F_limits[0],
            U <= self.sp.F_limits[1],
            # max_time
            p <= p_max,
            p >= 0,
            # positive slack variables
            nu_tc >= 0,
            nu_s >= 0,
        ]

        constraints = gen_constraints + dynamic_constraints + obstacles_constraints + tr_constraints

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]

        K = self.params.K
        lam = self.params.lambda_nu
        nu = self.variables["nu"]
        nu_s = self.variables["nu_s"]
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]

        # travelled distance component
        travelled_distance = cvx.sum([cvx.norm(X[0:2, k + 1] - X[0:2, k], 2) for k in range(K - 1)])
        # average control component
        average_input = cvx.sum(cvx.abs(U)) / K
        # cost of slack variables that must be heavily penalized
        slack_cost = lam * (cvx.norm1(nu) + cvx.norm1(nu_s) + cvx.norm1(nu_ic) + cvx.norm1(nu_tc))

        objective = self.params.weight_p @ p + slack_cost + 0.5 * travelled_distance + 0.5 * average_input

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        P = self.problem_parameters
        K = self.params.K
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )
        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)
        # Therefore the matrices need to be reshaped

        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        for k in range(K - 1):
            P["A_bar"][k].value = A_bar[:, k].reshape((n_x, n_x), order="F")
            P["B_plus_bar"][k].value = B_plus_bar[:, k].reshape((n_x, n_u), order="F")
            P["B_minus_bar"][k].value = B_minus_bar[:, k].reshape((n_x, n_u), order="F")
            P["F_bar"][k].value = F_bar[:, k].reshape((n_x, n_p), order="F")
            P["r_bar"][k].value = r_bar[:, k]

        sat_radius = (self.sg.w_half + self.sg.w_panel) * 1.1
        # planets
        planets_list = []
        for planet in self.planets.values():
            planets_list.append({"x": planet.center[0], "y": planet.center[1], "r": planet.radius})
        # get the convexification of the obstacle for every time step
        for k in range(K):
            bar_x = self.X_bar[0, k]
            bar_y = self.X_bar[1, k]
            for j, obs in enumerate(planets_list):
                obs_x = obs["x"]
                obs_y = obs["y"]
                obs_r = obs["r"]
                r_safe_sq = (sat_radius + obs_r) ** 2
                dx = bar_x - obs_x
                dy = bar_y - obs_y
                C_val = np.zeros((1, n_x))
                C_val[0, 0] = -2 * dx
                C_val[0, 1] = -2 * dy
                # C
                P["C_coll"][k][j].value = C_val
                # G
                P["G_coll"][k][j].value = np.zeros((1, n_p))
                val_g = -(dx**2) - (dy**2) + r_safe_sq
                # r
                c_dot_x = C_val[0, 0] * bar_x + C_val[0, 1] * bar_y
                P["r_coll"][k][j].value = val_g - c_dot_x

        # asteroids
        asteroids_list = []
        for asteroid in self.asteroids.values():
            asteroids_list.append(
                {
                    "x0": asteroid.start[0],
                    "y0": asteroid.start[1],
                    "vx": asteroid.velocity[0],
                    "vy": asteroid.velocity[1],
                    "r": asteroid.radius,
                }
            )
        n_planets = len(self.planets)
        # get the convexification of the obstacle for every time step
        for k in range(K):
            tau_k = k / (K - 1)
            t_k = tau_k * float(self.p_bar[0])
            bar_x = self.X_bar[0, k]
            bar_y = self.X_bar[1, k]
            for j, obs in enumerate(asteroids_list):
                idx = n_planets + j
                obs_x0 = obs["x0"]
                obs_y0 = obs["y0"]
                obs_vx = obs["vx"]
                obs_vy = obs["vy"]
                obs_r = obs["r"]
                # current center coordinates
                obs_x = obs_x0 + obs_vx * t_k
                obs_y = obs_y0 + obs_vy * t_k
                r_safe_sq = (sat_radius + obs_r) ** 2
                dx = bar_x - obs_x
                dy = bar_y - obs_y
                C_val = np.zeros((1, n_x))
                C_val[0, 0] = -2 * dx
                C_val[0, 1] = -2 * dy
                # C
                P["C_coll"][k][idx].value = C_val
                # G
                val_g = -(dx**2) - (dy**2) + r_safe_sq
                dp = -2 * tau_k * (dx * obs_vx + dy * obs_vy)
                G_val = np.zeros((1, n_p))
                G_val[0, 0] = -dp
                P["G_coll"][k][idx].value = G_val
                # r
                c_dot_x = C_val[0, 0] * bar_x + C_val[0, 1] * bar_y
                P["r_coll"][k][idx].value = val_g - c_dot_x - G_val[0, 0] * float(self.p_bar[0])

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        # Extract stopping criterion
        eps = self.params.stop_crit

        # Extract new optimized values, we have to use .value to get numeric values from the symbolic CVXPY variable
        X_star = self.variables["X"].value
        p_star = self.variables["p"].value

        if X_star is None or p_star is None:
            return False

        # Extract reference trajectory from previous iteration
        X_ref = self.X_bar
        p_ref = self.p_bar

        # Compute the second stopping criterion from the slides with q=2 (euclidean norm)
        diff_p = np.linalg.norm(p_star - p_ref)

        diff_X = np.linalg.norm(X_star - X_ref, axis=0)  # axis=0 to get norm over state dimension not over time
        max_diff_X = np.max(diff_X)

        diff_tot = diff_p + max_diff_X

        return bool(diff_tot < eps)

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        assert self.problem is not None
        status = self.problem.status
        val = self.problem.value

        if val is None:
            print("[TR] Warning: problem.value is None despite status", status)
            eta = self.problem_parameters["eta_tr"].value
            eta = max(self.params.min_tr_radius, eta / self.params.alpha)
            self.problem_parameters["eta_tr"].value = eta
            return

        if not isinstance(val, (int, float)):
            raise TypeError(f"Unexpected type for problem.value: {type(val)}")

        # Linear predicted cost
        L_star = float(val)

        # Nonlinear cost of reference trajectory
        J_bar = float(self._J_lambda(self.X_bar, self.U_bar, self.p_bar))

        # Extract new optimized values, we have to use .value to get numeric values from the symbolic CVXPY variable
        X_star = self.variables["X"].value
        U_star = self.variables["U"].value
        p_star = self.variables["p"].value

        # Nonlinear cost of optimized trajectory
        J_star = float(self._J_lambda(X_star, U_star, p_star))

        den = J_bar - L_star
        if den == 0:
            rho = 0.0
        else:
            rho = (J_bar - J_star) / den

        # Current trust region radius
        eta = self.problem_parameters["eta_tr"].value

        # Trust region radius update
        accept = True
        if rho <= self.params.rho_0:
            # reject
            eta = max(self.params.min_tr_radius, eta / self.params.alpha)
            accept = False
        elif self.params.rho_0 < rho <= self.params.rho_1:
            # shrink
            eta = max(self.params.min_tr_radius, eta / self.params.alpha)
        # elif self.params.rho_1 < rho <= self.params.rho_2:
        # keep - eta doesn't change, the solution is accepted
        elif rho >= self.params.rho_2:
            # expand
            eta = min(self.params.max_tr_radius, eta * self.params.beta)

        # update tr radius
        self.problem_parameters["eta_tr"].value = eta

        # Update reference trajectory if accepted
        if accept:
            self.X_bar = X_star.copy()
            self.U_bar = U_star.copy()
            self.p_bar = p_star.copy()

        return

    def _J_lambda(self, X, U, p):
        """
        Compute the nonlinear cost.
        """
        K = self.params.K
        lam = self.params.lambda_nu

        # Travelled distance component
        travelled_distance = np.sum([np.linalg.norm(X[0:2, k + 1] - X[0:2, k], 2) for k in range(K - 1)])

        # Average control component
        average_input = np.sum(np.abs(U)) / K

        # Time cost component
        time_cost = float(self.params.weight_p @ p)

        # Compute defects
        x0 = self.problem_parameters["init_vec"].value
        X_nl = self.integrator.integrate_nonlinear_full(x0, U, p)
        defects = X[:, 1:] - X_nl[:, 1:]

        # Dynamic violation
        dyn_violation = np.sum(np.abs(defects))

        # Initial condition violation
        x0_target = self.problem_parameters["init_vec"].value
        init_violation = np.sum(np.abs(X[:, 0] - x0_target))

        # Terminal condition violation
        goal = self.problem_parameters["goal_vec"].value
        pos_tol = self.params.pos_tol
        dir_tol = self.params.dir_tol
        vel_tol = self.params.vel_tol
        X_terminal = X[:, -1]

        pos_err = np.linalg.norm(
            [
                np.maximum(np.abs(X_terminal[0] - goal[0]) - pos_tol, 0),
                np.maximum(np.abs(X_terminal[1] - goal[1]) - pos_tol, 0),
            ]
        )
        dir_err = np.maximum(np.abs(X_terminal[2] - goal[2]) - dir_tol, 0)
        vel_err = np.linalg.norm(
            [
                np.maximum(np.abs(X_terminal[3] - goal[3]) - vel_tol, 0),
                np.maximum(np.abs(X_terminal[4] - goal[4]) - vel_tol, 0),
                np.maximum(np.abs(X_terminal[5] - goal[5]) - vel_tol, 0),
            ]
        )

        term_violation = pos_err + dir_err + vel_err

        # Obstacle constraint violation
        P = self.problem_parameters
        num_obstacles = len(self.planets) + len(self.asteroids)
        obs_violation = 0
        for k in range(K):
            for j in range(num_obstacles):
                c = P["C_coll"][k][j].value
                g = P["G_coll"][k][j].value
                r = P["r_coll"][k][j].value
                val = float(c @ X[:, k] + g @ p + r)
                obs_violation += max(val, 0.0)

        # Total penalty replacing the slack variables
        slack_penalty = lam * (dyn_violation + init_violation + term_violation + obs_violation)

        # Final nonlinear cost
        J = time_cost + slack_penalty + 0.5 * travelled_distance + 0.5 * average_input
        return float(J)

    # @staticmethod
    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Create a DgSampledSequence from numpy arrays and timestamps.
        """
        K = self.params.K

        # Timestamps from 0 to final time
        t_final = float(self.p_bar[0])
        ts = np.linspace(0.0, t_final, K)

        # Commands
        cmds_list = []
        for k in range(K):
            F_left = float(self.U_bar[0, k])
            F_right = float(self.U_bar[1, k])
            cmds_list.append(SatelliteCommands(F_left, F_right))

        cmds_seq = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # States
        state_list = []
        for k in range(K):
            x = float(self.X_bar[0, k])
            y = float(self.X_bar[1, k])
            psi = float(self.X_bar[2, k])
            vx = float(self.X_bar[3, k])
            vy = float(self.X_bar[4, k])
            dpsi = float(self.X_bar[5, k])
            state_list.append(SatelliteState(x, y, psi, vx, vy, dpsi))

        state_seq = DgSampledSequence[SatelliteState](timestamps=ts, values=state_list)

        return cmds_seq, state_seq
