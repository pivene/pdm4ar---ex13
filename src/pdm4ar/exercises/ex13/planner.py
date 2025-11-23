import ast
from dataclasses import dataclass, field
from typing import Union
from typing import Optional
import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.goals import PlanningGoal
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
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

    # Set max time
    p_max: float = 60

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
    goal: PlanningGoal

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
        goal: PlanningGoal,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp
        self.goal = goal

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
        num_asteroids = len(self.asteroids)
        num_planets = len(self.planets)
        K = self.params.K
        if num_asteroids != 0:
            self.variables.update({"nu_s_a": cvx.Variable((num_asteroids, K))})
        if num_planets != 0:
            self.variables.update({"nu_s_p": cvx.Variable((num_planets, K))})
        if isinstance(self.goal, DockingTarget):
            self.variables.update({"nu_s_dock": cvx.Variable(K - 5)})
            self.variables.update({"nu_pos_dock": cvx.Variable(6)})

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
        X_bar, U_bar, p_bar = self.initial_guess()
        # update self.
        self.X_bar = X_bar
        self.U_bar = U_bar
        self.p_bar = p_bar

        constraints = self._get_constraints()
        objective = self._get_objective()
        self.problem = cvx.Problem(objective, constraints)

        # update prob params
        self.problem_parameters["X_bar"].value = X_bar
        self.problem_parameters["U_bar"].value = U_bar
        self.problem_parameters["p_bar"].value = p_bar

        for i in range(self.params.max_iterations):
            print(i)

            self._convexification()

            try:
                error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
                break

            if self._check_convergence():
                # print slack variables for debugging
                # missing slck for asteroids
                print(np.max(self.variables["nu"].value))
                if len(self.planets) != 0:
                    print(np.max(self.variables["nu_s_p"].value))
                if len(self.asteroids) != 0:
                    print(np.max(self.variables["nu_s_a"].value))
                print(np.max(self.variables["nu_ic"].value))
                print(np.max(self.variables["nu_tc"].value))
                break

            print(
                np.max(self.variables["nu"].value),
                np.max(self.variables["nu_ic"].value),
                np.max(self.variables["nu_tc"].value),
            )
            if len(self.planets) != 0:
                print(np.max(self.variables["nu_s_p"].value))
            if len(self.asteroids) != 0:
                print(np.max(self.variables["nu_s_a"].value))
            if isinstance(self.goal, DockingTarget):
                print(np.max(self.variables["nu_s_dock"].value))
                print(np.max(self.variables["nu_pos_dock"].value))

            # self._update_trust_region()
            # update both prob params and self.
            self.problem_parameters["X_bar"].value = self.variables["X"].value
            self.problem_parameters["U_bar"].value = self.variables["U"].value
            self.problem_parameters["p_bar"].value = self.variables["p"].value
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

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
        p_bar[0] = 15

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
        num_planets = len(self.planets)

        variables = {
            "X": cvx.Variable((n_x, K)),
            "U": cvx.Variable((n_u, K)),
            "p": cvx.Variable(n_p),
            # slack, virtual control
            "nu": cvx.Variable((n_x, K - 1)),
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

        problem_parameters = {
            # reference trajectory
            "X_bar": cvx.Parameter((n_x, K)),
            "U_bar": cvx.Parameter((n_u, K)),
            "p_bar": cvx.Parameter(n_p),
            # vector form of initial and goal states
            "init_vec": cvx.Parameter(n_x),
            "goal_vec": cvx.Parameter(n_x),
            # linearized dynamics parameters
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
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
        p_max = self.params.p_max

        P = self.problem_parameters

        num_planets = len(self.planets)
        num_asteroids = len(self.asteroids)

        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu = self.variables["nu"]

        # dynamic constraints
        dynamic_constraints = []
        for k in range(K - 1):
            A_k = cvx.reshape(P["A_bar"][:, k], (n_x, n_x))
            Bm_k = cvx.reshape(P["B_minus_bar"][:, k], (n_x, n_u))
            Bp_k = cvx.reshape(P["B_plus_bar"][:, k], (n_x, n_u))
            F_k = cvx.reshape(P["F_bar"][:, k], (n_x, n_p))
            r_k = P["r_bar"][:, k]
            dynamic_constraints.append(
                X[:, k + 1] == A_k @ X[:, k] + Bm_k @ U[:, k] + Bp_k @ U[:, k + 1] + F_k @ p + r_k + nu[:, k]
            )

        # planets constraints
        sat_radius = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + max(self.sg.l_f, self.sg.l_r) ** 2)
        planets_constraints = []
        if num_planets != 0:
            for k in range(K):
                bar_x = self.problem_parameters["X_bar"][0, k]
                bar_y = self.problem_parameters["X_bar"][1, k]
                n_ps = 0
                for planet in self.planets.values():
                    obs_x = planet.center[0]
                    obs_y = planet.center[1]
                    obs_r = planet.radius

                    dx = bar_x - obs_x
                    dy = bar_y - obs_y

                    r_safe_sq = (sat_radius + obs_r) ** 2

                    r = -(dx**2) - (dy**2) + r_safe_sq + 2 * dx * bar_x + 2 * dy * bar_y

                    planets_constraints.append(
                        -2 * dx * self.variables["X"][0, k] - 2 * dy * self.variables["X"][1, k] + r
                        <= self.variables["nu_s_p"][n_ps, k]
                    )

                    n_ps += 1

        # asteroids constraints
        asteroids_constraints = []
        if num_asteroids != 0:
            for k in range(K):
                tau_k = k / (K - 1)
                t_k = tau_k * self.problem_parameters["p_bar"][0]
                bar_x = self.problem_parameters["X_bar"][0, k]
                bar_y = self.problem_parameters["X_bar"][1, k]
                j = 0
                for asteroid in self.asteroids.values():
                    obs_x0 = asteroid.start[0]
                    obs_y0 = asteroid.start[1]
                    obs_vx = asteroid.velocity[0]
                    obs_vy = asteroid.velocity[1]
                    obs_r = asteroid.radius
                    theta = asteroid.orientation
                    # correct for orientation
                    world_vx = obs_vx * np.cos(theta) - obs_vy * np.sin(theta)
                    world_vy = obs_vy * np.cos(theta) + obs_vx * np.sin(theta)
                    # current center coordinates
                    obs_x = obs_x0 + world_vx * t_k
                    obs_y = obs_y0 + world_vy * t_k

                    dx = bar_x - obs_x
                    dy = bar_y - obs_y

                    r_safe_sq = (sat_radius + obs_r) ** 2

                    dp = -2 * tau_k * (dx * obs_vx + dy * obs_vy)

                    r = (
                        -(dx**2)
                        - (dy**2)
                        + r_safe_sq
                        + 2 * dx * bar_x
                        + 2 * dy * bar_y
                        + dp * self.problem_parameters["p_bar"][0]
                    )

                    asteroids_constraints.append(
                        -2 * dx * self.variables["X"][0, k]
                        - 2 * dy * self.variables["X"][1, k]
                        - dp * self.variables["p"][0]
                        + r
                        <= self.variables["nu_s_a"][j, k]
                    )

                    j += 1

        # trust region constraints
        tr_constraints = []
        for k in range(K):
            dx = X[:, k] - self.X_bar[:, k]
            du = U[:, k] - self.U_bar[:, k]
            dp = p - self.p_bar[0]
            tr_constraints.append(
                cvx.norm(dx, 2) + cvx.norm(du, 2) + cvx.norm(dp, 2) <= self.problem_parameters["eta_tr"]
            )

        # general constraints
        gen_constraints = [
            # initial state
            self.variables["X"][:, 0] - self.problem_parameters["init_vec"] - self.variables["nu_ic"] == 0,
            # final state
            self.variables["X"][:, -1] - self.problem_parameters["goal_vec"] - self.variables["nu_tc"] == 0,
            # control inputs at start and goal
            self.variables["U"][:, 0] == 0,
            self.variables["U"][:, K - 1] == 0,
            # control inputs within limits
            self.variables["U"] >= self.sp.F_limits[0],
            self.variables["U"] <= self.sp.F_limits[1],
            # max_time
            self.variables["p"] <= p_max,
            self.variables["p"] >= 0,
            # within bounds
            self.variables["X"][0, :] >= -11 + sat_radius,
            self.variables["X"][1, :] >= -11 + sat_radius,
            self.variables["X"][0, :] <= 11 - sat_radius,
            self.variables["X"][1, :] <= 11 - sat_radius,
        ]

        if num_planets != 0:
            gen_constraints.append(self.variables["nu_s_p"] >= 0)

        if num_asteroids != 0:
            gen_constraints.append(self.variables["nu_s_a"] >= 0)

        # docking constraints
        docking_constraints = []
        if isinstance(self.goal, DockingTarget):
            gen_constraints.append(self.variables["nu_s_dock"] >= 0)
            A, B, C, A1, A2, half_p_angle = self.goal.get_landing_constraint_points()
            segment = np.array([A1[0] - A2[0], A1[1] - A2[1]])  # from A2 to A1
            len_seg = np.linalg.norm(segment)
            norm_seg = segment / len_seg
            midpoint = np.array([A2[0] + norm_seg[0] * len_seg / 2, A2[1] + norm_seg[1] * len_seg / 2])
            obs_x = midpoint[0]
            obs_y = midpoint[1]
            obs_r = len_seg / 2
            for k in range(K - 5):
                bar_x = self.problem_parameters["X_bar"][0, k]
                bar_y = self.problem_parameters["X_bar"][1, k]
                dx = bar_x - obs_x
                dy = bar_y - obs_y

                r_safe_sq = (sat_radius + obs_r) ** 2

                r = -(dx**2) - (dy**2) + r_safe_sq + 2 * dx * bar_x + 2 * dy * bar_y

                docking_constraints.append(
                    -2 * dx * self.variables["X"][0, k] - 2 * dy * self.variables["X"][1, k] + r
                    <= self.variables["nu_s_dock"][k]
                )
            seg_A_B = np.array([B[0] - A[0], B[1] - A[1]])  # from A to B
            seg_A_C = np.array([C[0] - A[0], C[1] - A[1]])  # from A to C
            for local_i, k in enumerate(range(K - 6, K)):
                xs = self.variables["X"][0, k]
                ys = self.variables["X"][1, k]
                expr1 = seg_A_B[0] * (xs - A[0]) + seg_A_B[1] * (ys - A[1])
                docking_constraints.append(expr1 >= self.variables["nu_pos_dock"][local_i])
                expr2 = seg_A_C[0] * (xs - A[0]) + seg_A_C[1] * (ys - A[1])
                docking_constraints.append(expr2 >= self.variables["nu_pos_dock"][local_i])

        constraints = (
            gen_constraints
            + dynamic_constraints
            + planets_constraints
            + asteroids_constraints
            + tr_constraints
            + docking_constraints
        )

        return constraints

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Define objective for SCvx.
        """
        # Example objective
        num_planets = len(self.planets)
        num_asteroids = len(self.asteroids)
        K = self.params.K

        # cost of slack variables that must be heavily penalized
        slack_cost = self.params.lambda_nu * (
            cvx.norm1(self.variables["nu"]) + cvx.norm1(self.variables["nu_ic"]) + cvx.norm1(self.variables["nu_tc"])
        )
        if num_planets != 0:
            slack_cost += self.params.lambda_nu * cvx.norm1(self.variables["nu_s_p"])
        if num_asteroids != 0:
            slack_cost += self.params.lambda_nu * cvx.norm1(self.variables["nu_s_a"])
        if isinstance(self.goal, DockingTarget):
            slack_cost += self.params.lambda_nu * cvx.norm1(self.variables["nu_s_dock"])
            slack_cost += self.params.lambda_nu * cvx.norm1(self.variables["nu_pos_dock"])

        objective = self.params.weight_p @ self.variables["p"] + slack_cost

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        # ZOH
        # A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
        # FOH
        A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
            self.X_bar, self.U_bar, self.p_bar
        )
        # HINT: be aware that the matrices returned by calculate_discretization are flattened in F order (this way affect your code later when you use them)
        # Therefore the matrices need to be reshaped

        self.problem_parameters["A_bar"].value = A_bar
        self.problem_parameters["B_minus_bar"].value = B_minus_bar
        self.problem_parameters["B_plus_bar"].value = B_plus_bar
        self.problem_parameters["F_bar"].value = F_bar
        self.problem_parameters["r_bar"].value = r_bar

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """
        eps = self.params.stop_crit

        diff_p = np.linalg.norm(self.variables["p"].value - self.problem_parameters["p_bar"].value)

        diff_X = np.linalg.norm(self.variables["X"].value - self.problem_parameters["X_bar"].value, axis=0)
        max_diff_X = np.max(diff_X)

        diff_tot = diff_p + max_diff_X
        print(diff_tot)

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
        print("L_star = ", L_star)

        # Nonlinear cost of reference trajectory
        J_bar = float(self._J_lambda(self.X_bar, self.U_bar, self.p_bar))
        print("J_bar = ", J_bar)

        # Nonlinear cost of optimized trajectory
        J_star = float(self._J_lambda(self.variables["X"].value, self.variables["U"].value, self.variables["p"].value))
        print("J_star = ", J_star)
        den = J_bar - L_star

        if den == 0:
            print("denominatore rho UGUALE A 0")
            rho = 0.0
        else:
            rho = (J_bar - J_star) / den
            print("Rho = ", rho)

        # Current trust region radius
        eta = self.problem_parameters["eta_tr"].value

        # Trust region radius update
        accept = True
        if rho < self.params.rho_0:
            # shrink and reject
            eta = max(self.params.min_tr_radius, eta / self.params.alpha)
            accept = False
        elif self.params.rho_0 <= rho < self.params.rho_1:
            # shrink and accept
            eta = max(self.params.min_tr_radius, eta / self.params.alpha)
        # elif self.params.rho_1 < rho <= self.params.rho_2:
        # keep - eta doesn't change, the solution is accepted
        elif rho >= self.params.rho_2:
            # expand and accept
            eta = min(self.params.max_tr_radius, eta * self.params.beta)

        print("Eta = ", eta)
        # update tr radius
        self.problem_parameters["eta_tr"].value = eta

        # Update reference trajectory if accepted
        if accept:
            self.problem_parameters["X_bar"].value = self.variables["X"].value
            self.problem_parameters["U_bar"].value = self.variables["U"].value
            self.problem_parameters["p_bar"].value = self.variables["p"].value
            self.X_bar = self.variables["X"].value
            self.U_bar = self.variables["U"].value
            self.p_bar = self.variables["p"].value

        return

    def _J_lambda(self, X, U, p):
        """
        Compute the nonlinear cost.
        """
        P = self.problem_parameters
        K = self.params.K
        lam = self.params.lambda_nu
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        # Time cost component
        time_cost = float(self.params.weight_p @ p)
        print("time_cost = ", time_cost)

        # Compute defects
        defects = []
        X_nl = self.integrator.integrate_nonlinear_piecewise(X, U, p)
        for k in range(K - 1):
            A_bar = self.problem_parameters["A_bar"][:, k].value
            B_minus_bar = self.problem_parameters["B_minus_bar"][:, k].value
            B_plus_bar = self.problem_parameters["B_minus_bar"][:, k].value
            F_bar = self.problem_parameters["F_bar"][:, k].value
            r_bar = self.problem_parameters["r_bar"][:, k].value
            A_k = A_bar.reshape((n_x, n_x), order="F")
            Bm_k = B_minus_bar.reshape((n_x, n_u), order="F")
            Bp_k = B_plus_bar.reshape((n_x, n_u), order="F")
            F_k = F_bar.reshape((n_x, n_p), order="F")
            r_k = r_bar
            defects.append(-X_nl[:, k + 1] + (A_k @ X[:, k] + Bp_k @ U[:, k + 1] + Bm_k @ U[:, k] + F_k @ p + r_k))

        # Dynamic violation
        dyn_violation = np.sum(np.abs(defects))
        print("dyn_violation = ", dyn_violation)

        # Initial condition violation
        x0_target = self.problem_parameters["init_vec"].value
        init_violation = np.sum(np.abs(X[:, 0] - x0_target))
        print("init_violation = ", init_violation)

        # Terminal condition violation
        goal = self.problem_parameters["goal_vec"].value
        final_violation = np.sum(np.abs(X[:, -1] - goal))
        print("final_violation = ", final_violation)

        # planets constraint violation
        sat_radius = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + max(self.sg.l_f, self.sg.l_r) ** 2)

        p_violation = 0.0
        for k in range(K):
            sat_x = X[0, k]
            sat_y = X[1, k]
            for planet in self.planets.values():
                obs_x = planet.center[0]
                obs_y = planet.center[1]
                obs_r = planet.radius

                r_safe = sat_radius + obs_r
                dist = np.sqrt((sat_x - obs_x) ** 2 + (sat_y - obs_y) ** 2)
                p_violation += np.maximum(r_safe - dist, 0.0)

        print("p_violation = ", p_violation)
        a_violation = 0.0
        num_asteroids = len(self.asteroids)
        if num_asteroids != 0:
            final_time = float(p[0])
            for k in range(K):
                tau = k / (K - 1)
                t_k = tau * final_time
                sat_x = X[0, k]
                sat_y = X[1, k]

                for asteroid in self.asteroids.values():
                    start_x = asteroid.start[0]
                    start_y = asteroid.start[1]
                    vel_x = asteroid.velocity[0]
                    vel_y = asteroid.velocity[1]
                    obs_r = asteroid.radius
                    # Calculate asteroid position at time t_k
                    obs_x = start_x + vel_x * t_k
                    obs_y = start_y + vel_y * t_k

                    r_safe = sat_radius + obs_r
                    dist = np.sqrt((sat_x - obs_x) ** 2 + (sat_y - obs_y) ** 2)
                    a_violation += np.maximum(r_safe - dist, 0.0)

        print("a_violation = ", a_violation)

        # Total penalty replacing the slack variables
        slack_penalty = lam * (dyn_violation + init_violation + final_violation + p_violation)
        if num_asteroids != 0:
            slack_penalty += lam * a_violation

        # Final nonlinear cost
        J = time_cost + slack_penalty

        return float(J)

    # @staticmethod
    def _extract_seq_from_array(self) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Create a DgSampledSequence from numpy arrays and timestamps.
        """
        K = self.params.K

        # Timestamps from 0 to final time
        t_final = float(self.variables["p"].value)
        ts = tuple((i * t_final / (K - 1) for i in range(K)))

        # Commands
        cmds_list = []
        for k in range(K):
            F_left = float(self.variables["U"].value[0, k])
            F_right = float(self.variables["U"].value[1, k])
            cmds_list.append(SatelliteCommands(F_left, F_right))

        cmds_seq = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # States
        state_list = []
        for k in range(K):
            x = float(self.variables["X"].value[0, k])
            y = float(self.variables["X"].value[1, k])
            psi = float(self.variables["X"].value[2, k])
            vx = float(self.variables["X"].value[3, k])
            vy = float(self.variables["X"].value[4, k])
            dpsi = float(self.variables["X"].value[5, k])
            state_list.append(SatelliteState(x, y, psi, vx, vy, dpsi))

        state_seq = DgSampledSequence[SatelliteState](timestamps=ts, values=state_list)

        return cmds_seq, state_seq
