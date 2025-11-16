import ast
from dataclasses import dataclass, field
from typing import Union

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)

from pdm4ar.exercises.ex13.discretization import *
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

        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

    def compute_trajectory(
        self, init_state: SatelliteState, goal_state: DynObstacleState
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        """
        self.init_state = init_state
        self.goal_state = goal_state

        #
        # TODO: Implement SCvx algorithm or comparable
        #

        """
        for SCvx it would follow a logic similar to:
        
        initial guess interpolation
        while stopping criterion not satisfied
            convexify
            discretize
            solve convex sub problem
            update trust region
            update stopping criterion
        """

        self._convexification()
        try:
            error = self.problem.solve(verbose=self.params.verbose_solver, solver=self.params.solver)
        except cvx.SolverError:
            print(f"SolverError: {self.params.solver} failed to solve the problem.")

        # Example data: sequence from array
        mycmds, mystates = self._extract_seq_from_array()

        return mycmds, mystates

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Define initial guess for SCvx.
        """
        K = self.params.K
        n_x = self.satellite.n_x
        n_u = self.satellite.n_u
        n_p = self.satellite.n_p

        X = np.zeros((n_x, K))
        U = np.zeros((n_u, K))
        p = np.zeros((n_p))

        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

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
            # slack
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
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(n_x),
            # tolerances
            "pos_tol": cvx.Parameter(nonneg=True),
            "dir_tol": cvx.Parameter(nonneg=True),
            "vel_tol": cvx.Parameter(nonneg=True),
            # max_time
            "p_max": cvx.Parameter(nonneg=True),
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

        P = self.problem_parameters

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

        goal = P["goal_state"]
        constraints = [
            # initial state
            X[:, 0] - P["init_state"] == nu_ic,
            # final state
            # position
            cvx.abs(X[0, K - 1] - goal[0]) <= P["pos_tol"] + nu_tc[0],
            cvx.abs(X[1, K - 1] - goal[1]) <= P["pos_tol"] + nu_tc[1],
            # orientation
            cvx.abs(X[2, K - 1] - goal[2]) <= P["dir_tol"] + nu_tc[2],
            # velocity
            cvx.abs(X[3, K - 1] - goal[3]) <= P["vel_tol"] + nu_tc[3],
            cvx.abs(X[4, K - 1] - goal[4]) <= P["vel_tol"] + nu_tc[4],
            cvx.abs(X[5, K - 1] - goal[5]) <= P["vel_tol"] + nu_tc[5],
            # control inputs at start and goal
            U[:, 0] == 0,
            U[:, K - 1] == 0,
            # control inputs within limits
            U >= self.sp.F_limits[0],
            U <= self.sp.F_limits[1],
            # max_time
            p <= P["p_max"],
            # positive slack variables
            nu_tc >= 0,
            nu_s >= 0,
        ]

        constraints = constraints + dynamic_constraints + obstacles_constraints

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

        objective = self.params.weight_p @ p + slack_cost + travelled_distance + average_input

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

        self.problem_parameters["init_state"].value = self.X_bar[:, 0]
        # ...

    def _check_convergence(self) -> bool:
        """
        Check convergence of SCvx.
        """

        pass

    def _update_trust_region(self):
        """
        Update trust region radius.
        """
        pass

    @staticmethod
    def _extract_seq_from_array() -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Example of how to create a DgSampledSequence from numpy arrays and timestamps.
        """
        ts = (0, 1, 2, 3, 4)
        # in case my planner returns 3 numpy arrays
        F = np.array([0, 1, 2, 3, 4])
        ddelta = np.array([0, 0, 0, 0, 0])
        cmds_list = [SatelliteCommands(f, dd) for f, dd in zip(F, ddelta)]
        mycmds = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds_list)

        # in case my state trajectory is in a 2d array
        npstates = np.random.rand(len(ts), 6)
        states = [SatelliteState(*v) for v in npstates]
        mystates = DgSampledSequence[SatelliteState](timestamps=ts, values=states)
        return mycmds, mystates
