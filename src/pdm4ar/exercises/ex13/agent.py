from dataclasses import dataclass
from math import pi
from typing import Sequence
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters
from shapely import LineString

from pdm4ar.exercises.ex13.planner import SatellitePlanner
from pdm4ar.exercises_def.ex13.goal import SpaceshipTarget, DockingTarget
from pdm4ar.exercises_def.ex13.utils_params import PlanetParams, AsteroidParams
from pdm4ar.exercises_def.ex13.utils_plot import plot_traj


# HINT: as a good practice we suggest to use the config class to centralise activation of the debugging options
class Config:
    PLOT = True
    VERBOSE = False


@dataclass(frozen=True)
class MyAgentParams:
    """
    You can for example define some agent parameters.
    """

    my_tol: float = 0.1


class SatelliteAgent(Agent):
    # How does it enter in the simulation? The SpaceshipAgent object is created as value
    # corresponding to key "PDM4ARSpaceship" in dict "players", which is an attribute of
    # SimContext returned by "sim_context_from_yaml" in utils_config.py
    """
    This is the PDM4AR agent.
    Do *NOT* modify this class name
    Do *NOT* modify the naming of the existing methods and input/output types.
    """

    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.plan_start_time = 0.0
        self.max_replanning = 5
        self.replanning_iter = 0

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        print("Inizio simulazione")
        self.actual_trajectory = []
        self.myname = init_sim_obs.my_name
        self.sg = init_sim_obs.model_geometry
        self.sp = init_sim_obs.model_params
        self.static_obstacles = init_sim_obs.dg_scenario.static_obstacles

        if self.static_obstacles:
            for obs in self.static_obstacles:
                if hasattr(obs, "shape") and isinstance(obs.shape, LineString):
                    coords = np.array(obs.shape.coords)
                    boundary_dict = {
                        "x_min": np.min(coords[:, 0]),
                        "x_max": np.max(coords[:, 0]),
                        "y_min": np.min(coords[:, 1]),
                        "y_max": np.max(coords[:, 1]),
                    }
                    break

        self.planner = SatellitePlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
            goal=init_sim_obs.goal,
            boundaries=boundary_dict,
        )
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        # make sure you consider both types of goals accordingly
        # (Docking is a subclass of SpaceshipTarget and may require special handling
        # to take into account the docking structure)
        self.goal_state = init_sim_obs.goal.target

        # Plot docking station (this is optional, for better visualization)
        if Config.PLOT and isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, half_p_angle = init_sim_obs.goal.get_landing_constraint_points()
            init_sim_obs.goal.plot_landing_points(A, B, C, A1, A2)

        ##############################################
        # TO DO: Implement Compute Initial Trajectory#
        ##############################################

        # Don't think there's nothing to do here. At the beginning of the simulation, we compute the the full planned trajectory,
        # which is then recomputed only if necessary.
        self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)
        self.plan_start_time = 0.0
        self.max_replanning = 7
        self.replanning_iter = 0

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """
        c_state = sim_obs.players[self.myname].state
        assert isinstance(c_state, SatelliteState)
        current_state = c_state

        # time relative to start of the current plan
        t_rel = float(sim_obs.time) - self.plan_start_time
        self.actual_trajectory.append(current_state)
        expected_state = self.state_traj.at_interp(t_rel)

        # plotting the trajectory every 2.5 sec (this is optional, for better visualization)
        if Config.PLOT and int(10 * sim_obs.time) % 25 == 0:
            plot_traj(self.state_traj, self.actual_trajectory)

        """pos_tol = 0.5
        dir_tol = pi / 6

        dx = current_state.x - expected_state.x
        dy = current_state.y - expected_state.y
        diff = current_state.psi - expected_state.psi
        dpsi = np.abs((diff + np.pi) % (2 * np.pi) - np.pi)

        pos_error = (dx**2 + dy**2) ** 0.5
        angle_error = abs(dpsi)

        if self.replanning_iter <= self.max_replanning:
            if pos_error > pos_tol or angle_error > dir_tol:
                self.replanning_iter += 1
                self.cmds_plan, self.state_traj = self.planner.compute_trajectory(current_state, self.goal_state)
                # new plan starts "now"
                self.plan_start_time = float(sim_obs.time)
                # for this same step, recompute command using new plan
                t_rel = 0.0"""

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(sim_obs.time)

        return cmds

        # return SatelliteCommands(
        #     F_left=1, F_right=1
        # )  # can be replaced by SatelliteCommands(F_left=1, F_right=1) if you want to test constant commands
