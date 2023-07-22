import math
import random

import akro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import sympy as sp
from garage import Environment, EnvSpec, EnvStep, StepType
from gym import spaces

"""Simple 2D environment containing a point and a goal location."""

class CAVVelEnv(Environment):
    """A simple 2D point environment.

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
        max_episode_length (int): The maximum steps allowed for an episode.

    """

    def __init__(self,
                 goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 done_bonus=0.,
                 never_done=False,
                 max_episode_length=math.inf):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size

        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(3, ),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-0.1,
                                      high=0.1,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)
        # #########################################################################################################################

        # self.infeasible_penalty = -10

        # ##################################### DNN model parameters ###############################################################
        # # computing demnad (in cycles) for feature extraction
        # self.delta_1 = 120268800/30
        # # computing demnad (in cycles) for feature fusion
        # self.delta_2 = 1000/30
        # # computing demnad (in cycles) for fast inference
        # self.delta_3 = 9255020/30
        # # computing demnad (in cycles) for full inference
        # self.delta_4 = 2.3000e+09/30

        # self.rho = 0.3  # Average early exit ratio in the default DNN model
        # self.rho_tilde = 0.6  # Average early exit ratio in the data-fusion DNN model

        # self.w = 295936  # feature data size (in bit)

        # self.Delta = 0.1  # Delay requirement in s, i.e., 100ms

        # ##
        # self.kappa = 1e-28
        # self.delta = self.delta_1 + self.delta_3 + (1-self.rho) * self.delta_4
        # self.delta_tilde = 2*self.delta_1 + self.delta_2 + \
        #     self.delta_3 + (1-self.rho_tilde) * self.delta_4
        # self.delta_hat = self.delta_1 + self.delta_2 + \
        #     self.delta_3 + (1-self.rho_tilde) * self.delta_4

        # ##
        # self.w_new = self.w/1e6*500
        # self.delta_hat_new = self.delta_hat/1e9*500
        # self.Delta_new = self.Delta*500

        # self.varphi = np.sqrt(2*np.power(self.delta, 3) /
        #                       (np.power(self.delta_hat, 2)*self.delta_tilde))

        # ############################################### Communication parameters #################################################
        # self.noise_power = np.power(10, -104/10)/1000  # -104dBm

        # self.Transmit_power = np.power(10, 23/10)/1000  # 23dBm

        # self.Interference_constant = 0

        # self.bandwidth_basis = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
        #                         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        #                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        #                         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        #                         7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # *1000000

        # self.f_center = 6  # Ghz

        # self.episode_length = 75

        # self.switch_coeff = 0.4

        # ############################################### optimal resource allocation ###################################

        # self.obj_opt_2CAV = np.squeeze(sio.loadmat(
        #     'KKT_opt/KKT_2CAV_opt_data.mat').get('KKT_obj_2CAV'))
        # self.obj_opt_1CAV = np.squeeze(sio.loadmat(
        #     'KKT_opt/KKT_1CAV_opt_data.mat').get('KKT_obj_1CAV'))

        # ############################################### CAV environment ###############################################

        # self.CAV_pair_num = 2
        # self.n_agents = self.CAV_pair_num
        # self.n = self.n_agents

        # self.n_adversaries = 0

        # # self.action_space = [spaces.Discrete(2) for i in range(self.n)]

        # # bandwidth, own workload, own distance, avg workload of others, average distance of others
        # # self.observation_space = [(6,) for i in range(self.n)]

        # # All candidate states for transmitter-receiver distances
        # self.distances = np.arange(6, 36, 6)

        # # state transition prob. matrix for distance states
        # self.trans_matrix = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
        #                               [0.25, 0.30, 0.25, 0.15, 0.05],
        #                               [0.10, 0.25, 0.30, 0.25, 0.10],
        #                               [0.05, 0.15, 0.25, 0.30, 0.25],
        #                               [0.05, 0.10, 0.20, 0.30, 0.35]])

        # self.workloads = np.array([4, 5, 6, 7, 8])
        # self.load_trans = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
        #                             [0.25, 0.30, 0.25, 0.15, 0.05],
        #                             [0.10, 0.25, 0.30, 0.25, 0.10],
        #                             [0.05, 0.15, 0.25, 0.30, 0.25],
        #                             [0.05, 0.10, 0.20, 0.30, 0.35]])

        # self.O_init = 8

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        self._point = np.zeros_like(self._goal)
        dist = np.linalg.norm(self._point - self._goal)

        first_obs = np.concatenate([self._point, (dist, )])
        self._step_cnt = 0

        return first_obs, dict(goal=self._goal)

    def step(self, action):
        """Step the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """
        print(f"\n\n\nAction: {action}\nAction_space: {self.action_space}\n Action_space.low {self.action_space.low}\n Action_space.high{self.action_space.high} \n\n\n")
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')

        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)

        self._point = np.clip(self._point + a, -self._arena_size,
                              self._arena_size)
        if self._visualize:
            print(self.render('ascii'))

        dist = np.linalg.norm(self._point - self._goal)
        succ = dist < np.linalg.norm(self.action_space.low)

        # dense reward
        reward = -dist
        # done bonus
        if succ:
            reward += self._done_bonus
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        # sometimes we don't want to terminate
        done = succ and not self._never_done

        obs = np.concatenate([self._point, (dist, )])

        self._step_cnt += 1

        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=obs,
                       env_info={
                           'task': self._task,
                           'success': succ
                       },
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """
        return f'Point: {self._point}, Goal: {self._goal}'

    def visualize(self):
        """Creates a visualization of the environment."""
        self._visualize = True
        print(self.render('ascii'))

    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        # Start with CAV
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']
