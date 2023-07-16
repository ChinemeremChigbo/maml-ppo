# from gym import spaces
# import numpy as np
# from gym.utils import seeding
# from gym.envs.mujoco import mujoco_env
# """Simple 2D environment containing a point and a goal location."""
# import math

# import akro
# import numpy as np

# from garage import Environment, EnvSpec, EnvStep, StepType

# import numpy as np
# from gym import utils
# from gym.envs.mujoco import mujoco_env
"""Simple 2D environment containing a point and a goal location."""
import math

import akro
import numpy as np

from garage import Environment, EnvSpec, EnvStep, StepType


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



# class CAVVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):

#     def __init__(self, task=None):
#         self._task = task or {'direction': 1.}
#         self.spec = EnvSpec(action_space=self.action_space,
#                     observation_space=self.observation_space)
#         mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
#         utils.EzPickle.__init__(self)

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def _get_obs(self):

#         return np.concatenate([
#             self.sim.data.qpos.flat[1:],
#             self.sim.data.qvel.flat,
#             self.get_body_com('torso').flat,
#         ]).astype(np.float32).flatten()

#     def viewer_setup(self):
#         camera_id = self.model.camera_name2id('track')
#         self.viewer.cam.type = 2
#         self.viewer.cam.fixedcamid = camera_id
#         self.viewer.cam.distance = self.model.stat.extent * 0.35
#         self.viewer._hide_overlay = True

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]

#         forward_vel = (xposafter - xposbefore) / self.dt
#         forward_reward = self._task['direction'] * forward_vel
#         ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

#         observation = self._get_obs()
#         reward = forward_reward - ctrl_cost
#         done = False
#         if self._task['direction'] == 1.:
#             task_name = 'fowrad'
#         elif self._task['direction'] == -1.:
#             task_name = 'backward'
#         else:
#             raise ValueError('task direction should be 1. or -1.')
#         infos = dict(reward_forward=np.asarray([forward_reward]),
#                      reward_ctrl=np.asarray([-ctrl_cost]),
#                      task_dir=np.asarray([self._task['direction']]),
#                      task_name=task_name)
#         return observation, reward, done, infos

#     def sample_tasks(self, num_tasks):
#         directions = (
#             2 * self.np_random.binomial(1, p=0.5, size=(num_tasks, )) - 1)
#         tasks = [{'direction': direction} for direction in directions]
#         return tasks

#     def set_task(self, task):
#         self._task = task


# class CAVVelEnv(Environment):
#     """A simple 2D point environment.

#     Args:
#         goal (np.ndarray): A 2D array representing the goal position
#         arena_size (float): The size of arena where the point is constrained
#             within (-arena_size, arena_size) in each dimension
#         done_bonus (float): A numerical bonus added to the reward
#             once the point as reached the goal
#         never_done (bool): Never send a `done` signal, even if the
#             agent achieves the goal
#         max_episode_length (int): The maximum steps allowed for an episode.

#     """

#     def __init__(self,
#                  goal=np.array((1., 1.), dtype=np.float32),
#                  arena_size=5.,
#                  done_bonus=0.,
#                  never_done=False,
#                  max_episode_length=math.inf):
#         goal = np.array(goal, dtype=np.float32)
#         self._goal = goal
#         self._done_bonus = done_bonus
#         self._never_done = never_done
#         self._arena_size = arena_size

#         assert ((goal >= -arena_size) & (goal <= arena_size)).all()

#         self._step_cnt = None
#         self._max_episode_length = max_episode_length
#         self._visualize = False

#         self._point = np.zeros_like(self._goal)
#         self._task = {'goal': self._goal}
#         self._observation_space = akro.Box(low=-np.inf,
#                                            high=np.inf,
#                                            shape=(3, ),
#                                            dtype=np.float32)
#         self._action_space = akro.Box(low=-0.1,
#                                       high=0.1,
#                                       shape=(2, ),
#                                       dtype=np.float32)
#         self._spec = EnvSpec(action_space=self.action_space,
#                              observation_space=self.observation_space,
#                              max_episode_length=max_episode_length)

#     @property
#     def action_space(self):
#         """akro.Space: The action space specification."""
#         return self._action_space

#     @property
#     def observation_space(self):
#         """akro.Space: The observation space specification."""
#         return self._observation_space

#     @property
#     def spec(self):
#         """EnvSpec: The environment specification."""
#         return self._spec

#     @property
#     def render_modes(self):
#         """list: A list of string representing the supported render modes."""
#         return [
#             'ascii',
#         ]

#     def reset(self):
#         """Reset the environment.

#         Returns:
#             numpy.ndarray: The first observation conforming to
#                 `observation_space`.
#             dict: The episode-level information.
#                 Note that this is not part of `env_info` provided in `step()`.
#                 It contains information of he entire episode， which could be
#                 needed to determine the first action (e.g. in the case of
#                 goal-conditioned or MTRL.)

#         """
#         self._point = np.zeros_like(self._goal)
#         dist = np.linalg.norm(self._point - self._goal)

#         first_obs = np.concatenate([self._point, (dist, )])
#         self._step_cnt = 0

#         return first_obs, dict(goal=self._goal)

#     def step(self, action):
#         """Step the environment.

#         Args:
#             action (np.ndarray): An action provided by the agent.

#         Returns:
#             EnvStep: The environment step resulting from the action.

#         Raises:
#             RuntimeError: if `step()` is called after the environment
#             has been
#                 constructed and `reset()` has not been called.

#         """
#         if self._step_cnt is None:
#             raise RuntimeError('reset() must be called before step()!')

#         # enforce action space
#         a = action.copy()  # NOTE: we MUST copy the action before modifying it
#         a = np.clip(a, self.action_space.low, self.action_space.high)

#         self._point = np.clip(self._point + a, -self._arena_size,
#                               self._arena_size)
#         if self._visualize:
#             print(self.render('ascii'))

#         dist = np.linalg.norm(self._point - self._goal)
#         succ = dist < np.linalg.norm(self.action_space.low)

#         # dense reward
#         reward = -dist
#         # done bonus
#         if succ:
#             reward += self._done_bonus
#         # Type conversion
#         if not isinstance(reward, float):
#             reward = float(reward)

#         # sometimes we don't want to terminate
#         done = succ and not self._never_done

#         obs = np.concatenate([self._point, (dist, )])

#         self._step_cnt += 1

#         step_type = StepType.get_step_type(
#             step_cnt=self._step_cnt,
#             max_episode_length=self._max_episode_length,
#             done=done)

#         if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
#             self._step_cnt = None

#         return EnvStep(env_spec=self.spec,
#                        action=action,
#                        reward=reward,
#                        observation=obs,
#                        env_info={
#                            'task': self._task,
#                            'success': succ
#                        },
#                        step_type=step_type)

#     def render(self, mode):
#         """Renders the environment.

#         Args:
#             mode (str): the mode to render with. The string must be present in
#                 `self.render_modes`.

#         Returns:
#             str: the point and goal of environment.

#         """
#         return f'Point: {self._point}, Goal: {self._goal}'

#     def visualize(self):
#         """Creates a visualization of the environment."""
#         self._visualize = True
#         print(self.render('ascii'))

#     def close(self):
#         """Close the env."""
#         if self.viewer is not None:
#             # self.viewer.finish()
#             self.viewer = None
#             self._viewers = {}

#     # pylint: disable=no-self-use
#     def sample_tasks(self, num_tasks):
#         """Sample a list of `num_tasks` tasks.

#         Args:
#             num_tasks (int): Number of tasks to sample.

#         Returns:
#             list[dict[str, np.ndarray]]: A list of "tasks", where each task is
#                 a dictionary containing a single key, "goal", mapping to a
#                 point in 2D space.

#         """
#         goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
#         tasks = [{'goal': goal} for goal in goals]
#         return tasks

#     def set_task(self, task):
#         """Reset with a task.

#         Args:
#             task (dict[str, np.ndarray]): A task (a dictionary containing a
#                 single key, "goal", which should be a point in 2D space).

#         """
#         self._task = task
#         self._goal = task['goal']

# class CAVVelEnv(mujoco_env.MujocoEnv):        
#     metadata = {"render.modes": ["human", "rgb_array"], "render.fps": 4}

#     def __init__(self, render_mode=None, size=5):
#         self.size = size  # The size of the square grid
#         self.window_size = 512  # The size of the PyGame window
#         mujoco_env.MujocoEnv.__init__(self, '', 5)

#         # Observations are dictionaries with the agent's and the target's location.
#         # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
#         self.observation_space = spaces.Dict(
#             {
#                 "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#                 "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#             }
#         )

#         # We have 4 actions, corresponding to "right", "up", "left", "down"
#         self.action_space = spaces.Discrete(4)

#         """
#         The following dictionary maps abstract actions from `self.action_space` to 
#         the direction we will walk in if that action is taken.
#         I.e. 0 corresponds to "right", 1 to "up" etc.
#         """
#         self._action_to_direction = {
#             0: np.array([1, 0]),
#             1: np.array([0, 1]),
#             2: np.array([-1, 0]),
#             3: np.array([0, -1]),
#         }

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         """
#         If human-rendering is used, `self.window` will be a reference
#         to the window that we draw to. `self.clock` will be a clock that is used
#         to ensure that the environment is rendered at the correct framerate in
#         human-mode. They will remain `None` until human-mode is used for the
#         first time.
#         """
#         self.window = None
#         self.clock = None
#         self.seed()

#     def sample_tasks(self, num_tasks):
#         directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
#         tasks = [{'direction': direction} for direction in directions]
#         return tasks

# class CAVEnv_(mujoco_env.MujocoEnv):
#     # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
#     metadata = {
#         'render.modes' : ['human', 'rgb_array'], "render_fps": 4
#     }
#     def __init__(self, render_mode=None, size=5):
#         self.size = size  # The size of the square grid
#         self.window_size = 512  # The size of the PyGame window

#         # Observations are dictionaries with the agent's and the target's location.
#         # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
#         self.observation_space = spaces.Dict(
#             {
#                 "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#                 "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
#             }
#         )

#         # We have 4 actions, corresponding to "right", "up", "left", "down"
#         self.action_space = spaces.Discrete(4)

#         """
#         The following dictionary maps abstract actions from `self.action_space` to 
#         the direction we will walk in if that action is taken.
#         I.e. 0 corresponds to "right", 1 to "up" etc.
#         """
#         self._action_to_direction = {
#             0: np.array([1, 0]),
#             1: np.array([0, 1]),
#             2: np.array([-1, 0]),
#             3: np.array([0, -1]),
#         }

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

#         """
#         If human-rendering is used, `self.window` will be a reference
#         to the window that we draw to. `self.clock` will be a clock that is used
#         to ensure that the environment is rendered at the correct framerate in
#         human-mode. They will remain `None` until human-mode is used for the
#         first time.
#         """
#         self.window = None
#         self.clock = None
#     def _get_obs(self):
#         return {"agent": self._agent_location, "target": self._target_location}
#     def _get_info(self):
#         return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
    

#     def reset(self, seed=None, options=None):
#         # We need the following line to seed self.np_random
#         super().reset(seed=seed)

#         # Choose the agent's location uniformly at random
#         self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

#         # We will sample the target's location randomly until it does not coincide with the agent's location
#         self._target_location = self._agent_location
#         while np.array_equal(self._target_location, self._agent_location):
#             self._target_location = self.np_random.integers(
#                 0, self.size, size=2, dtype=int
#             )

#         observation = self._get_obs()
#         info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

#         return observation, info


# class CAVEnv(CAVEnv_):
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[1:],
#             self.sim.data.qvel.flat,
#             self.get_body_com("torso").flat,
#         ]).astype(np.float32).flatten()

#     def viewer_setup(self):
#         camera_id = self.model.camera_name2id('track')
#         self.viewer.cam.type = 2
#         self.viewer.cam.fixedcamid = camera_id
#         self.viewer.cam.distance = self.model.stat.extent * 0.35
#         # Hide the overlay
#         self.viewer._hide_overlay = True

#     def render(self, mode='human'):
#         if mode == 'rgb_array':
#             width, height = 500, 500
#             self._get_viewer(mode=mode).render(width, height)
#             # window size used for old mujoco-py:

#             data = self._get_viewer(mode).read_pixels(width, height, depth=False)
#             return data
#         elif mode == 'human':
#             self._get_viewer(mode=mode).render()

# class CAVVelEnv(CAVEnv):
#     """Half-cheetah environment with target velocity, as described in [1]. The 
#     code is adapted from
#     https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

#     The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
#     time step a reward composed of a control cost and a penalty equal to the 
#     difference between its current velocity and the target velocity. The tasks 
#     are generated by sampling the target velocities from the uniform 
#     distribution on [0, 2].

#     [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
#         Meta-Learning for Fast Adaptation of Deep Networks", 2017 
#         (https://arxiv.org/abs/1703.03400)
#     [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
#         model-based control", 2012 
#         (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
#     """
#     def __init__(self, task={}):
#         self._task = task
#         self._goal_vel = task.get('velocity', 0.0)
#         super(CAVVelEnv, self).__init__()

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]

#         forward_vel = (xposafter - xposbefore) / self.dt
#         forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
#         ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

#         observation = self._get_obs()
#         reward = forward_reward - ctrl_cost
#         done = False
#         infos = dict(reward_forward=forward_reward,
#             reward_ctrl=-ctrl_cost, task=self._task)
#         return (observation, reward, done, infos)

#     def sample_tasks(self, num_tasks):
#         velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
#         tasks = [{'velocity': velocity} for velocity in velocities]
#         return tasks

#     def reset_task(self, task):
#         self._task = task
#         self._goal_vel = task['velocity']