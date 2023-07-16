import numpy as np
# from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
from cav_environment import CAVEnv

class HalfCheetahEnv(CAVEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            width, height = 500, 500
            self._get_viewer(mode=mode).render(width, height)
            # window size used for old mujoco-py:

            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode=mode).render()

class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self, task={}):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']

class HalfCheetahDirEnv(HalfCheetahEnv):
    def __init__(self, task={}):
        self._task = task
        self._goal_dir = task.get('direction', 1)
        super(HalfCheetahDirEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        directions = 2 * self.np_random.binomial(1, p=0.5, size=(num_tasks,)) - 1
        tasks = [{'direction': direction} for direction in directions]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_dir = task['direction']
