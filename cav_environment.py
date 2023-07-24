import math

import akro
import numpy as np
import scipy.io as sio
from garage import Environment, EnvSpec, EnvStep, StepType

class CAVVelEnv(Environment):
    """A simple CAV point environment.

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
                 goal=np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]),
                 never_done=False,
                 max_episode_length=math.inf):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._never_done = never_done

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=(6, ),
                                           dtype=np.float32)
        self._action_space = akro.Box(low=-1,
                                      high=1,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)
        #########################################################################################################################

        self.infeasible_penalty = -10

        ##################################### DNN model parameters ###############################################################
        # computing demnad (in cycles) for feature extraction
        self.delta_1 = 120268800/30
        # computing demnad (in cycles) for feature fusion
        self.delta_2 = 1000/30
        # computing demnad (in cycles) for fast inference
        self.delta_3 = 9255020/30
        # computing demnad (in cycles) for full inference
        self.delta_4 = 2.3000e+09/30

        self.rho = 0.3  # Average early exit ratio in the default DNN model
        self.rho_tilde = 0.6  # Average early exit ratio in the data-fusion DNN model

        self.w = 295936  # feature data size (in bit)

        self.Delta = 0.1  # Delay requirement in s, i.e., 100ms

        ##
        self.kappa = 1e-28
        self.delta = self.delta_1 + self.delta_3 + (1-self.rho) * self.delta_4
        self.delta_tilde = 2*self.delta_1 + self.delta_2 + \
            self.delta_3 + (1-self.rho_tilde) * self.delta_4
        self.delta_hat = self.delta_1 + self.delta_2 + \
            self.delta_3 + (1-self.rho_tilde) * self.delta_4

        ##
        self.w_new = self.w/1e6*500
        self.delta_hat_new = self.delta_hat/1e9*500
        self.Delta_new = self.Delta*500

        self.varphi = np.sqrt(2*np.power(self.delta, 3) /
                              (np.power(self.delta_hat, 2)*self.delta_tilde))

        ############################################### Communication parameters #################################################
        self.noise_power = np.power(10, -104/10)/1000  # -104dBm

        self.Transmit_power = np.power(10, 23/10)/1000  # 23dBm

        self.Interference_constant = 0

        self.bandwidth_basis = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # *1000000

        self.f_center = 6  # Ghz

        self.switch_coeff = 0.4

        ############################################### optimal resource allocation ###################################

        self.obj_opt_2CAV = np.squeeze(sio.loadmat(
            'KKT_opt/KKT_2CAV_opt_data.mat').get('KKT_obj_2CAV'))
        self.obj_opt_1CAV = np.squeeze(sio.loadmat(
            'KKT_opt/KKT_1CAV_opt_data.mat').get('KKT_obj_1CAV'))

        ############################################### CAV environment ###############################################

        self.CAV_pair_num = 2
        self.n_agents = self.CAV_pair_num
        self.n = self.n_agents

        self.n_adversaries = 0


        # All candidate states for transmitter-receiver distances
        self.distances = np.arange(6, 36, 6)

        # state transition prob. matrix for distance states
        self.trans_matrix = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
                                      [0.25, 0.30, 0.25, 0.15, 0.05],
                                      [0.10, 0.25, 0.30, 0.25, 0.10],
                                      [0.05, 0.15, 0.25, 0.30, 0.25],
                                      [0.05, 0.10, 0.20, 0.30, 0.35]])

        self.workloads = np.array([4, 5, 6, 7, 8])
        self.load_trans = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
                                    [0.25, 0.30, 0.25, 0.15, 0.05],
                                    [0.10, 0.25, 0.30, 0.25, 0.10],
                                    [0.05, 0.15, 0.25, 0.30, 0.25],
                                    [0.05, 0.10, 0.20, 0.30, 0.35]])

        self.O_init = 8

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

    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of the entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        self._step_cnt = 0
       
        ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self._step_cnt]
        if band_basis == 7:
            self.Bandwidth_available = np.random.default_rng().choice([6,7],1, [0.5, 0.5])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.default_rng().choice([2,3],1, [0.5, 0.5])
        else:
            self.Bandwidth_available = np.random.default_rng().choice([band_basis-1, band_basis, band_basis+1], 1, [0.25, 0.5, 0.25])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
           
        # Dynmaic workload: number of objects for detection by the DNN model
        self.O_vehs_all = self.O_init * np.ones(self.CAV_pair_num)
        O_norm = (self.O_vehs_all - np.array([4,4]))/4
           
        
        ############################# Distance and channel gain ###############################################
        
        self.Distance_all = 6*np.ones(self.CAV_pair_num)
        Distance_norm = (self.Distance_all - np.array([6,6]))/24

        ########################################## State ######################################################
        self.curr_state = [0]*self.n_agents
        self.pre_action = np.array([1]*self.n_agents)
        for i_agent in range(0, self.n_agents):            
            self.curr_state[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent]]
        # print("self.curr_state:", self.curr_state)   
        first_obs = np.array(self.curr_state).flatten()

        # self._step_cnt = 1
   
        return first_obs, dict(goal=self._goal)

    def step(self, action):
     
        ########################################################################### MADDPG ######################################################################################################
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        
        actions_array = np.array(action)
        actions = []
        for a in actions_array:
            if (a < 0):
                actions.append(0)
            else:
                actions.append(1)
        actions = np.array(actions)
        ################################### Action processing #################################################
        # Number of CAV pairs in cooperation mode
        Activated_CAV_pair_num = sum(actions)
        Activated_index = np.where(actions == 1)
        # Activated_index = np.array(Activated_index)[0].tolist()
        # print("Activated_index:", Activated_index)   
        
        ################################### Given state and action ############################################
        
        obj_opt = 0

        if Activated_CAV_pair_num > 0:
            O_vehs = self.O_vehs_all[Activated_index]
            Distance = self.Distance_all[Activated_index]
            if Activated_CAV_pair_num > 1:
                sort_index = np.array(Distance).argsort()
                O_vehs_ordered = O_vehs[sort_index]
                Distance_ordered = Distance[sort_index]

        if Activated_CAV_pair_num == 2:
            obj_opt = self.obj_opt_2CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1)]
        elif Activated_CAV_pair_num == 1:
            obj_opt = self.obj_opt_1CAV[int(self.Bandwidth_available-2), int(O_vehs[0]-4), int(Distance[0]/3-1)]

        ####################################### Calculate the reward ##########################################
        reward = 0
        reward_modified = 0

        if obj_opt == -1:           
            obj_modified = 0
            actions_modified = np.array([0,0])           
            switch_cost = self.CAV_pair_num - np.count_nonzero(actions_modified == self.pre_action)
            reward_modified = obj_modified - switch_cost * self.switch_coeff
            reward = reward_modified + self.infeasible_penalty
        else:
            obj_modified = obj_opt    
            actions_modified = actions  
            switch_cost = self.CAV_pair_num - np.count_nonzero(actions_modified == self.pre_action)
            reward = obj_opt - switch_cost * self.switch_coeff
            reward_modified = reward
        
        self.pre_action = actions_modified

        ####################################### Get the next state ##########################################
        
         ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self._step_cnt]
        if band_basis == 7:
            self.Bandwidth_available = np.random.default_rng().choice([6,7],1, [0.5, 0.5])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.default_rng().choice([2,3],1, [0.5, 0.5])
        else:
            self.Bandwidth_available = np.random.default_rng().choice([band_basis-1, band_basis, band_basis+1], 1, [0.25, 0.5, 0.25])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
        O_next = np.zeros([self.n_agents])
        for i_agent in range(0, self.n_agents): 
            p_O = self.load_trans[int(self.O_vehs_all[i_agent]-4),:]           
            O_next[i_agent] = np.random.default_rng().choice(self.workloads, 1, p=p_O)

        self.O_vehs_all = O_next
        O_norm = (self.O_vehs_all - np.array([4,4]))/4

        
        ############################# Distance and channel gain ###############################################
 
        distance_next = np.zeros([self.n_agents])
        for i_agent in range(0, self.n_agents): 
            p_D = self.trans_matrix[int(self.Distance_all[i_agent]/6-1),:]           
            distance_next[i_agent] = np.random.default_rng().choice(self.distances, 1, p=p_D)

        self.Distance_all = distance_next
        Distance_norm = (self.Distance_all - np.array([6,6]))/24

        ##################################### State ###########################################################
        state_next = [0]*self.n_agents # This is a list
        rew_n = [0]*self.n_agents
        done_n = [0]*self.n_agents

        obj_n = [0]*self.n_agents
        switch_n = [0]*self.n_agents

        self._step_cnt += 1
        succ = False
        if self._step_cnt == self._max_episode_length:
            succ = True
            # self.episode += 1

        
        for i_agent in range(0, self.n_agents):            
            state_next[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent]]
            rew_n[i_agent] = reward
            done_n[i_agent] = succ
            obj_n[i_agent] = obj_modified
            switch_n[i_agent] = switch_cost
        self.curr_state = state_next # This is a list of one dimensional (4,) np arrays

        done = succ and not self._never_done

        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)
        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=np.array(state_next).flatten(),
                       env_info={
                           'task': self._task,
                           'success': succ
                       },
                       step_type=step_type)
        


    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `Environment.render_modes`.
        """

    def visualize(self):
        """Creates a visualization of the environment."""

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
        tasks = [{"goal": np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7])} for _ in range(num_tasks)]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']
