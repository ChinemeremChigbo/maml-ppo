# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:32:12 2023

@author: Kaige
"""

# for 20230131-190919 and 20230201-002841

import os
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
import random
import math
import pandas as pd

import sympy as sp  

from gym import spaces


##########################################################################################################################
def cal_delta(a,b,c,d):
    p = (3*a*c-b*b)/(3*a*a)
    q = (27*a*a*d - 9*a*b*c + 2*b*b*b)/(27*a*a*a)
    delta = (q/2)*(q/2)+(p/3)*(p/3)*(p/3)
    return delta

       
##################################################################################


class Network():
    def __init__(self):

        #########################################################################################################################
         
        self.infeasible_penalty = -10
        
        ##################################### DNN model parameters ###############################################################
        self.delta_1 = 120268800/30 # computing demnad (in cycles) for feature extraction
        self.delta_2 = 1000/30 # computing demnad (in cycles) for feature fusion
        self.delta_3 = 9255020/30 # computing demnad (in cycles) for fast inference
        self.delta_4 = 2.3000e+09/30 # computing demnad (in cycles) for full inference
        
        self.rho = 0.3; # Average early exit ratio in the default DNN model
        self.rho_tilde = 0.6;  # Average early exit ratio in the data-fusion DNN model
        
        self.w = 295936; # feature data size (in bit)
        
        self.Delta = 0.1; # Delay requirement in s, i.e., 100ms
        
        ##
        self.kappa = 1e-28
        self.delta = self.delta_1 + self.delta_3 + (1-self.rho) * self.delta_4
        self.delta_tilde = 2*self.delta_1 + self.delta_2 + self.delta_3 + (1-self.rho_tilde) * self.delta_4
        self.delta_hat = self.delta_1 + self.delta_2 + self.delta_3 + (1-self.rho_tilde) * self.delta_4
        
        ##
        self.w_new = self.w/1e6*500
        self.delta_hat_new = self.delta_hat/1e9*500
        self.Delta_new = self.Delta*500
        
        
        self.varphi = np.sqrt(2*np.power(self.delta,3)/(np.power(self.delta_hat,2)*self.delta_tilde)) 
        
        ############################################### Communication parameters #################################################
        self.noise_power = np.power(10, -104/10)/1000 # -104dBm
         
        self.Transmit_power = np.power(10, 23/10)/1000 # 23dBm
        
        
        self.Interference_constant = 0 

        self.bandwidth_basis = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                                6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                                6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                                7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] #*1000000


        self.f_center = 6 # Ghz

        self.episode_length = 75



        ############################################### optimal resource allocation ###################################

        self.obj_opt_3CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_3CAV_opt_data_all.mat').get('KKT_obj_3CAV_all'))
        self.obj_opt_2CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_2CAV_opt_data.mat').get('KKT_obj_2CAV'))
        self.obj_opt_1CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_1CAV_opt_data.mat').get('KKT_obj_1CAV'))

        # self.obj_opt_3CAV = np.zeros([6,5,5,5,10,10,10])
        # self.obj_opt_2CAV = np.zeros([6,5,5,10,10])
        # self.obj_opt_1CAV = np.zeros([6,5,10])



        ############################################### CAV environment ###############################################
       
        self.CAV_pairs = 2

        self.n_adversaries = 0

        self.action_space = [spaces.Discrete(2) for i in range(self.CAV_pairs)]
        # self.observation_space =  [spaces.Box(5,) for i in range(self.n)]
        self.observation_space =  [(6,) for i in range(self.CAV_pairs)] # bandwidth, own workload, own distance, avg workload of others, average distance of others

        self.distances = np.arange(6,36,6) # All candidate states for transmitter-receiver distances
        # state transition prob. matrix for distance states
        self.trans_matrix = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
                                    [0.25, 0.30, 0.25, 0.15, 0.05],
                                    [0.10, 0.25, 0.30, 0.25, 0.10],
                                    [0.05, 0.15, 0.25, 0.30, 0.25],
                                    [0.05, 0.10, 0.20, 0.30, 0.35]])

        # self.trans_matrix = np.array([[0.9, 0.1, 0, 0, 0],
        #                             [0.05, 0.9, 0.05, 0, 0],
        #                             [0, 0.05, 0.9, 0.05, 0],
        #                             [0, 0, 0.05, 0.9, 0.05],
        #                             [0, 0, 0, 0.1, 0.9]])



        self.workloads = np.array([4,5,6,7,8])
        self.load_trans = np.array([[0.35, 0.30, 0.20, 0.10, 0.05],
                                    [0.25, 0.30, 0.25, 0.15, 0.05],
                                    [0.10, 0.25, 0.30, 0.25, 0.10],
                                    [0.05, 0.15, 0.25, 0.30, 0.25],
                                    [0.05, 0.10, 0.20, 0.30, 0.35]])

        # self.workloads = np.array([4,5,6,7,8])
        # self.load_trans = np.array([[0.55, 0.24, 0.12, 0.06, 0.03],
        #                             [0.16, 0.575, 0.16, 0.07, 0.035],
        #                             [0.07, 0.14, 0.58, 0.14, 0.07],
        #                             [0.035, 0.07, 0.16, 0.575, 0.16],
        #                             [0.03, 0.06, 0.12, 0.24, 0.55]])
        # print(np.sum(self.load_trans,axis=1))
        # self.load_trans = np.array([[0.9, 0.1, 0, 0, 0],
        #                             [0.05, 0.9, 0.05, 0, 0],
        #                             [0, 0.05, 0.9, 0.05, 0],
        #                             [0, 0, 0.05, 0.9, 0.05],
        #                             [0, 0, 0, 0.1, 0.9]])

        self.O_init = 8

        
    def reset(self):
        
        self.step_count = 0
       
        ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self.step_count]
        if band_basis == 7:
            self.Bandwidth_available = np.random.default_rng().choice([6,7],1, [0.5, 0.5])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.default_rng().choice([2,3],1, [0.5, 0.5])
        else:
            self.Bandwidth_available = np.random.default_rng().choice([band_basis-1, band_basis, band_basis+1], 1, [0.25, 0.5, 0.25])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
           
        # Dynmaic workload: number of objects for detection by the DNN model
        self.O_vehs_all = self.O_init * np.ones(self.CAV_pairs)
        O_norm = (self.O_vehs_all - np.array([4,4,4]))/4
           
        
        ############################# Distance and channel gain ###############################################
        
        self.Distance_all = 6*np.ones(self.CAV_pairs)
        Distance_norm = (self.Distance_all - np.array([6,6,6]))/24
        
        # Path_loss_dB_all = 32.4 + 20*np.log10(self.Distance_all) + 20*math.log10(self.f_center) # NR-V2X 37.885 highway case, d in meter, f_center in GhZ, Path_loss_dB in dB       
        # self.Channel_gain_all = 1/np.power(10, Path_loss_dB_all/10)
        

        ########################################## State ######################################################
        self.curr_state = [0]*self.CAV_pairs
        self.pre_action = np.array([1]*self.CAV_pairs)
        self.pre_action_BFoptimal = np.array([1]*self.CAV_pairs)
        self.pre_action_random = np.array([1]*self.CAV_pairs)
        # print("self.pre_action_BFoptimal:",self.pre_action_BFoptimal)
        
        for i_agent in range(0, self.CAV_pairs):            
            self.curr_state[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.CAV_pairs-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.CAV_pairs-1),self.pre_action[i_agent]]
            # self.curr_state[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.n_agents-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.n_agents-1)]

        # print("self.curr_state:", self.curr_state)   
        state = self.curr_state

        self.step_count = 1
   
        return state
    
    
    
    def step(self, curr_action):

        # switch_coeff = 0
        switch_coeff = 0.4

        ########################################################################### Brute force optimal ######################################################################################################
        # of all candidate action from 000 to 111
        switch_costs_BF_optimal = np.zeros([2,2,2])
        objs_BF_optimal = np.zeros([2,2,2])
        rewards_BF_optimal = np.zeros([2,2,2])
        for action_agent_0 in range(0,2):
            for action_agent_1 in range(0,2):
                for action_agent_2 in range(0,2):
                    actions_temp = np.array([action_agent_0, action_agent_1, action_agent_2])
                    # print("actions_temp:",actions_temp)
                    # print("self.pre_action_BFoptimal:",self.pre_action_BFoptimal)
                    switch_cost = self.CAV_pairs - np.count_nonzero(actions_temp == self.pre_action_BFoptimal)
                    # print(switch_cost)

                    Activated_CAV_pair_num = sum(actions_temp)
                    Activated_index = np.where(actions_temp == 1)

        
                    obj_opt = 0

                    if Activated_CAV_pair_num > 0:
                        O_vehs = self.O_vehs_all[Activated_index]
                        Distance = self.Distance_all[Activated_index]
                        # print("O_vehs:", O_vehs)    
                        # print("Distance:", Distance)
                        if Activated_CAV_pair_num > 1:
                            sort_index = np.array(Distance).argsort()
                            # print("sort_index:", sort_index)
                            O_vehs_ordered = O_vehs[sort_index]
                            Distance_ordered = Distance[sort_index]

                    if Activated_CAV_pair_num == 3:
                        obj_opt = self.obj_opt_3CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(O_vehs_ordered[2]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1), int(Distance_ordered[2]/3-1)]
                    elif Activated_CAV_pair_num == 2:
                        obj_opt = self.obj_opt_2CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1)]
                    elif Activated_CAV_pair_num == 1:
                        obj_opt = self.obj_opt_1CAV[int(self.Bandwidth_available-2), int(O_vehs[0]-4), int(Distance[0]/3-1)]
                   
                    reward = 0
                    if obj_opt == -1:
                        reward = self.infeasible_penalty # No impact, as the brute-force will select 000, reward = 0
                    else:
                        reward = obj_opt - switch_cost*switch_coeff


                    switch_costs_BF_optimal[action_agent_0, action_agent_1, action_agent_2] = switch_cost
                    objs_BF_optimal[action_agent_0, action_agent_1, action_agent_2] = obj_opt
                    rewards_BF_optimal[action_agent_0, action_agent_1, action_agent_2] = reward

        # print("rewards_BF_optimal:",rewards_BF_optimal)
        # print("switch_costs_BF_optimal:",switch_costs_BF_optimal)
        # print("objs_BF_optimal:",objs_BF_optimal)
        
        reward_BF_optimal = np.max(rewards_BF_optimal)
        # print("reward_BF_optimal:",reward_BF_optimal)
        actions_BF_optimal_cand = np.where(rewards_BF_optimal==reward_BF_optimal)
        # print(actions_BF_optimal_cand[0][0])
        # print(actions_BF_optimal_cand[1][0])
        # print(actions_BF_optimal_cand[2][0])
        

        
        reward_BF_optimal = rewards_BF_optimal[actions_BF_optimal_cand]
        switch_cost_BF_optimal = switch_costs_BF_optimal[actions_BF_optimal_cand]
        obj_BF_optimal = objs_BF_optimal[actions_BF_optimal_cand]

        actions_BF_optimal = np.array([actions_BF_optimal_cand[0][0], actions_BF_optimal_cand[1][0], actions_BF_optimal_cand[2][0]])
        self.pre_action_BFoptimal = actions_BF_optimal 
        # print("actions_BF_optimal:",actions_BF_optimal)  

        
        # print("reward_BF_optimal:",reward_BF_optimal)
        # print("switch_cost_BF_optimal:",switch_cost_BF_optimal)
        # print("obj_BF_optimal:",obj_BF_optimal)


        ########################################################################### Random ######################################################################################################
        
        
        actions_random = np.random.randint(2, size=3)
        

        Activated_CAV_pair_num = sum(actions_random)
        Activated_index = np.where(actions_random == 1)
         
        obj_opt = 0

        if Activated_CAV_pair_num > 0:
            O_vehs = self.O_vehs_all[Activated_index]
            Distance = self.Distance_all[Activated_index]
            # print("O_vehs:", O_vehs)    
            # print("Distance:", Distance)
            if Activated_CAV_pair_num > 1:
                sort_index = np.array(Distance).argsort()
                # print("sort_index:", sort_index)
                O_vehs_ordered = O_vehs[sort_index]
                Distance_ordered = Distance[sort_index]

        if Activated_CAV_pair_num == 3:
            obj_opt = self.obj_opt_3CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(O_vehs_ordered[2]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1), int(Distance_ordered[2]/3-1)]
        elif Activated_CAV_pair_num == 2:
            obj_opt = self.obj_opt_2CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1)]
        elif Activated_CAV_pair_num == 1:
            obj_opt = self.obj_opt_1CAV[int(self.Bandwidth_available-2), int(O_vehs[0]-4), int(Distance[0]/3-1)]
       
        reward_random = 0
        
        if obj_opt == -1:
            # reward_random = self.infeasible_penalty
            actions_random = np.array([0,0,0])
            obj_opt = 0
            
        switch_cost_random = self.CAV_pairs - np.count_nonzero(actions_random == self.pre_action_random)
        reward_random = obj_opt - switch_cost_random*switch_coeff

        obj_random = obj_opt
        
        self.pre_action_random = actions_random

        # print("reward_random:",reward_random)
        # print("switch_cost_random:",switch_cost_random)
        # print("obj_random:",obj_random)
        
        ########################################################################### MADDPG ######################################################################################################
        
        actions_array = np.array(curr_action)
        # print("actions_array:", actions_array)
        actions = np.array([np.argmax(actions_array[0]), np.argmax(actions_array[1]), np.argmax(actions_array[2])])
        # print("actions:", actions)

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
            # print("O_vehs:", O_vehs)    
            # print("Distance:", Distance)
            if Activated_CAV_pair_num > 1:
                sort_index = np.array(Distance).argsort()
                # print("sort_index:", sort_index)
                O_vehs_ordered = O_vehs[sort_index]
                Distance_ordered = Distance[sort_index]

        if Activated_CAV_pair_num == 3:
            obj_opt = self.obj_opt_3CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(O_vehs_ordered[2]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1), int(Distance_ordered[2]/3-1)]
        elif Activated_CAV_pair_num == 2:
            obj_opt = self.obj_opt_2CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1)]
        elif Activated_CAV_pair_num == 1:
            obj_opt = self.obj_opt_1CAV[int(self.Bandwidth_available-2), int(O_vehs[0]-4), int(Distance[0]/3-1)]
       
        ####################################### Calculate the reward ##########################################
        reward = 0
        reward_modified = 0

        if obj_opt == -1:           
            obj_modified = 0
            actions_modified = np.array([0,0,0])           
            switch_cost = self.CAV_pairs - np.count_nonzero(actions_modified == self.pre_action)
            reward_modified = obj_modified - switch_cost*switch_coeff
            reward = reward_modified + self.infeasible_penalty
        else:
            obj_modified = obj_opt    
            actions_modified = actions  
            switch_cost = self.CAV_pairs - np.count_nonzero(actions_modified == self.pre_action)
            reward = obj_opt - switch_cost*switch_coeff
            reward_modified = reward
        # print("reward_modified:",reward_modified)
        # if obj_opt != -1:
        #     reward = obj_opt - switch_cost/10

        
        
        # print("switch_cost:", switch_cost)
        self.pre_action = actions_modified
        # print("self.pre_action:", self.pre_action)


        ####################################### Get the next state ##########################################
        
         ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self.step_count]
        if band_basis == 7:
            self.Bandwidth_available = np.random.default_rng().choice([6,7],1, [0.5, 0.5])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.default_rng().choice([2,3],1, [0.5, 0.5])
        else:
            self.Bandwidth_available = np.random.default_rng().choice([band_basis-1, band_basis, band_basis+1], 1, [0.25, 0.5, 0.25])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
        O_next = np.zeros([self.CAV_pairs])
        for i_agent in range(0, self.CAV_pairs): 
            p_O = self.load_trans[int(self.O_vehs_all[i_agent]-4),:]           
            O_next[i_agent] = np.random.default_rng().choice(self.workloads, 1, p=p_O)

        self.O_vehs_all = O_next
        O_norm = (self.O_vehs_all - np.array([4,4,4]))/4

        
        ############################# Distance and channel gain ###############################################
 
        distance_next = np.zeros([self.CAV_pairs])
        for i_agent in range(0, self.CAV_pairs): 
            p_D = self.trans_matrix[int(self.Distance_all[i_agent]/6-1),:]           
            distance_next[i_agent] = np.random.default_rng().choice(self.distances, 1, p=p_D)

        self.Distance_all = distance_next
        Distance_norm = (self.Distance_all - np.array([6,6,6]))/24

        # print("self.O_vehs_all:", self.O_vehs_all)
        # print("self.Distance_all:", self.Distance_all)
       

        ##################################### State ###########################################################
        state_next = [0]*self.CAV_pairs # This is a list
        rew_n = [0]*self.CAV_pairs
        done_n = [0]*self.CAV_pairs

        obj_n = [0]*self.CAV_pairs
        switch_n = [0]*self.CAV_pairs

        self.step_count += 1
        done = False
        if self.step_count == self.episode_length:
            done = True
            # self.episode += 1

        
        for i_agent in range(0, self.CAV_pairs):            
            state_next[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.CAV_pairs-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.CAV_pairs-1),self.pre_action[i_agent]]
            # state_next[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.n_agents-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.n_agents-1)]
            rew_n[i_agent] = reward
            done_n[i_agent] = done
            obj_n[i_agent] = obj_modified
            switch_n[i_agent] = switch_cost

        self.curr_state = state_next # This is a list of one dimensional (4,) np arrays

 
        
        return state_next, rew_n, done_n, obj_n, switch_n, reward_modified, reward_BF_optimal[0], obj_BF_optimal[0], switch_cost_BF_optimal[0], reward_random, obj_random, switch_cost_random



# ###-------------------------------
# env = Network()
# x = env.reset()
# print("Initial state:", x)

# curr_action = np.random.randint(0,2,env.CAV_pair_num)
# print("actions",curr_action)

# w, y, z = env.step(curr_action)
# print("reward:", y)
# print("state_next:", w)
# print("done:", z)

# curr_action = np.random.randint(0,2,env.CAV_pair_num)
# print("actions",curr_action)

# w, y, z = env.step(curr_action)
# print("reward:", y)
# print("state_next:", w)
# print("done:", z)

# curr_action = np.random.randint(0,2,env.CAV_pair_num)
# print("actions",curr_action)

# w, y, z = env.step(curr_action)
# print("reward:", y)
# print("state_next:", w)
# print("done:", z)

# curr_action = np.random.randint(0,2,env.CAV_pair_num)
# print("actions",curr_action)

# w, y, z = env.step(curr_action)
# print("reward:", y)
# print("state_next:", w)
# print("done:", z)


  