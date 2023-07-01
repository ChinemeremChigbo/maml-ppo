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
         
        self.infeasible_penalty = -100
        
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

        # self.bandwidth_basis = [7,7,7,6,6,6,5,5,5,4,4,4,3,3,3,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7] #*1000000
        self.bandwidth_basis = [7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
                                6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                                3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                                2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                                3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                                4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                                5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,
                                6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,
                                7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7] #*1000000

        self.f_center = 6 # Ghz

        self.episode_length = 192



        ############################################### optimal resource allocation ###################################

        self.obj_opt_3CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_3CAV_opt_data_all.mat').get('KKT_obj_3CAV_all'))
        self.obj_opt_2CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_2CAV_opt_data.mat').get('KKT_obj_2CAV'))
        self.obj_opt_1CAV = np.squeeze(sio.loadmat('KKT_opt/KKT_1CAV_opt_data.mat').get('KKT_obj_1CAV'))

        # self.obj_opt_3CAV = np.zeros([6,5,5,5,10,10,10])
        # self.obj_opt_2CAV = np.zeros([6,5,5,10,10])
        # self.obj_opt_1CAV = np.zeros([6,5,10])



        ############################################### CAV environment ###############################################
       
        self.CAV_pair_num = 3
        self.n_agents = self.CAV_pair_num
        self.n = self.n_agents

        self.n_adversaries = 0

        self.action_space = [spaces.Discrete(2) for i in range(self.n)]
        # self.observation_space =  [spaces.Box(5,) for i in range(self.n)]
        self.observation_space =  [(5,) for i in range(self.n)] # bandwidth, own workload, own distance, avg workload of others, average distance of others

        self.distances = np.arange(6,36,6) # All candidate states for transmitter-receiver distances
        # state transition prob. matrix for distance states
        self.trans_matrix = np.array([[0.54, 0.24, 0.12, 0.06, 0.03],
                                    [0.16, 0.575, 0.16, 0.07, 0.035],
                                    [0.07, 0.14, 0.58, 0.14, 0.07],
                                    [0.035, 0.07, 0.16, 0.575, 0.16],
                                    [0.03, 0.06, 0.12, 0.24, 0.54]])

        self.workloads = np.array([4,5,6,7,8])
        self.load_trans = np.array([[0.54, 0.24, 0.12, 0.06, 0.03],
                                    [0.16, 0.575, 0.16, 0.07, 0.035],
                                    [0.07, 0.14, 0.58, 0.14, 0.07],
                                    [0.035, 0.07, 0.16, 0.575, 0.16],
                                    [0.03, 0.06, 0.12, 0.24, 0.54]])


        
    def reset(self):
        
        self.step_count = 0
       
        ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self.step_count]
        if band_basis == 7:
            self.Bandwidth_available = np.random.choice([6,7],1, [0.2, 0.8])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.choice([2,3],1, [0.8, 0.2])
        else:
            self.Bandwidth_available = np.random.choice([band_basis-1, band_basis, band_basis+1], 1, [0.1, 0.8, 0.1])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
           
        # Dynmaic workload: number of objects for detection by the DNN model
        self.O_vehs_all = np.array([4,6,8])
        O_norm = (self.O_vehs_all - np.array([4,4,4]))/4
           
        
        ############################# Distance and channel gain ###############################################
        
        self.Distance_all = np.array([6,18,30])
        Distance_norm = (self.Distance_all - np.array([6,6,6]))/24
        
        # Path_loss_dB_all = 32.4 + 20*np.log10(self.Distance_all) + 20*math.log10(self.f_center) # NR-V2X 37.885 highway case, d in meter, f_center in GhZ, Path_loss_dB in dB       
        # self.Channel_gain_all = 1/np.power(10, Path_loss_dB_all/10)
        

        ########################################## State ######################################################
        self.curr_state = [0]*self.n_agents
        self.pre_action = np.array([1]*self.n_agents)
        
        for i_agent in range(0, self.n_agents):            
            # self.curr_state[i_agent] = np.r_[self.Bandwidth_available, self.O_vehs_all[i_agent], self.Distance_all[i_agent], (sum(self.O_vehs_all)-self.O_vehs_all[i_agent])/(self.n_agents-1),(sum(self.Distance_all)-self.Distance_all[i_agent])/(self.n_agents-1)]
            self.curr_state[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.n_agents-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.n_agents-1)]
        # print("self.curr_state:", self.curr_state)   
        state = self.curr_state

        self.step_count = 1
   
        return state
    
    
    
    def step(self, curr_action):
        
        #########################################################################################################
        
        actions_array = np.array(curr_action)
        # print("actions_array:", actions_array)
        actions = np.array([np.argmax(actions_array[0]), np.argmax(actions_array[1]), np.argmax(actions_array[2])])
        # print("actions:", actions)
        # print("self.pre_action:", self.pre_action)
        switch_cost = self.CAV_pair_num - np.count_nonzero(actions == self.pre_action)
        # print("switch_cost:", switch_cost)
        self.pre_action = actions
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

                # print("O_vehs_ordered:", O_vehs_ordered)
                # print("Distance_ordered:", Distance_ordered)

        # print(O_vehs_ordered[0]-4)
        # print(Distance_ordered[0]/3-1)

        if Activated_CAV_pair_num == 3:
            obj_opt = self.obj_opt_3CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(O_vehs_ordered[2]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1), int(Distance_ordered[2]/3-1)]
        elif Activated_CAV_pair_num == 2:
            obj_opt = self.obj_opt_2CAV[int(self.Bandwidth_available-2), int(O_vehs_ordered[0]-4), int(O_vehs_ordered[1]-4), int(Distance_ordered[0]/3-1), int(Distance_ordered[1]/3-1)]
        elif Activated_CAV_pair_num == 1:
            obj_opt = self.obj_opt_1CAV[int(self.Bandwidth_available-2), int(O_vehs[0]-4), int(Distance[0]/3-1)]
        # else: #Activated_CAV_pair_num == 0:
        #     obj_opt = 0

        # print("obj_opt:", obj_opt)


        #    sio.savemat("Results_vs_mu\CAV_num\Results_vs_mu_"+str(CAV_pair_num)+".mat", {"f_1s": f_1s, "mu_hist_array": mu_hist_array, "obj_hist_array": obj_hist_array, "check_sum_hist_array": check_sum_hist_array})
            
        ####################################### Calculate the reward ##########################################
        reward = 0
        if obj_opt == -1:
            reward = self.infeasible_penalty
        else:
            reward = obj_opt #- switch_cost/10

        # if obj_opt != -1:
        #     reward = obj_opt #- switch_cost/10
            
        ####################################### Get the next state ##########################################
        
         ####################### Available bandwidth (depending on HDV status) #####################################################
        band_basis = self.bandwidth_basis[self.step_count]
        if band_basis == 7:
            self.Bandwidth_available = np.random.choice([6,7],1, [0.2, 0.8])
        elif band_basis == 2:
            self.Bandwidth_available = np.random.choice([2,3],1, [0.8, 0.2])
        else:
            self.Bandwidth_available = np.random.choice([band_basis-1, band_basis, band_basis+1], 1, [0.1, 0.8, 0.1])
        Bandwidth_norm = (self.Bandwidth_available - 2)/5
        ########################### CAV Workload (number of objects for detection) ##################################################################
        
        O_next = np.zeros([self.n_agents])
        for i_agent in range(0, self.n_agents): 
            p_O = self.load_trans[int(self.O_vehs_all[i_agent]-4),:]           
            O_next[i_agent] = np.random.choice(self.workloads, 1, p_O.any())

        self.O_vehs_all = O_next
        O_norm = (self.O_vehs_all - np.array([4,4,4]))/4
        
        ############################# Distance and channel gain ###############################################
 
        distance_next = np.zeros([self.n_agents])
        for i_agent in range(0, self.n_agents): 
            p = self.trans_matrix[int(self.Distance_all[i_agent]/6-1),:]           
            distance_next[i_agent] = np.random.choice(self.distances, 1, p.any())

        self.Distance_all = distance_next
        Distance_norm = (self.Distance_all - np.array([6,6,6]))/24

        # print("self.O_vehs_all:", self.O_vehs_all)
        # print("self.Distance_all:", self.Distance_all)
       

        ##################################### State ###########################################################
        state_next = [0]*self.n_agents # This is a list
        rew_n = [0]*self.n_agents
        done_n = [0]*self.n_agents

        obj_n = [0]*self.n_agents
        switch_n = [0]*self.n_agents

        self.step_count += 1
        done = False
        if self.step_count == self.episode_length:
            done = True
            # self.episode += 1

        
        for i_agent in range(0, self.n_agents):            
            # state_next[i_agent] = np.r_[self.Bandwidth_available, self.O_vehs_all[i_agent], self.Distance_all[i_agent], self.pre_action[i_agent]]
            # state_next[i_agent] = np.r_[self.Bandwidth_available, self.O_vehs_all[i_agent], self.Distance_all[i_agent], (sum(self.O_vehs_all)-self.O_vehs_all[i_agent])/(self.n_agents-1),(sum(self.Distance_all)-self.Distance_all[i_agent])/(self.n_agents-1)] 
            state_next[i_agent] = np.r_[Bandwidth_norm, O_norm[i_agent], Distance_norm[i_agent], (sum(O_norm)-O_norm[i_agent])/(self.n_agents-1),(sum(Distance_norm)-Distance_norm[i_agent])/(self.n_agents-1)]

            rew_n[i_agent] = reward
            done_n[i_agent] = done
            obj_n[i_agent] = obj_opt
            switch_n[i_agent] = switch_cost

        self.curr_state = state_next # This is a list of one dimensional (4,) np arrays

 
        
        return state_next, rew_n, done_n, obj_n, switch_n



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


  