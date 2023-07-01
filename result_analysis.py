import pickle
import os
import scipy.io as sio
import datetime
import numpy as np


ex_path = os.path.join('results', 'sacred', '5')

rew_file_name = os.path.join(ex_path, 'rewards.pkl')
with open(rew_file_name, 'rb') as fp:
    final_ep_rewards = pickle.load(fp)
# print("final_ep_rewards:",final_ep_rewards)

agrew_file_name = os.path.join(ex_path, 'agrewards.pkl')
with open(agrew_file_name, 'rb') as fp:
    final_ep_ag_rewards = pickle.load(fp)
# print("final_ep_ag_rewards :",final_ep_ag_rewards)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_name = ex_path + '/' + current_time + '-learning_results.mat'
print("file_name:", file_name)
sio.savemat(file_name, {"final_ep_rewards": np.array(final_ep_rewards), "final_ep_ag_rewards": np.array(final_ep_ag_rewards)})
