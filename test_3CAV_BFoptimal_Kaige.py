"""
This file can be used to train an agent on a given environment, i.e. to replicate some
results from the papers.
"""

import time
import os
from typing import List

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver

from tf2marl.common.logger_benchmark import RLLogger
from tf2marl.agents import MADDPGAgent, MADDPGAgent_Kaige, MATD3Agent, MASACAgent, MAD3PGAgent
from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.multiagent.environment import MultiAgentEnv
from tf2marl.common.util import softmax_to_argmax


from env_3CAV_cooperation_20230319_benchmarks import Network

if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0],
                                             True)

train_ex = Experiment('train')


# This file uses Sacred for logging purposes as well as for config management.
# I recommend logging to a Mongo Database which can nicely be visualized with
# Omniboard.
@train_ex.config
def train_config():
    # Logging
    exp_name = 'default'            # name for logging

    display = False                 # render environment
    restore_fp = None               # path to restore models from, e.g.
    #                                 # 'results/sacred/182/models', or None to train a new model
    # restore_fp = 'results/sacred/29/models'             # path to restore models from, e.g.
                                    # 'results/sacred/182/models', or None to train a new model
    save_rate = 10                  # frequency to save policy as number of episodes

    # Environment
    scenario_name = 'simple_spread' # environment name
    num_episodes = 15000           # total episodes
    max_episode_len = 75            # timesteps per episodes

    # Agent Parameters
    # good_policy = 'matd3'          # policy of "good" agents in env
    # adv_policy = 'matd3'           # policy of adversary agents in env
    good_policy = 'maddpg'          # policy of "good" agents in env
    adv_policy = 'maddpg'           # policy of adversary agents in env
    # available agent: maddpg, matd3, mad3pg, masac

    # General Training Hyperparameters
    lr = 1e-2                       # learning rate for critics and policies
    lr_c = 1e-2                       # learning rate for critic
    lr_a = 1e-3                       # learning rate for policies
    gamma = 0.95                    # decay used in environments
    batch_size = 1024               # batch size for training
    num_layers = 2                  # hidden layers per network
    num_units = 64                  # units per hidden layer

    update_rate = 100               # update policy(?) after each x steps
    critic_zero_if_done = False     # set the value to zero in terminal steps
    # buff_size = 1e6                 # size of the replay buffer (1e6 seems too large?)
    buff_size = 1e5                 # size of the replay buffer (1e6 seems too large?)
    tau = 0.01                      # Update for target networks
    hard_max = False                # use Straight-Through (ST) Gumbel

    priori_replay = False           # enable prioritized replay
    alpha = 0.6                     # alpha value (weights prioritization vs random)
    beta = 0.5                      # beta value  (controls importance sampling)

    use_target_action = True        # use target action in environment, instead of normal action

    # MATD3
    if good_policy == 'tf2marl':
        policy_update_rate = 2      # Frequency of policy updates, compared to critic
    else:
        policy_update_rate = 1
    critic_action_noise_stddev = 0.0  # Added noise in critic updates
    action_noise_clip = 0.5         # limit for this noise

    # MASAC
    entropy_coeff = 0.05            # weight for entropy in Soft-Actor-Critic

    # MAD3PG
    num_atoms = 51                  # number of atoms in support
    min_val = -400                  # minimum atom value
    max_val = 0                     # largest atom value


# @train_ex.capture
# def make_env(scenario_name) -> MultiAgentEnv:
#     """
#     Create an environment
#     :param scenario_name:
#     :return:
#     """
#     import tf2marl.multiagent.scenarios as scenarios

#     # load scenario from script
#     scenario = scenarios.load(scenario_name + '.py').Scenario()
#     # create world
#     world = scenario.make_world()
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
#     return env


@train_ex.main
def train(_run, exp_name, save_rate, display, restore_fp,
          hard_max, max_episode_len, num_episodes, batch_size, update_rate,
          use_target_action):
    """
    This is the main training function, which includes the setup and training loop.
    It is meant to be called automatically by sacred, but can be used without it as well.

    :param _run:            Sacred _run object for legging
    :param exp_name:        (str) Name of the experiment
    :param save_rate:       (int) Frequency to save networks at
    :param display:         (bool) Render the environment
    :param restore_fp:      (str)  File-Patch to policy to restore_fp or None if not wanted.
    :param hard_max:        (bool) Only output one action
    :param max_episode_len: (int) number of transitions per episode
    :param num_episodes:    (int) number of episodes
    :param batch_size:      (int) batch size for updates
    :param update_rate:     (int) perform critic update every n environment steps
    :param use_target_action:   (bool) use action from target network
    :return:    List of episodic rewards
    """
    # # Create environment
    # print(_run.config)
    # env = make_env()
    env = Network()

    # Create agents
    agents = get_agents(_run, env, env.n_adversaries)
    print("env.n_adversaries:", env.n_adversaries) # give result = 0

    logger = RLLogger(exp_name, _run, len(agents), env.n_adversaries, save_rate)

    # Load previous results, if necessary
    if restore_fp is not None:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(restore_fp, 'agent_{}'.format(ag_idx))
            agent.load(fp)

    obs_n = env.reset()
    # print("obs_n_reset:", obs_n)

    q_loss_all = [[] for _ in range(env.n)]
    pol_loss_all = [[] for _ in range(env.n)]
    train_steps_plot = []

    episode_rewards_BF_optimal = [0]
    episode_objs_BF_optimal = [0]
    episode_switches_BF_optimal = [0]

    episode_rewards_random = [0]
    episode_objs_random = [0]
    episode_switches_random = [0]

    episode_rewards_modified = [0]



    print('Starting iterations...')
    while True:
        # get action
        if use_target_action:
            action_n = [agent.target_action(obs.astype(np.float32)[None])[0] for agent, obs in
                        zip(agents, obs_n)]
        else:
            action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]
        # print("action_n:", action_n)
        # environment step
        # if hard_max: # default: False
        #     hard_action_n = softmax_to_argmax(action_n, agents)
        #     new_obs_n, rew_n, done_n, info_n = env.step(hard_action_n)
        # else:
        #     action_n = [action.numpy() for action in action_n]
        #     new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        action_n = [action.numpy() for action in action_n]
        # new_obs_n, rew_n, done_n, obj_n, switch_n = env.step(action_n)
        new_obs_n, rew_n, done_n, obj_n, switch_n, reward_modified, reward_BF_optimal, obj_BF_optimal, switch_cost_BF_optimal, reward_random, obj_random, switch_cost_random = env.step(action_n)
        # print("action_n:", action_n)
        # print("new_obs_n:", new_obs_n)
        # print("rew_n:", rew_n)
        # print("done_n:", done_n)
        logger.episode_step += 1

        done = all(done_n)
        terminal = (logger.episode_step >= max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
        obs_n = new_obs_n

        episode_rewards_BF_optimal[-1] += reward_BF_optimal
        episode_objs_BF_optimal[-1] += obj_BF_optimal
        episode_switches_BF_optimal[-1] += switch_cost_BF_optimal

        # print("episode_rewards_BF_optimal:",episode_rewards_BF_optimal)

        episode_rewards_random[-1] += reward_random
        # print("episode_rewards_random:",episode_rewards_random)
        episode_objs_random[-1] += obj_random
        episode_switches_random[-1] += switch_cost_random

        episode_rewards_modified[-1] += reward_modified

        

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += rew
            logger.agent_rewards[ag_idx][-1] += rew

        for ag_idx, obj in enumerate(obj_n):
            logger.cur_episode_obj += obj
            # logger.agent_objs[ag_idx][-1] += obj

        for ag_idx, switch in enumerate(switch_n):
            logger.cur_episode_switch += switch
            # logger.agent_rewards[ag_idx][-1] += switch

        if done:
            obs_n = env.reset()
            episode_step = 0
            logger.record_episode_end(agents)
            episode_rewards_BF_optimal.append(0.0)
            episode_objs_BF_optimal.append(0.0)
            episode_switches_BF_optimal.append(0.0)

            episode_rewards_random.append(0.0)
            episode_objs_random.append(0.0)
            episode_switches_random.append(0.0)
            # print("episode:", len(logger.episode_rewards))

            episode_rewards_modified.append(0.0)

        logger.train_step += 1

        # policy updates
        train_cond = not display # at the beginning, display is False, train_cond is True
        # print("train_cond:", train_cond)
        for i, agent in enumerate(agents):
            # if train_cond and len(agent.replay_buffer) > batch_size * max_episode_len:
            if train_cond and len(agent.replay_buffer) > 1e4:
                if logger.train_step % update_rate == 0:  # only update every 100 steps
                    q_loss, pol_loss = agent.update(agents, logger.train_step)
                    q_loss_all[i].append(q_loss)
                    pol_loss_all[i].append(pol_loss)   
                    if i == 0:
                        train_steps_plot.append(logger.train_step)   


        # # for displaying learned policies
        # if display:
        #     time.sleep(0.1)
        #     env.render()

        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > num_episodes:
            logger.experiment_end(q_loss_all,pol_loss_all,train_steps_plot,episode_rewards_BF_optimal,episode_objs_BF_optimal,episode_switches_BF_optimal,episode_rewards_random,episode_objs_random,episode_switches_random,episode_rewards_modified)
            return logger.get_sacred_results()


@train_ex.capture
def get_agents(_run, env, num_adversaries, good_policy, adv_policy, lr, lr_c, lr_a, batch_size,
               buff_size, num_units, num_layers, gamma, tau, priori_replay, alpha, num_episodes,
               max_episode_len, beta, policy_update_rate, critic_action_noise_stddev,
               entropy_coeff, num_atoms, min_val, max_val) -> List[AbstractAgent]:
    """
    This function generates the agents for the environment. The parameters are meant to be filled
    by sacred, and are therefore documented in the configuration function train_config.

    :returns List[AbstractAgent] returns a list of instantiated agents
    """
    agents = []
    for agent_idx in range(num_adversaries):
        if adv_policy == 'maddpg':
            agent = MADDPGAgent_Kaige(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr_c, lr_a, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                _run=_run)
        elif adv_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev, _run=_run
                               )
        elif adv_policy == 'mad3pg':
            agent = MAD3PGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                num_atoms=num_atoms, min_val=min_val, max_val=max_val,
                                _run=_run
                                )
        elif good_policy == 'masac':
            agent = MASACAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               entropy_coeff=entropy_coeff, policy_update_freq=policy_update_rate,
                               _run=_run
                               )
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    for agent_idx in range(num_adversaries, env.n):
        if good_policy == 'maddpg':
            agent = MADDPGAgent_Kaige(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr_c, lr_a, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                _run=_run)
        elif good_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev, _run=_run
                               )
        elif adv_policy == 'mad3pg':
            agent = MAD3PGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                num_atoms=num_atoms, min_val=min_val, max_val=max_val,
                                _run=_run
                                )
        elif good_policy == 'masac':
            agent = MASACAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               entropy_coeff=entropy_coeff, policy_update_freq=policy_update_rate,
                               _run=_run
                               )
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    print('Using good policy {} and adv policy {}'.format(good_policy, adv_policy))
    return agents


def main():
    file_observer = FileStorageObserver.create(os.path.join('results', 'sacred'))
    # use this code to attach a mongo database for logging
    # mongo_observer = MongoObserver(url='localhost:27017', db_name='sacred')
    # train_ex.observers.append(mongo_observer)
    train_ex.observers.append(file_observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
