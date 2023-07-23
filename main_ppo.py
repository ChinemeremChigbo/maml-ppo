#!/usr/bin/env python3
"""This is an example to train MAML-VPG on HalfCheetahDirEnv environment."""
# pylint: disable=no-value-for-parameter
import click
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.mujoco import HalfCheetahDirEnv
from garage.experiment import MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
from cav_environment import CAVVelEnv


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=15)
@click.option('--episodes_per_task', default=4)
@click.option('--meta_batch_size', default=3)
@wrap_experiment(snapshot_mode='all')
def main(ctxt, seed, epochs, episodes_per_task,
                              meta_batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)
    max_episode_length = 75
    env = normalize(CAVVelEnv(max_episode_length=max_episode_length),
                    expected_action_scale=10.)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16, 16),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(8, 8),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    def set_length(env, _task):
        env.close()
        return normalize(CAVVelEnv(max_episode_length=max_episode_length))
    
    task_sampler = SetTaskSampler(CAVVelEnv, wrapper=set_length)

    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   n_test_tasks=2,
                                   n_test_episodes=2)

    trainer = Trainer(ctxt)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=max_episode_length)

    algo = MAMLPPO(env=env,
                   policy=policy,
                   sampler=sampler,
                   task_sampler=task_sampler,
                   value_function=value_function,
                   meta_batch_size=meta_batch_size,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * max_episode_length)


main()