import os

import click
import torch
from dowel import CsvOutput, logger
from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from cav_environment import CAVVelEnv


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=2000)
@click.option('--episodes_per_task', default=10)
@click.option('--max_episode_length', default=75)
@click.option('--saved_dir', default=os.getcwd()+"/logs")
@click.option('--log_file', default=os.getcwd()+"/logs/ppo.csv")
@wrap_experiment
def main(ctxt, seed, epochs, episodes_per_task, max_episode_length, saved_dir, log_file):
    """Train PPO with CAV environment.

    Set up environment and algorithm and run the task.

    Args:
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        max_episode_length (int): The maximum steps allowed for an
            episode.
        saved_dir (str): Path where snapshots are saved.
        log_file (str): Path where csvs are saved.
    """

    set_seed(seed)
    logger.add_output(CsvOutput(log_file))
    env = normalize(CAVVelEnv(max_episode_length=max_episode_length),
                    expected_action_scale=10.)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(16, 16),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         is_tf_worker=True)

    value_function = GaussianMLPValueFunction(env_spec=env.spec)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.97,
               lr_clip_range=2e-1)

    ctxt.snapshot_dir = saved_dir
    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * max_episode_length)


main()
