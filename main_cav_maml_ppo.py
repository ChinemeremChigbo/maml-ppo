import os

import click
from dowel import CsvOutput, logger
from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import Snapshotter
from garage.trainer import Trainer

from cav_environment import CAVVelEnv


@click.command()
@click.option('--epochs', default=2000)
@click.option('--max_episode_length', default=75)
@click.option('--saved_dir', default=os.getcwd()+"/logs")
@click.option('--log_file', default=os.getcwd()+"/logs/maml_ppo.csv")
@wrap_experiment
def main(ctxt, epochs, max_episode_length, saved_dir, log_file):
    """CAV environment with pretrained MAML.

    Set up environment and algorithm and run the task.

    Args:
        epochs (int): Number of training epochs.
        max_episode_length (int): The maximum steps allowed for an
            episode.
        saved_dir (str): Path where snapshots are saved.
        log_file (str): Path where csvs are saved.
    """
    csv_output = CsvOutput(log_file)
    logger.add_output(csv_output)
    snapshotter = Snapshotter()
    data = snapshotter.load(saved_dir)
    algo = data['algo']
    env = normalize(CAVVelEnv(max_episode_length=max_episode_length),
                    expected_action_scale=10.)
    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=1)


main()
