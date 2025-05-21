# Comparing Reinforcement Learning with Neuroevolution

This repository provides the scripts and results from our experiments with
Advantage Actor-Critic (A2C) and NeuroEvolution of Augmenting Topologies (NEAT)
for our project for the course Artificial-Inteligence Algorithms in Robotics.

The contents of this repository are organized into the following directiories:

- `code`, which holds the core implementation (files `agent.py`,
`environment.py`, `a2c.py`, and `util.py`), the training and testing scripts and
configurations (files `train_a2c.py`, `train_neat.py`, `test_a2c.py`,
`test_neat.py`, and `neat_config.py`), the visualization scripts (files
`plot_training_results.py` and `table_test_results.py`), and the script for user
inetraction with the environment (file `interactive.py`);

- `results`, which contains the logs from our experiments with A2C (subdirectory
`rl`) and NEAT (subdirectory `ea`);

- `figures`, where the learning progressions from our experiments with A2C
(subdirectory `rl`) and NEAT (subdirectory `ea`) can be found;

- `recordings`, where we share the recordings from a few test episodes for A2C
(subdirectory `rl`) and NEAT (subdirectory `ea`).


# Installation 

These instructions provide a guide to create a conda virtual environment
(airob-rl-neat). Setting up this environment is required to use our scripts.

The environment airob-rl-neat can be created as follows:

	1. Install conda if this has not yet been done (for example, see
	   https://www.anaconda.com/docs/getting-started/miniconda/install).
	
	2. Create the environment airob-rl-neat from YML:
			conda env create -f environment.yml
