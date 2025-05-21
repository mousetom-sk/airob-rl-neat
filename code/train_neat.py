from typing import Union, List

import time
import argparse
import json
import pickle
import multiprocessing

from pathlib import Path

import numpy as np
import neat

from agent import Actor
from environment import Environment
from util import test_episode, log_results, draw_genome


def save_run(run: int, ep: Union[int, None], genome: neat.DefaultGenome) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    with open(f"{log_dir}/run_{run}{ep_str}.genome", "wb") as out:
        pickle.dump(genome, out)

def next_epoch(ep: int, run: int, genome: neat.DefaultGenome) -> None:
    if ep % 10 == 0:
        save_run(run, ep, genome)

def evaluate(env: Environment, genome: neat.DefaultGenome, config: neat.Config) -> float:
    actor = Actor(genome, config)
    rets = np.array([test_episode(actor, env)[0]
                     for _ in range(args["training"]["episode_per_epoch"])])
    
    return rets.mean()

def evaluate_genomes(genomes: List[neat.DefaultGenome], config: neat.Config) -> None:
    with multiprocessing.Pool(num_cores) as pool:
        jobs = []
        for i, (_, g) in enumerate(genomes):
            jobs.append(pool.apply_async(evaluate, (train_env_pool[i], g, config)))
        
        for job, (_, g) in zip(jobs, genomes):
            g.fitness = job.get()

    snapshot = train_env_pool[0].create_snapshot()
    for j in range(i, len(train_env_pool)):
        train_env_pool[j].restore_snapshot(snapshot)


# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)

log_dir = f"results/ea/neat_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
config_path = "code/neat_config.txt"

args = {
    "train_env_kwargs": {
        "horizon": 1000,
        "render_mode": None
    },
    "test_env_kwargs": {
        "horizon": 1000,
        "render_mode": None
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "training": {
        "epochs": 300,
        "episode_per_epoch": 10,
        "episode_per_test": 10
    },
    "test_episodes": 1000
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())
    num_cores = multiprocessing.cpu_count()

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    test_log = f"{log_dir}/test.log"

    for run in range(args["num_runs"]):
        # Prepare training
        log = f"{log_dir}/run_{run}.log"

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        
        population = neat.Population(config)
        population.add_reporter(neat.StdOutReporter(True))

        # Prepare environments
        seed = int(np.random.rand() * (2 ** 32 - 1))
        train_env_pool = [Environment(**args["train_env_kwargs"])
                          for _ in range(config.pop_size + 5)]
        for env in train_env_pool:
            env.reset(seed=seed)
        
        test_env = Environment(**args["test_env_kwargs"])
        test_env.reset(seed=args["test_train_seed"])

        # Train
        for ep in range(args["training"]["epochs"]):
            genomes = list(population.population.items())
            
            # Evaluate population
            population.run(evaluate_genomes, 1)
            
            # Test best actor
            best_genome = sorted(genomes, key=lambda g: g[1].fitness, reverse=True)[0][1]
            best_actor = Actor(best_genome, config)
            
            next_epoch(ep, run, best_genome)
    
            res = np.array([test_episode(best_actor, test_env)
                            for _ in range(args["training"]["episode_per_test"])])
            log_results(log, res)

        # Evaluate population
        genomes = list(population.population.items())
        evaluate_genomes(genomes, config)
        
        # Test best actor
        best_genome = sorted(genomes, key=lambda g: g[1].fitness, reverse=True)[0][1]
        best_actor = Actor(best_genome, config)

        next_epoch(ep, run, best_actor)
    
        res = np.array([test_episode(best_actor, test_env)
                        for _ in range(args["training"]["episode_per_test"])])
        log_results(log, res)

        # Save model
        save_run(run, None, best_genome)
        draw_genome(best_genome, config, filename=f"{log_dir}/run_{run}_architecture")

        # Test
        test_env.reset(seed=args["test_test_seed"])
        
        res = np.array([test_episode(best_actor, test_env)
                        for _ in range(args["test_episodes"])])
        log_results(test_log, res)
