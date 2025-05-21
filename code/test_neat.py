import time
import argparse
import json
import pickle

from pathlib import Path

import numpy as np
import neat

from agent import Actor
from environment import Environment
from util import test_episode, log_results, draw_genome


# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory with the model to test", type=str, required=True)
parser.add_argument("--run", help="run to test", type=int, required=True)
parser.add_argument("--ep", help="episode snapshot to test", type=int)

log_dir = f"results/ea/neat_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
config_path = "code/neat_config.txt"

args = {
    "test_env_kwargs": {
        "horizon": 1000,
        "render_mode": "human"
    },
    "test_test_seed": 47,
    "test_episodes": 1000
}


if __name__ == "__main__":
    args |= vars(parser.parse_args())

    # Prepare log directory
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{log_dir}/args.txt", "w") as out:
        json.dump(args, out, indent=4)

    test_log = f"{log_dir}/test.log"

    # Prepare environment
    test_env = Environment(**args["test_env_kwargs"])

    # Prepare agent
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    ep_str = f"_ep_{args['ep']}" if args["ep"] is not None else ""
    with open(f"{args['dir']}/run_{args['run']}{ep_str}.genome", "rb") as src:
        genome = pickle.load(src)
    
    draw_genome(genome, config, filename=f"{log_dir}/architecture")

    # actor = Actor(genome, config)

    # # Test
    # test_env.reset(seed=args["test_test_seed"])
    
    # rets = np.array([test_episode(actor, test_env)
    #                     for _ in range(args["test_episodes"])])
    # log_results(test_log, rets)
