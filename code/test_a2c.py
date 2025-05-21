import time
import argparse
import json

from pathlib import Path

import numpy as np
import torch

from a2c import A2C
from environment import NormalizedEnvironment
from util import test_episode, log_results


def load_normalization(dir: str, run: int, ep: int, env: NormalizedEnvironment) -> None:
    keys = ["mean", "var", "max_norm", "count", "eps"]
    attrs = []

    with open(f"{dir}/run_{run}_ep_{ep}.norm", "r") as src:
        for line in src:
            line = line[:-1]
            if line.startswith("["):
                attrs.append([])
                is_array = True
                attrs[-1].extend([float(x) for x in line[1:].split()])
            elif line.endswith("]"):
                is_array = False
                attrs[-1].extend([float(x) for x in line[:-1].split()])
                attrs[-1] = np.array(attrs[-1])
            elif is_array:
                attrs[-1].extend([float(x) for x in line.split()])
            else:
                attrs.append(float(line))
    
    for k, v in zip(keys, attrs):
        env.__setattr__(k, v)


# Basic configuration
parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="directory with the model to test", type=str, required=True)
parser.add_argument("--run", help="run to test", type=int, required=True)
parser.add_argument("--ep", help="episode snapshot to test", type=int, required=True)

log_dir = f"results/rl/a2c_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

args = {
    "test_env_kwargs": {
        "update": False,
        "horizon": 1000,
        "render_mode": "human"
    },
    "test_test_seed": 47,
    "policy" : {
        "gamma": 0.99
    },
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

    # Prepare environments
    test_env = NormalizedEnvironment(**args["test_env_kwargs"])
    load_normalization(args['dir'], args['run'], args['ep'], test_env)

    # Prepare agent
    actor = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}_actor.model", device, weights_only=False)
    critic = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}_critic.model", device, weights_only=False)
    optim = torch.load(f"{args['dir']}/run_{args['run']}_ep_{args['ep']}.optim", device, weights_only=False)
    
    to_device = lambda m: setattr(m, "device", device) if hasattr(m, "device") else None
    actor.apply(to_device)
    critic.apply(to_device)

    policy = A2C(
        actor=actor,
        critic=critic,
        gamma=args["policy"]["gamma"],
        actor_optim=optim,
        critic_optim=optim
    )

    # Test
    test_env.reset(seed=args["test_test_seed"])
    policy.eval()
    
    scaled_rets = np.array([test_episode(policy, test_env)
                            for _ in range(args["test_episodes"])])
    log_results(test_log, scaled_rets)
