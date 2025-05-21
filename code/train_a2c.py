from typing import Union

import time
import argparse
import json

from pathlib import Path
from itertools import chain

import numpy as np
import torch

from agent import ActorProb, Critic
from a2c import A2C
from environment import NormalizedEnvironment
from util import test_episode, log_results


def init_weights(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        torch.nn.init.zeros_(module.bias)

def rescale_weights(module: torch.nn.Module, scale: float) -> None:
    if isinstance(module, torch.nn.Linear):
        module.weight.data.copy_(scale * module.weight.data)

def save_run(
    run: int, ep: Union[int, None], policy: A2C, train_env: NormalizedEnvironment
) -> None:
    ep_str = f"_ep_{ep}" if ep is not None else ""

    torch.save(policy.actor, f"{log_dir}/run_{run}{ep_str}_actor.model")
    torch.save(policy.critic, f"{log_dir}/run_{run}{ep_str}_critic.model")
    torch.save(policy.actor_optim, f"{log_dir}/run_{run}{ep_str}.optim")

    with open(f"{log_dir}/run_{run}{ep_str}.norm", "w") as out:
        for attr in (train_env.mean, train_env.var, train_env.max_norm, train_env.count, train_env.eps):
            print(attr, file=out)

def next_epoch(
    ep: int, run: int, policy: A2C,
    train_env: NormalizedEnvironment, test_env: NormalizedEnvironment
) -> None:
    if ep % 10 == 0:
        save_run(run, ep, policy, train_env)

    test_env.copy_normalization_params(train_env)


# Basic configuration
optimizers = {
    "rms": torch.optim.RMSprop,
    "adam": torch.optim.Adam
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-runs", help="number of times to run the experiment", type=int, choices=range(1, 11), required=True)
parser.add_argument("--actor-hidden", help="sizes of hidden layers in the actor net", nargs='*', type=int, required=True)
parser.add_argument("--critic-hidden", help="sizes of hidden layers in the critic net", nargs='*', type=int, required=True)
parser.add_argument("--optim", help="optimizer of both the actor's and the critic's parameters", choices=optimizers, required=True)
parser.add_argument("--lr", help="learning rate for both the actor and the critic", type=float, required=True)

log_dir = f"results/rl/a2c_{time.strftime('%d%m%Y_%H%M%S', time.gmtime(time.time()))}"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float64)

args = {
    "train_env_kwargs": {
        "update": True,
        "horizon": 1000,
        "render_mode": None
    },
    "test_env_kwargs": {
        "update": False,
        "horizon": 1000,
        "render_mode": None
    },
    "test_train_seed": 42,
    "test_test_seed": 47,
    "actor": {
        "last_scale": 0.001,
        "sigma": -2
    },
    "critic": {
        "last_scale": 0.001
    },
    "policy" : {
        "gamma": 0.99
    },
    "training": {
        "epochs": 300,
        "step_per_epoch": 1000,
        "episode_per_test": 10,
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

    for run in range(args["num_runs"]):
        # Prepare environments
        train_env = NormalizedEnvironment(**args["train_env_kwargs"])
        test_env = NormalizedEnvironment(**args["test_env_kwargs"])

        # Prepare agent
        actor = ActorProb(
            train_env.observation_dim, train_env.action_dim,
            hidden_sizes=args["actor_hidden"],
            hidden_activations=[torch.nn.Tanh] * len(args["actor_hidden"])
        ).to(device)

        critic = Critic(
            train_env.observation_dim,
            hidden_sizes=args["critic_hidden"],
            hidden_activations=[torch.nn.Tanh] * len(args["critic_hidden"])
        ).to(device)
        
        actor.apply(init_weights)
        actor.last.apply(lambda m: rescale_weights(m, args["actor"]["last_scale"]))
        torch.nn.init.constant_(actor.sigma, args["actor"]["sigma"])
        
        critic.apply(init_weights)
        critic.last.apply(lambda m: rescale_weights(m, args["critic"]["last_scale"]))
        
        optim = optimizers[args["optim"]](
            params=chain(actor.parameters(), critic.parameters()),
            lr=args["lr"]
        )

        policy = A2C(
            actor=actor,
            critic=critic,
            gamma=args["policy"]["gamma"],
            actor_optim=optim,
            critic_optim=optim
        )

        # Prepare training
        log = f"{log_dir}/run_{run}.log"
        test_env.reset(seed=args["test_train_seed"])
        obs, _ = train_env.reset()
        done = False

        # Train
        for ep in range(args["training"]["epochs"]):
            print(f"Epoch {ep}")
            
            next_epoch(ep, run, policy, train_env, test_env)

            policy.eval()
    
            res = np.array([test_episode(policy, test_env)
                            for _ in range(args["training"]["episode_per_test"])])
            log_results(log, res)

            policy.train()

            for _ in range(args["training"]["step_per_epoch"]):
                if done:
                    obs, _ = train_env.reset()
                    done = False
                
                act = policy.forward(obs).detach().numpy()

                obs, rew, done, _ = train_env.step(act)
                policy.learn(rew, obs)
        
        next_epoch(ep, run, policy, train_env, test_env)

        policy.eval()
    
        res = np.array([test_episode(policy, test_env)
                        for _ in range(args["training"]["episode_per_test"])])
        log_results(log, res)

        policy.train()

        # Save models
        save_run(run, args["training"]["epochs"], policy, train_env)

        # Test
        test_env.reset(seed=args["test_test_seed"])
        policy.eval()
        
        res = np.array([test_episode(policy, test_env)
                        for _ in range(args["test_episodes"])])
        log_results(test_log, res)
