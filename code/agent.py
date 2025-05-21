from typing import List, Tuple, Type, Union

import torch
import neat
from numpy.typing import NDArray


class Actor:

    model: neat.nn.FeedForwardNetwork

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config):
        self.model = neat.nn.FeedForwardNetwork.create(genome, config)

    def forward(self, observation: Union[torch.Tensor, NDArray]) -> torch.Tensor:
        action = torch.tensor(self.model.activate(observation))

        return action


class ActorProb(torch.nn.Module):

    mu: torch.nn.Module
    sigma: torch.nn.Parameter

    def __init__(
        self,
        observation_dim: int, action_dim: int,
        hidden_sizes: List[int], hidden_activations: List[Type[torch.nn.Module]]
    ) -> None:
        super().__init__()

        layers = []
        sizes = [observation_dim] + hidden_sizes
        o = observation_dim

        for i, o, a in zip(sizes, sizes[1:], hidden_activations):
            layers.append(torch.nn.Linear(i, o))
            layers.append(a())
        
        self.last = torch.nn.Linear(o, action_dim)
        layers.append(self.last)

        self.mu = torch.nn.Sequential(*layers)
        self.sigma = torch.nn.Parameter(torch.empty(action_dim))

    def forward(self, observation: Union[torch.Tensor, NDArray]) -> Tuple[torch.Tensor]:
        mu = self.mu.forward(torch.tensor(observation))
        sigma = torch.exp(self.sigma)

        return mu, sigma


class Critic(torch.nn.Module):

    model: torch.nn.Module

    def __init__(
        self,
        observation_dim: int,
        hidden_sizes: List[int], hidden_activations: List[Type[torch.nn.Module]]
    ) -> None:
        super().__init__()

        layers = []
        sizes = [observation_dim] + hidden_sizes
        o = observation_dim

        for i, o, a in zip(sizes, sizes[1:], hidden_activations):
            layers.append(torch.nn.Linear(i, o))
            layers.append(a())
        
        self.last = torch.nn.Linear(o, 1)
        layers.append(self.last)

        self.model = torch.nn.Sequential(*layers)

    def forward(self, observation: Union[torch.Tensor, NDArray]) -> torch.Tensor:
        value = self.model.forward(torch.tensor(observation))

        return value
