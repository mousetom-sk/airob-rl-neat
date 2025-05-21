from typing import Union

import torch
from numpy.typing import NDArray

from agent import ActorProb, Critic


class A2C(torch.nn.Module):

    actor: ActorProb
    critic: Critic
    gamma: float
    actor_optim: torch.optim.Optimizer
    critic_optim: torch.optim.Optimizer

    _last_observation: torch.Tensor
    _last_log_prob: torch.Tensor
    
    def __init__(
        self,
        actor: ActorProb, critic: Critic, gamma: float,
        actor_optim: torch.optim.Optimizer, critic_optim: torch.optim.Optimizer
    ) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
    
    def forward(self, observation: Union[torch.Tensor, NDArray]) -> torch.Tensor:
        mu, sigma = self.actor.forward(observation)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        
        self._last_observation = torch.tensor(observation).to(action)
        self._last_log_prob = dist.log_prob(action)

        action = torch.nn.functional.tanh(action)

        return action
    
    def learn(self, reward: Union[torch.Tensor, NDArray], next_observation: Union[torch.Tensor, NDArray]) -> None:
        reward = torch.tensor(reward).to(self._last_observation)
        next_observation = torch.tensor(next_observation).to(self._last_observation)

        value = self.critic.forward(self._last_observation)
        next_value = self.critic.forward(next_observation)
        advantage = (reward + self.gamma * next_value - value).detach()
        
        actor_loss = - advantage * self._last_log_prob.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        critic_loss = - advantage * value
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
