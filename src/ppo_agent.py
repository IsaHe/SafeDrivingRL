import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        actor_features = self.actor(state)
        action_mean = self.actor_mean(actor_features)

        action_std = torch.exp(self.actor_log_std)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        action_log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)

        dist_entropy = dist.entropy().sum(axis=-1, keepdim=True)

        state_value = self.critic(state)

        return action, action_log_prob, dist_entropy, state_value


class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=10,
        hidden_dim=256,
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_log_prob, _, value = self.policy.get_action_and_value(
                state_tensor
            )

        return (
            action.cpu().numpy().flatten(),
            action_log_prob.cpu().item(),
            value.cpu().item(),
        )

    def update(self, memory):
        old_states = torch.FloatTensor(np.array(memory["states"])).to(self.device)
        old_actions = torch.FloatTensor(np.array(memory["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory["log_probs"])).to(self.device)
        rewards = memory["rewards"]
        is_terminals = memory["dones"]

        rewards_to_go = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards_to_go.insert(0, discounted_reward)

        rewards_to_go = torch.FloatTensor(rewards_to_go).to(self.device)
        rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (
            rewards_to_go.std() + 1e-7
        )
        rewards_to_go = rewards_to_go.unsqueeze(1)

        for _ in range(self.k_epochs):
            _, logprobs, dist_entropy, state_values = self.policy.get_action_and_value(
                old_states, old_actions
            )

            ratios = torch.exp(logprobs - old_log_probs.unsqueeze(1))

            advantages = rewards_to_go - state_values.detach()

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.mse_loss(state_values, rewards_to_go)
                - 0.01 * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
