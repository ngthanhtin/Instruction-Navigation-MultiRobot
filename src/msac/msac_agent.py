"""MASAC"""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from msac.buffer import ReplayBuffer
from msac.mactor_critic import Actor, CriticQ, CriticV
from torch.nn.utils.clip_grad import clip_grad_norm_

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class MSAC:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        memory_size: int, 
        batch_size: int,
        gamma: float=0.99, 
        tau: float=5e-3,
        initial_random_steps: int= int(1e2), 
        policy_update_fequency: int=2,
        num_agents: int=2):

        self.action_size = action_dim
        self.state_size = state_dim
        self.num_agents = num_agents
        
        self.memory = ReplayBuffer(self.state_size, self.action_size, memory_size, batch_size)
        self.batch_size = batch_size
        self.memor_size = memory_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = initial_random_steps
        self.policy_update_frequecy = policy_update_fequency

        self.device = device
        
        self.target_alpha = -np.prod((self.action_size,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.actor = Actor(self.state_size, self.action_size).to(self.device)

        self.vf = CriticV(self.state_size).to(self.device)
        self.vf_target = CriticV(self.state_size).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())

        self.qf1 = CriticQ(self.state_size + self.action_size).to(self.device)
        self.qf2 = CriticQ(self.state_size + self.action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.vf_optimizer = optim.Adam(self.vf.parameters(), lr=3e-4)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=3e-4)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=3e-4)

        self.transition = [[] for i in range(self.num_agents)]

        self.total_step = 0

        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(-1, 1, (self.num_agents, self.action_size))
        else:
            selected_action = []
            for i in range(self.num_agents):
                action = self.actor(
                    torch.FloatTensor(state[i]).to(self.device)
                )[0].detach().cpu().numpy()
                selected_action.append(action)
            selected_action = np.array(selected_action)
            selected_action = np.clip(selected_action, -1, 1)
        
        for i in range(self.num_agents):
            self.transition[i] = [state[i], selected_action[i]]

        return selected_action
    
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        device = self.device

        samples = self.memory.sample_batch()
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.FloatTensor(samples["acts"].reshape(-1, self.action_size)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        new_action, log_prob = self.actor(state)


        alpha_loss = (
            -self.log_alpha.exp() * (log_prob + self.target_alpha).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        alpha = self.log_alpha.exp()

        mask = 1 - done
        q1_pred = self.qf1(state, action)
        q2_pred = self.qf2(state, action)
        vf_target = self.vf_target(next_state)
        q_target = reward + self.gamma * vf_target * mask
        qf1_loss = F.mse_loss(q_target.detach(), q1_pred)
        qf2_loss = F.mse_loss(q_target.detach(), q2_pred)

        v_pred = self.vf(state)
        q_pred = torch.min(
            self.qf1(state, new_action), self.qf2(state, new_action)
        )
        v_target = q_pred - alpha * log_prob
        v_loss = F.mse_loss(v_pred, v_target.detach())

        if self.total_step % self.policy_update_frequecy == 0:
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        qf_loss = qf1_loss + qf2_loss

        self.vf_optimizer.zero_grad()
        v_loss.backward()
        self.vf_optimizer.step()

        return actor_loss.data, qf_loss.data, v_loss.data, alpha_loss.data
    
    def _target_soft_update(self):
        tau = self.tau

        for t_param, l_param in zip(
            self.vf_target.parameters(), self.vf.parameters()
        ):
            t_param.data.copy_( tau * l_param.data + (1.0 - tau) * t_param.data)

    
        
