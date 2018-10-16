import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgentBase:
    def __init__(self, net, target_net, action_dim=None, device=None, update_every=4, minibatch_size=64,
                 tau=1e-3, gamma=0.99, lr=5e-4, **kwargs):
        self.net = net
        self.target_net = target_net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.memory = None
        self.__action_dim = action_dim
        self.device = device
        self.__step_i = 0
        self.__update_every = update_every
        self.__minibatch_size = minibatch_size
        self.__tau = tau
        self.gamma = gamma
        pass

    def act(self, state, eps):
        with torch.no_grad():
            action_values = self.net(torch.from_numpy(state).float().to(self.device))
        if random.random() < eps:
            action = random.randint(0, self.__action_dim - 1)
        else:
            action = action_values.max(1)[1].item()
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        loss = 0
        self.__step_i += 1
        if self.__step_i % self.__update_every == 0 and self.memory.size() > self.__minibatch_size:
            # sample and train
            samples = self.memory.sample()
            loss = self._learn(samples)
            self.soft_update()
        return loss

    def soft_update(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(self.__tau*local_param.data + (1.0-self.__tau)*target_param.data)

    def save(self, fname):
        ckpt = {'net': self.net,
                'target_net': self.target_net,
                'action_dim': self.__action_dim,
                'device': self.device
                }
        torch.save(ckpt, fname)
        pass

    @classmethod
    def load(cls, fname):
        ckpt = torch.load(fname)
        obj = cls(ckpt['net'], ckpt['target_net'], action_dim=ckpt['action_dim'], device=ckpt['device'])
        return obj

    def _learn(self, samples):
        pass


class DQNAgent(DQNAgentBase):
    def __init__(self, net, target_net, **kwargs):
        super(DQNAgent, self).__init__(net, target_net, **kwargs)
        self.memory = ReplayBuffer(**kwargs)

    def _learn(self, samples):
        states, actions, rewards, next_states, dones = samples
        expected_q_values = self.net(states).gather(1, actions)
        # DQN target
        target_q_values_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * target_q_values_next * (1 - dones))
        loss = F.mse_loss(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


class DDQNAgent(DQNAgentBase):
    def __init__(self, net, target_net, **kwargs):
        super(DDQNAgent, self).__init__(net, target_net, **kwargs)
        self.memory = ReplayBuffer(**kwargs)

    def _learn(self, samples):
        states, actions, rewards, next_states, dones = samples
        expected_q_values = self.net(states).gather(1, actions.view(-1, 1))
        # Double DQN target
        next_a = self.net(next_states).max(1)[1].unsqueeze(1)
        target_q_values = rewards + self.gamma * self.target_net(next_states).gather(1, next_a).detach() * (1-dones)
        loss = F.mse_loss(expected_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()


class DQNAgentPER(DQNAgentBase):
    def __init__(self, net, target_net, alpha=0.6, beta=0.4, beta_delta=1.001, e=1e-8, **kwargs):
        super(DQNAgentPER, self).__init__(net, target_net, **kwargs)
        self.memory = PrioritizedReplayBuffer(**kwargs)
        self.__alpha = alpha
        self.__beta = beta
        self.__beta_delta = beta_delta
        self.__e = e

    def _learn(self, samples):
        states, actions, rewards, next_states, dones, idxs, probs = samples
        expected_q_values = self.net(states).gather(1, actions)
        # DQN target
        target_q_values_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * target_q_values_next * (1 - dones)
        td_err = expected_q_values - target_q_values  # calc td error
        weights = (probs * self.memory.size()).pow(-self.__beta).to(self.device)
        weights = weights / weights.max()
        loss = torch.mean(td_err.pow(2).squeeze() * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update(idxs.cpu().numpy(),
                             td_err.abs().detach().cpu().numpy().squeeze() ** self.__alpha + self.__e)
        return loss.detach().cpu().numpy()

    def step(self, state, action, reward, next_state, done):
        loss = super(DQNAgentPER, self).step(state, action, reward, next_state, done)
        if done:
            self.__beta = min(1., self.__beta * self.__beta_delta)
        return loss


class DDQNAgentPER(DQNAgentBase):
    def __init__(self, net, target_net,  alpha=0.6, beta=0.4, beta_delta=1.001, e=1e-8, **kwargs):
        super(DDQNAgentPER, self).__init__(net, target_net, **kwargs)
        self.memory = PrioritizedReplayBuffer(**kwargs)
        self.__alpha = alpha
        self.__beta = beta
        self.__beta_delta = beta_delta
        self.__e = e

    def _learn(self, samples):
        states, actions, rewards, next_states, dones, idxs, probs = samples
        expected_q_values = self.net(states).gather(1, actions)
        # DDQN target
        next_a = self.net(next_states).max(1)[1].unsqueeze(1)
        target_q_values = rewards + \
                          self.gamma * self.target_net(next_states).gather(1, next_a) * (1 - dones)
        td_err = expected_q_values - target_q_values  # calc td error
        weights = (probs * self.memory.size()).pow(-self.__beta).to(self.device)
        weights = weights / weights.max()
        loss = torch.mean(td_err.pow(2).squeeze() * weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update(idxs.cpu().numpy(),
                             td_err.abs().detach().cpu().numpy().squeeze() ** self.__alpha + self.__e)
        return loss.detach().cpu().numpy()

    def step(self, state, action, reward, next_state, done):
        loss = super(DDQNAgentPER, self).step(state, action, reward, next_state, done)
        if done:
            self.__beta = min(1., self.__beta * self.__beta_delta)
        return loss
