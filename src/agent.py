import random
import torch
import torch.nn as nn

from neural_net import MlpQNetwork
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

SEED = 1234             # seed used in sampling from the replay buffer and e-greedy policy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self,
                 state_dim,
                 num_actions,
                 initial_epsilon=1.,
                 epsilon_decay=0.995,
                 min_epsilon=0.001,
                 lr=5e-4,
                 minibatch_size=100,
                 gamma=0.99,
                 tau=1e-3,
                 update_every=4,
                 buffer_size=int(1e5)):
        self.__state_dim = state_dim
        self.__num_actions = num_actions

        self.__Qnetwork = MlpQNetwork(self.__state_dim, self.__num_actions)
        self.__target_Qnetwork = MlpQNetwork(self.__state_dim, self.__num_actions)
        self.__optimizer = torch.optim.Adam(self.__Qnetwork.parameters(), lr=lr)
        self.__step_i = 0
        self.__epsilon = self.__initial_epsilon = initial_epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__update_every = update_every
        self.__minibatch_size = minibatch_size
        self.__tau = tau
        self.__gamma = gamma
        self.__memory = ReplayBuffer(buffer_size, minibatch_size)

    def choose_action(self, state, greedy=False):
        q_s_values = self.__Qnetwork(torch.from_numpy(state).float().view(1, -1))
        # follow e-greedy policy if in train mode, otherwise follow greedy policy
        if not greedy and random.random() < self.__epsilon:
            action = random.randint(0, self.__num_actions - 1)
        else:
            action = q_s_values.max(1)[1].item()
        return action

    def reset_epsilon(self):
        self.__epsilon = self.__initial_epsilon

    def step(self, state, action, reward, next_state, done):
        self.__memory.add(state, action, reward, next_state, done)

        if self.__step_i % self.__update_every == 0 and self.__memory.size() > self.__minibatch_size:
            # sample and train
            samples = self.__memory.sample()
            self.__update(samples)
            # decay epsilon after each update
            self.__epsilon = max(self.__min_epsilon, self.__epsilon * self.__epsilon_decay)
            
            self.soft_update(self.__Qnetwork, self.__target_Qnetwork, self.__tau)
        self.__step_i += 1
        
    def __update(self, samples):
        states, actions, rewards, next_states, dones = samples
        self.__optimizer.zero_grad()

        expected_q_values = self.__Qnetwork(states).gather(1, actions.view(-1, 1))

        # Double DQN target
        next_action = self.__Qnetwork(next_states).max(1)[1].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) +\
            self.__gamma * self.__target_Qnetwork(next_states).gather(1, next_action) * (1 - dones).unsqueeze(1)

        # DQN target
        # target_q_values = rewards + self.__gamma * self.__target_Qnetwork(next_states).max(1)[0] * (1 - dones)
        loss = nn.MSELoss()(expected_q_values, target_q_values.detach().view(-1, 1))
        loss.backward()
        self.__optimizer.step()
        return loss
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save_ckpt(self, fname):
        ckpt = {
            'state_dim': self.__state_dim,
            'num_actions': self.__num_actions,
            # 'replay_buffer': self.__memory,
            'q_network': self.__Qnetwork.state_dict(),
            'target_q_network': self.__target_Qnetwork.state_dict(),
            'optimizer': self.__optimizer.state_dict(),
            'epsilon': self.__epsilon,
            'init_epsilon': self.__initial_epsilon,
            'epsilon_decay': self.__epsilon_decay,
            'min_epsilon': self.__min_epsilon,
            'update_every': self.__update_every,
            'minibatch_size': self.__minibatch_size,
            'tau': self.__tau,
            'gamma': self.__gamma
        }
        torch.save(ckpt, fname)

    def save_model(self, fname):
        torch.save(self.__Qnetwork.state_dict(), fname)

    @staticmethod
    def load_from_ckpt(fname):
        ckpt = torch.load(fname)
        agent = DQNAgent(ckpt['state_dim'], ckpt['num_actions'])
        agent.load_from_dict(ckpt)
        return agent

    def load_from_dict(self, ckpt):
        self.__state_dim = ckpt['state_dim']
        self.__num_actions = ckpt['num_actions']
        # self.__memory = ckpt['replay_buffer']
        self.__Qnetwork.load_state_dict(ckpt['q_network'])
        self.__target_Qnetwork.load_state_dict(ckpt['target_q_network'])
        self.__optimizer.load_state_dict(ckpt['optimizer'])
        self.__epsilon = ckpt['epsilon']
        self.__initial_epsilon = ckpt['init_epsilon']
        self.__epsilon_decay = ckpt['epsilon_decay']
        self.__min_epsilon = ckpt['min_epsilon']
        self.__update_every = ckpt['update_every']
        self.__minibatch_size = ckpt['minibatch_size']
        self.__tau = ckpt['tau']
        self.__gamma = ckpt['gamma']

    def load_model(self, fname):
        state_dict = torch.load(fname)
        self.__Qnetwork.load_state_dict(state_dict)


class DQNAgentWithPrioritizedReplay:
    def __init__(self,
                 state_dim,
                 num_actions,
                 initial_epsilon=1.,
                 epsilon_decay=0.995,
                 min_epsilon=0.001,
                 lr=5e-4,
                 minibatch_size=100,
                 gamma=0.99,
                 tau=1e-3,
                 update_every=4,
                 buffer_size=int(1e5),
                 alpha=0.6,
                 beta=0.4,
                 e=1e-8):
        self.__state_dim = state_dim
        self.__num_actions = num_actions

        self.__Qnetwork = MlpQNetwork(self.__state_dim, self.__num_actions)
        self.__target_Qnetwork = MlpQNetwork(self.__state_dim, self.__num_actions)
        self.__optimizer = torch.optim.Adam(self.__Qnetwork.parameters(), lr=lr)
        self.__step_i = 0
        self.__epsilon = self.__initial_epsilon = initial_epsilon
        self.__epsilon_decay = epsilon_decay
        self.__min_epsilon = min_epsilon
        self.__update_every = update_every
        self.__minibatch_size = minibatch_size
        self.__tau = tau
        self.__gamma = gamma
        self.__memory = PrioritizedReplayBuffer(buffer_size, minibatch_size)
        self.__alpha = alpha
        self.__beta = beta
        self.__e = e

    def choose_action(self, state, greedy=False):
        q_s_values = self.__Qnetwork(torch.from_numpy(state).float().view(1, -1))
        # follow e-greedy policy if in train mode, otherwise follow greedy policy
        if not greedy and random.random() < self.__epsilon:
            action = random.randint(0, self.__num_actions - 1)
        else:
            action = q_s_values.max(1)[1].item()
        return action

    def reset_epsilon(self):
        self.__epsilon = self.__initial_epsilon

    def step(self, state, action, reward, next_state, done):
        priority = self.__memory.total() / (self.__minibatch_size - 1) + self.__e
        # priority = 1. + self.__e
        self.__memory.add(state, action, reward, next_state, done, priority)

        if self.__step_i % self.__update_every == 0 and self.__memory.size() > self.__minibatch_size:
            # sample and train
            samples = self.__memory.sample()
            self.__update(samples)
            # decay epsilon after each update
            self.__epsilon = max(self.__min_epsilon, self.__epsilon * self.__epsilon_decay)

            self.soft_update(self.__Qnetwork, self.__target_Qnetwork, self.__tau)
        self.__step_i += 1

    def __update(self, samples):
        states, actions, rewards, next_states, dones, idxs, probs = samples
        self.__optimizer.zero_grad()

        expected_q_values = self.__Qnetwork(states).gather(1, actions.view(-1, 1))

        # Double DQN target
        next_action = self.__Qnetwork(next_states).max(1)[1].unsqueeze(1)
        target_q_values = rewards.unsqueeze(1) +\
            self.__gamma * self.__target_Qnetwork(next_states).gather(1, next_action) * (1 - dones).unsqueeze(1)

        losses = nn.MSELoss(reduce=False)(expected_q_values, target_q_values.detach().view(-1, 1))
        weights = [(p * self.__memory.size()) ** (-self.__beta) for p in probs]
        weights = [w / max(weights) for w in weights]
        loss = torch.mean(losses * torch.Tensor(weights))
        loss.backward()
        self.__optimizer.step()
        self.__memory.update(idxs, losses.detach().numpy().squeeze() ** self.__alpha + self.__e)
        return loss

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_ckpt(self, fname):
        ckpt = {
            'state_dim': self.__state_dim,
            'num_actions': self.__num_actions,
            # 'replay_buffer': self.__memory,
            'q_network': self.__Qnetwork.state_dict(),
            'target_q_network': self.__target_Qnetwork.state_dict(),
            'optimizer': self.__optimizer.state_dict(),
            'epsilon': self.__epsilon,
            'init_epsilon': self.__initial_epsilon,
            'epsilon_decay': self.__epsilon_decay,
            'min_epsilon': self.__min_epsilon,
            'update_every': self.__update_every,
            'minibatch_size': self.__minibatch_size,
            'tau': self.__tau,
            'gamma': self.__gamma,
            'alpha': self.__alpha,
            'beta': self.__beta,
            'e': self.__e
        }
        torch.save(ckpt, fname)

    def save_model(self, fname):
        torch.save(self.__Qnetwork.state_dict(), fname)

    @staticmethod
    def load_from_ckpt(fname):
        ckpt = torch.load(fname)
        agent = DQNAgent(ckpt['state_dim'], ckpt['num_actions'])
        agent.load_from_dict(ckpt)
        return agent

    def load_from_dict(self, ckpt):
        self.__state_dim = ckpt['state_dim']
        self.__num_actions = ckpt['num_actions']
        # self.__memory = ckpt['replay_buffer']
        self.__Qnetwork.load_state_dict(ckpt['q_network'])
        self.__target_Qnetwork.load_state_dict(ckpt['target_q_network'])
        self.__optimizer.load_state_dict(ckpt['optimizer'])
        self.__epsilon = ckpt['epsilon']
        self.__initial_epsilon = ckpt['init_epsilon']
        self.__epsilon_decay = ckpt['epsilon_decay']
        self.__min_epsilon = ckpt['min_epsilon']
        self.__update_every = ckpt['update_every']
        self.__minibatch_size = ckpt['minibatch_size']
        self.__tau = ckpt['tau']
        self.__gamma = ckpt['gamma']
        self.__alpha = ckpt['alpha']
        self.__beta = ckpt['beta']
        self.__e = ckpt['e']

    def load_model(self, fname):
        state_dict = torch.load(fname)
        self.__Qnetwork.load_state_dict(state_dict)


