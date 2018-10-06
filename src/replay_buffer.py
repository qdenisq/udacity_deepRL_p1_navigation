import numpy as np
import random
from collections import namedtuple, deque
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, minibatch_size, seed=0):
        self.__deque = deque(maxlen=buffer_size)
        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__minibatch_size = minibatch_size
        self.__seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        experience = self.__experience(state, action, reward, next_state, done)
        self.__deque.append(experience)

    def sample(self, num_samples=None):
        k = self.__minibatch_size if num_samples is None else num_samples
        samples = random.sample(self.__deque, k)
        states = torch.Tensor([s.state for s in samples if s is not None]).float()
        actions = torch.Tensor([s.action for s in samples if s is not None]).long()
        rewards = torch.Tensor([s.reward for s in samples if s is not None]).float()
        next_states = torch.Tensor([s.next_state for s in samples if s is not None]).float()
        dones = torch.Tensor([s.done for s in samples if s is not None]).float()

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.__deque)


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, minibatch_size, seed=0):
        self.__deque = deque(maxlen=buffer_size)
        self.__keys = np.zeros(2 * buffer_size)
        self.__experiences = np.zeros(buffer_size, dtype=object)
        self.__len = 0
        self.__insert_pos = 0
        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__minibatch_size = minibatch_size
        self.__seed = random.seed(seed)
        self.__buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done, key):
        experience = self.__experience(state, action, reward, next_state, done)
        self.__deque.append(experience)

    def _add_to_tree(self, experience, key):
        idx = self.__insert_pos + self.__buffer_size

        self.__experiences[self.__insert_pos] = experience
        if self.__len < self.__buffer_size:
            self.__len += 1

        self.__insert_pos = (self.__insert_pos + 1) % self.__buffer_size
        self._update_tree(idx, key)
        return idx

    def sample(self):
        k = self.__minibatch_size - len(self.__deque)

        total = self.__keys[1]
        idxs = [self._sift(random.uniform(seg / k, (seg + 1) / k) * total) for seg in range(k)]
        probs = [self.__keys[i] / total for i in idxs]
        samples = [self.__experiences[i - self.__buffer_size] for i in idxs]

        # add elems from deque
        for v in self.__deque:
            samples.append(v)
            idx = self._add_to_tree(v, 0.)
            idxs.append(idx)
            probs.append(0)
        self.__deque.clear()

        states = torch.Tensor([s.state for s in samples if s is not None]).float()
        actions = torch.Tensor([s.action for s in samples if s is not None]).long()
        rewards = torch.Tensor([s.reward for s in samples if s is not None]).float()
        next_states = torch.Tensor([s.next_state for s in samples if s is not None]).float()
        dones = torch.Tensor([s.done for s in samples if s is not None]).float()

        return states, actions, rewards, next_states, dones, idxs, probs

    def update(self, idxs, new_keys):
        for idx, key in zip(idxs, new_keys):
            self._update_tree(idx, key)

    def size(self):
        return self.__len + len(self.__deque)

    def total(self):
        return self.__keys[1]

    def _update_tree(self, idx, key):
        delta = key - self.__keys[idx]
        while idx >= 1:
            self.__keys[idx] += delta
            idx = idx // 2

    def _sift(self, rn):
        i = 1

        while i < self.__buffer_size:
            if self.__keys[2*i] > rn:
                i *= 2
            else:
                rn = rn - self.__keys[2*i]
                i = i * 2 + 1
        return i
