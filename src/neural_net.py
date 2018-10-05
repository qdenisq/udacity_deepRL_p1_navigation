import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(MlpQNetwork, self).__init__()

        self.__state_dim = state_dim
        self.__num_actions = num_actions

        self.__fc1 = nn.Linear(self.__state_dim, 8 * self.__state_dim)
        self.__fc2 = nn.Linear(8 * self.__state_dim, 4 * self.__state_dim)
        self.__fc3 = nn.Linear(4 * self.__state_dim, self.__num_actions)

    def forward(self, x):
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = self.__fc3(x)
        return x
