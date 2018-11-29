import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, **kwargs):
        super(MlpQNetwork, self).__init__()

        self.__state_dim = state_dim
        self.__num_actions = num_actions
        n_hidden_units = kwargs['hidden_size']
        self.__fc1 = nn.Linear(self.__state_dim, n_hidden_units[0])
        self.__fc2 = nn.Linear(n_hidden_units[0], n_hidden_units[1])
        self.__fc3 = nn.Linear(n_hidden_units[1], n_hidden_units[2])
        self.__fc4 = nn.Linear(n_hidden_units[2], self.__num_actions)

    def forward(self, x, training=False):
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = F.relu(self.__fc3(x))
        x = self.__fc4(x)
        return x


class ConvQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, **kwargs):
        super(ConvQNetwork, self).__init__()
        self.__state_dim = list(state_dim)
        in_channels = state_dim[1]
        num_filters = [128, 256, 256]
        # self.__bn0 = nn.BatchNorm3d(in_channels)
        self.__conv1 = torch.nn.Conv3d(in_channels, num_filters[0], kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.__bn1 = nn.BatchNorm3d(num_filters[0])
        self.__conv2 = torch.nn.Conv3d(num_filters[0], num_filters[1], kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.__bn2 = nn.BatchNorm3d(num_filters[1])
        self.__conv3 = torch.nn.Conv3d(num_filters[1], num_filters[2], kernel_size=(4, 3, 3), stride=(1, 1, 1))
        self.__bn3 = nn.BatchNorm3d(num_filters[2])
        self.__in_fc = self._get_shape()
        self.__num_actions = num_actions

        self.__fc1 = nn.Linear(self.__in_fc, 512)
        self.__fc2 = nn.Linear(512, 32)
        self.__fc_out = nn.Linear(32, self.__num_actions)

    def forward(self, x, training=False):
        x = F.relu(self.__bn1(self.__conv1(x)))
        x = F.relu(self.__bn2(self.__conv2(x)))
        x = F.relu(self.__bn3(self.__conv3(x)))
        x = x.view(x.size(0), -1)

        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = self.__fc_out(x)
        return x

    def _get_shape(self):
        x = torch.rand(self.__state_dim)
        x = F.relu(self.__bn1(self.__conv1(x)))
        x = F.relu(self.__bn2(self.__conv2(x)))
        x = F.relu(self.__bn3(self.__conv3(x)))
        size = x.data.view(1, -1).size(1)
        return size