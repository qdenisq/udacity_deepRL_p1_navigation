from unityagents import UnityEnvironment
import numpy as np


class BananaEnvironment:
    def __init__(self, file_name=None, **kwargs):
        self.__env = UnityEnvironment(file_name=file_name, seed=1234)  # create environment
        self.__brain_name = self.__env.brain_names[0]
        self.__env.reset()
        self.__state_dim = self.__env.brains[self.__brain_name].vector_observation_space_size
        self.__action_dim = self.__env.brains[self.__brain_name].vector_action_space_size

    def step(self, action):
        env_info = self.__env.step(action)[self.__brain_name]  # step
        next_state = env_info.vector_observations[0][np.newaxis]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return next_state, reward, done

    def reset(self, train_mode=True):
        env_info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        state = env_info.vector_observations[0][np.newaxis]
        return state

    def get_state_dim(self):
        return self.__state_dim

    def get_action_dim(self):
        return self.__action_dim

    def close(self):
        self.__env.close()


class VisualBananaEnvironment:
    def __init__(self, file_name=None, num_stacked_frames=4, **kwargs):
        self.__env = UnityEnvironment(file_name=file_name, seed=1234)  # create environment
        self.__brain_name = self.__env.brain_names[0]
        self.__num_stacked_frames = num_stacked_frames
        self.reset()
        self.__state_dim = np.array(self.__cur_state.shape)
        self.__action_dim = self.__env.brains[self.__brain_name].vector_action_space_size

    def step(self, action):
        env_info = self.__env.step(action)[self.__brain_name]  # step
        next_obs = env_info.visual_observations[0].transpose((3, 0, 1, 2))  # get the next state
        self.__cur_state = np.roll(self.__cur_state, 1, axis=2)
        self.__cur_state[0, :, 0, :, :] = next_obs.squeeze()
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return self.__cur_state, reward, done

    def reset(self, train_mode=True):
        env_info = self.__env.reset(train_mode=train_mode)[self.__brain_name]
        obs = env_info.visual_observations[0].transpose((3, 0, 1, 2))
        self.__cur_state = np.zeros((1, obs.shape[0], self.__num_stacked_frames, obs.shape[2], obs.shape[3]))
        self.__cur_state[0, :, 0, :, :] = obs.squeeze()
        self.__cur_state[0, :, 1, :, :] = obs.squeeze()
        self.__cur_state[0, :, 2, :, :] = obs.squeeze()
        self.__cur_state[0, :, 3, :, :] = obs.squeeze()
        return self.__cur_state

    def get_state_dim(self):
        return self.__state_dim

    def get_action_dim(self):
        return self.__action_dim

    def close(self):
        self.__env.close()
