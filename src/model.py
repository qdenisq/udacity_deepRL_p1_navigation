import numpy as np
from collections import deque

class DQNTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        self.__brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.__brain_name]
        self.__action_size = brain.vector_action_space_size
        print('Action dim:', self.__action_size)
        self.__state_size = brain.vector_observation_space_size
        print('State dim:', self.__state_size)

    def train(self, num_episodes):
        scores = []
        self.agent.reset_epsilon()

        for i in range(1, num_episodes + 1):
            done = False
            env_info = self.env.reset(train_mode=True)[self.__brain_name]  # reset the environment
            state = env_info.vector_observations[0]  # get the current state
            score = 0  # initialize the score
            losses = []
            while not done:
                action = self.agent.choose_action(state)  # choose next action
                env_info = self.env.step(action)[self.__brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                loss = self.agent.step(state, action, reward, next_state, done)  # store experience in agent's replay buffer and make an update of agent
                if loss:
                    losses.append(loss)
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
            scores.append(score)
            avg_score = np.mean(scores[max(len(scores) - 100, 0):])
            print("\r|progress: {:.1f}%| episode: {}| score: {}| avg score: {:.1f}| loss: {:.4f}"
                  .format(i*100/num_episodes, i, score, avg_score, np.mean(losses)), end='')
            if i % 100 == 0:
                print()
        return scores


class DQNVisualBananaTrainer:
    def __init__(self, env, agent, is_visual_state=False, num_stacked_frames=1):
        self.env = env
        self.agent = agent

        self.__brain_name = self.env.brain_names[0]
        self.__visual = is_visual_state
        self.__num_stacked_frames = num_stacked_frames

        # brain = self.env.brains[self.__brain_name]
        # self.__action_size = brain.vector_action_space_size
        # print('Action dim:', self.__action_size)
        # env_info = env.reset(train_mode=True)[self.__brain_name]
        # self.__state_size = env_info
        # print('State dim:', self.__state_size)

    def train(self, num_episodes):
        frames_deque = deque(maxlen=self.__num_stacked_frames)
        scores = []
        self.agent.reset_epsilon()

        for i in range(1, num_episodes + 1):
            done = False
            env_info = self.env.reset(train_mode=True)[self.__brain_name]  # reset the environment
            state = env_info.visual_observations[0].squeeze() if self.__visual else env_info.vector_observations[0].squeeze() # get the current state
            for s in range(self.__num_stacked_frames):
                frames_deque.append(state)
            score = 0  # initialize the score
            losses = []
            while not done:
                state = np.array(frames_deque)
                if self.__visual:
                    state = np.moveaxis(state, -1, 0)
                action = self.agent.choose_action(state)  # choose next action
                env_info = self.env.step(action)[self.__brain_name]  # send the action to the environment
                obs = env_info.visual_observations[0].squeeze() if self.__visual else env_info.vector_observations[0].squeeze()
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                frames_deque.append(obs)
                next_state = np.array(frames_deque)
                if self.__visual:
                    next_state = np.moveaxis(next_state, -1, 0)
                loss = self.agent.step(state, action, reward, next_state, done)  # store experience in agent's replay buffer and make an update of agent
                if loss:
                    losses.append(loss)
                score += reward  # update the score
            scores.append(score)
            avg_score = np.mean(scores[max(len(scores) - 100, 0):])
            print("\r|progress: {:.1f}%| episode: {}| score: {}| avg score: {:.1f}| loss: {:.4f}"
                  .format(i*100/num_episodes, i, score, avg_score, np.mean(losses)), end='')
            if i % 100 == 0:
                print()
        return scores
