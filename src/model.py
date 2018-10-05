import numpy as np


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
            while not done:
                action = self.agent.choose_action(state)  # choose next action
                env_info = self.env.step(action)[self.__brain_name]  # send the action to the environment
                next_state = env_info.vector_observations[0]  # get the next state
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                self.agent.step(state, action, reward, next_state, done)  # store experience in agent's replay buffer and make an update of agent
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
            scores.append(score)
            avg_score = np.mean(scores[max(len(scores) - 100, 0):])
            print("\r|progress: {:.1f}%| episode: {}| score: {}| avg score: {:.1f}|"
                  .format(i*100/num_episodes, i, score, avg_score), end='')
            if i % 100 == 0:
                print()
        return scores
