import numpy as np


class DQN:
    def __init__(self, env, agent, initial_eps=1.0, min_eps=0.01, eps_decay=0.995, **kwargs):
        self.__env = env
        self.__agent = agent
        self.__eps = self.__eps_init = initial_eps
        self.__min_eps = min_eps
        self.__eps_decay = eps_decay

    def train(self, num_episodes, verbose=1):
        scores = []
        losses = []
        for i in range(num_episodes):
            state = self.__env.reset()
            done = False
            score = 0
            loss = 0
            while not done:
                action = self.__agent.act(state, self.__eps)  # choose action
                next_state, reward, done = self.__env.step(action)  # rol out transition
                step_loss = self.__agent.step(state, action, reward, next_state, done)  # agent's update routine
                if step_loss:
                    loss += step_loss
                score += reward
            self.__eps = max(self.__min_eps, self.__eps * self.__eps_decay)  # decay epsilon
            scores.append(score)  # track scores
            losses.append(loss)  # track losses
            if verbose:  # print routine
                avg_score = np.mean(scores[max(len(scores) - 100, 0):])
                avg_loss = np.mean(scores[max(len(losses) - 100, 0):])

                print("\r|progress: {:.1f}%| episode: {}| score: {}| avg score: {:.1f}| loss: {:.2f}| avg_loss: {:.2f}"
                      .format(i * 100 / num_episodes, i, score, avg_score, loss, avg_loss), end='')
                if i % 100 == 0:
                    print()
        return scores, losses



