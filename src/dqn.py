import numpy as np


class DQN:
    def __init__(self, env, agent, initial_eps=1.0, min_eps=0.01, eps_decay=0.995, **kwargs):
        self.__env = env
        self.__agent = agent
        self.__eps = self.__eps_init = initial_eps
        self.__min_eps = min_eps
        self.__eps_decay = eps_decay

    def train(self, num_episodes, target_score=13.0, verbose=1):
        solved = False
        scores = []
        losses = []
        for i in range(1, num_episodes + 1):
            state = self.__env.reset()
            done = False
            score = 0
            loss = 0
            while not done:
                action = self.__agent.act(state, self.__eps)  # choose action
                next_state, reward, done = self.__env.step(action)  # roll out transition
                loss += self.__agent.step(state, action, reward, next_state, done)  # agent's update routine
                score += reward
                state = next_state
            self.__eps = max(self.__min_eps, self.__eps * self.__eps_decay)  # decay epsilon
            scores.append(score)  # track scores
            losses.append(loss)  # track losses

            avg_score = np.mean(scores[max(len(scores) - 100, 0):])
            avg_loss = np.mean(losses[max(len(losses) - 100, 0):])

            if not solved and avg_score > target_score:
                solved = True
                print('\n\n----------Env solved: score = {} | num_episodes = {}| -------------\n\n'.format(avg_score, i - 100))
                return scores, losses
            if verbose:  # print routine
                print("\r|progress: {:.1f}%| episode: {}| score: {}| avg score: {:.2f}| loss: {:.2f}| avg_loss: {:.2f}"
                      .format(i * 100 / num_episodes, i, score, avg_score, loss, avg_loss), end='')
                if i % 100 == 0:
                    print()
            i += 1
        return scores, losses



