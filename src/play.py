from environment import VisualBananaEnvironment, BananaEnvironment
from agent import DQNAgent, DDQNAgent, DQNAgentPER, DDQNAgentPER
import argparse


def play(**kwargs):
    env = BananaEnvironment(file_name=kwargs['env_file'], num_stacked_frames=kwargs['num_stacked_frames'])
    agent_name = kwargs['agent_fname']
    is_per = 'PER' in agent_name
    if 'ddqn' in agent_name:
        agent = DDQNAgentPER.load(agent_name) if is_per else DDQNAgent.load(agent_name)
    elif 'dqn' in agent_name:
        agent = DQNAgentPER.load(agent_name) if is_per else DQNAgent.load(agent_name)
    else:
        raise KeyError('Unknown agent type')

    for i in range(kwargs['num_plays']):
        done = False
        score = 0
        state = env.reset(train_mode=False)
        while not done:
            action = agent.act(state, eps=0.)
            env.step(action)
            state, reward, done = env.step(action)  # roll out transition
            score += reward
            print("\r play #{}, reward: {} | score: {}".format(i+1, reward, score), end='')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, default='Visual Banana Collector',
                        help='tag for current run')
    parser.add_argument('--env_file', type=str,
                        help='file path of Unity environment')
    parser.add_argument('--agent_fname', type=str,
                        help='file to load agent from')
    parser.add_argument('--num_plays', type=int, default=4,
                        help='number of episodes to run agent')
    parser.add_argument('--num_stacked_frames', type=int, default=4,
                        help='number of frames to stack for state representation')
    args = parser.parse_args()
    play(**vars(args))

