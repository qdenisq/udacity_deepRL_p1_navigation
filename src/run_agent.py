import argparse
from unityagents import UnityEnvironment

from agent import DQNAgent


def run_epsiode(env_fname, agent_fname):
    env = UnityEnvironment(file_name=env_fname)  # create environment
    agent = DQNAgent.load_from_ckpt(agent_fname)  # load agent

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state, greedy=True)  # choose next action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
    env.close()
    print('score: {}'. format(score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', type=str,
                        help='file path of Unity environment')
    parser.add_argument('--agent_file', type=str,
                        help='file path of agent')
    args = parser.parse_args()
    run_epsiode(args.env_file, args.agent_file)
