import argparse
import datetime
import pickle
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from model import DQNTrainer
from agent import DQNAgent, DQNAgentWithPrioritizedReplay


def train(**kwargs):
    env = UnityEnvironment(file_name=kwargs['env_file'], no_graphics=True)  # create environment
    brain_name = env.brain_names[0]
    state_dim = env.brains[brain_name].vector_observation_space_size
    num_actions = env.brains[brain_name].vector_action_space_size
    if kwargs['use_prioritized_buffer']:
        agent = DQNAgentWithPrioritizedReplay(state_dim,
                         num_actions,
                         kwargs['init_epsilon'],
                         kwargs['epsilon_decay'],
                         kwargs['min_epsilon'],
                         kwargs['lr'],
                         kwargs['batch_size'],
                         kwargs['gamma'],
                         kwargs['tau'],
                         kwargs['update_every'],
                         kwargs['replay_buffer_size'],
                         kwargs['alpha'],
                         kwargs['beta'],
                         kwargs['e']
                         )  # create agent
    else:
        agent = DQNAgent(state_dim,
                     num_actions,
                     kwargs['init_epsilon'],
                     kwargs['epsilon_decay'],
                     kwargs['min_epsilon'],
                     kwargs['lr'],
                     kwargs['batch_size'],
                     kwargs['gamma'],
                     kwargs['tau'],
                     kwargs['update_every'],
                     kwargs['replay_buffer_size'],
                     )  # create agent
    trainer = DQNTrainer(env, agent)  # create trainer

    scores = trainer.train(kwargs['num_episodes'])  # run the trainer
    # save agent
    dt = str(datetime.datetime.now().strftime("%m_%d_%Y_%I_%M_%p"))
    model_fname = kwargs['model_dir'] + '/banana_collector_agent_{}.pt'.format(dt)
    agent.save_ckpt(model_fname)

    scores_fname = kwargs['reports_dir'] + '/banana_collector_agent_{}.pt'.format(dt)
    with open(scores_fname, "wb") as f:
        pickle.dump(scores, f)

    plt.plot(scores)
    plt.show()

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                        help='tag for current run')
    parser.add_argument('--env_file', type=str,
                        help='file path of Unity environment')
    parser.add_argument('--model_dir', type=str, default='../data/models',
                        help='basedir for saving model weights')
    parser.add_argument('--reports_dir', type=str, default='../reports',
                        help='basedir for saving training reports')
    # train params
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='number of episodes to train an agent')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    # replay buffer params
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e5),
                        help='size of the replay buffer')
    parser.add_argument('--use_prioritized_buffer', type=bool, default=False,
                        help='if set True, buffer uses TDerror for importance sampling')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='alpha param for prioritized replay buffer')
    parser.add_argument('--beta', type=float, default=0.4,
                        help='beta param for prioritized replay buffer')
    parser.add_argument('--e', type=float, default=1e-8,
                        help='additive constant for priorities in prioritized replay buffer')
    # dqn params
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='soft update for target networks')
    parser.add_argument('--update_every', type=int, default=4,
                        help='update target networks each n steps')
    # agent params
    parser.add_argument('--init_epsilon', type=float, default=1.0,
                        help='initial epsilon of the e-greedy policy')
    parser.add_argument('--epsilon_decay', type=float, default=0.99,
                        help='epsilon decay')
    parser.add_argument('--min_epsilon', type=float, default=0.001,
                        help='minimum of the epsilon')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='discount factor')
    # q_net params
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of the hidden layer')

    args = parser.parse_args()
    train(**vars(args))

