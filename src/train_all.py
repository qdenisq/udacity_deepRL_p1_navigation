from train import train
import argparse
import time

def main(**kwargs):
    banana_env_fname = "../Banana_env/Banana.exe"
    visual_banana_env_fname = "../VisualBanana_env/Banana.exe"

    algs = ['dqn', 'dqn']
    per = [False, True]

    kwargs['num_episodes'] = 2000

    kwargs['worker_id'] = 0
    # run simple banana

    # kwargs['env_type'] = 'simple'
    # kwargs['env_file'] = banana_env_fname
    #
    # for a in algs:
    #     kwargs['agent_type'] = a
    #     for p in per:
    #         kwargs['use_prioritized_buffer'] = p
    #         print('-----Train config: \nEnv: {} \nAgent type: {} \nPER: {}\n\n'
    #               .format(kwargs['env_type'], kwargs['agent_type'], kwargs['use_prioritized_buffer']))
    #         train(**kwargs)
    #         time.sleep(2)
    #         kwargs['worker_id'] += 1

    # run visual banana
    kwargs['env_type'] = 'visual'
    kwargs['env_file'] = visual_banana_env_fname
    for a in algs:
        kwargs['agent_type'] = a
        for p in per:
            kwargs['use_prioritized_buffer'] = p
            print('\n\n\n--------Train config--------: \nEnv: {} \nAgent type: {} \nPER: {}\n'
                  . format(kwargs['env_type'], kwargs['agent_type'], kwargs['use_prioritized_buffer']))
            train(**kwargs)
            kwargs['worker_id'] += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', type=str, default='../data/models',
                        help='basedir for saving model weights')
    parser.add_argument('--reports_dir', type=str, default='../reports',
                        help='basedir for saving training reports')
    # train params
    parser.add_argument('--agent_type', type=str, default='dqn',
                        help='number of episodes to train an agent')
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='number of episodes to train an agent')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--num_stacked_frames', type=int, default=4,
                        help='number of frames to stack for state representation')
    # replay buffer params
    parser.add_argument('--replay_buffer_size', type=int, default=10000,
                        help='size of the replay buffer')
    parser.add_argument('--use_prioritized_buffer', type=bool, default=False,
                        help='if set True, use prioritized experience replay buffer')
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
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='epsilon decay')
    parser.add_argument('--min_epsilon', type=float, default=0.01,
                        help='minimum of the epsilon')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='discount factor')
    # q_net params
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of the hidden layer')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='whether use gpu or not')

    args = parser.parse_args()
    main(**vars(args))

