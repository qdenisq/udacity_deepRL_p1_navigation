# Udacity Deep Reinforcment Learning Nanodegree: project 1

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

### 1 - Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### 2 - Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

## 3 - Project Structure
- folder `Banana_env` contains simple banana environment with continuous state space
- folder `VisualBanana_env` contains banana environment with pixel state representation (`84*84` RGB image) 
- folder `data/models` contains saved trained models
- folder `reports` contains saved training scores
- folder `src` contains all source code
  - `replay_buffer.py` contains 2 classes for experience replay: `ReplayBuffer` for regular experience replay, and `PrioritizedReplayBuffer` for prioritized experience replay
  - `neural_net.py` contains simple MLP and Convolution NNs
  - `environment.py` contains wrappers for 2 environments presented in this project
  - `agent.py` contains implementations of 4 algorithms: DQN, Double DQN, DQN+PER, Double DQN+PER
  - `dqn.py` contains common dqn routine used for training all the agents
  - `train.py`  script for training any presented agent in any of 2 environments
  - `play.py`  script for running trained agent
  - `plot_graphs.ipynb`  notebook for plotting training scores

## 4 - Training

To train the agent you need to run the following script:
>```console
> (drlnd) $ python3 ./train.py --env_file=[path to the environment file] --env_type='simple'
>```

You can also pass additional optional arguments to specify all hyperparameters used in the training.
To see the list of these optional parameters, run the following script:
>```console
> (drlnd) $ python3 ./train.py -h
>```

**Note**
To train the agent in the visual banana environment, pass `--env_type='visual':
>```console
> (drlnd) $ python3 ./train.py --env_file=[path to the environment file] --env_type='visual'
>```

## 5 - Play
To play the trained agent, you need to run the following script

>```console
> (drlnd) $ python3 ./play.py --env_file=[path to the environment file] --agent_fname=[path to the saved agent's file]
>``` 


