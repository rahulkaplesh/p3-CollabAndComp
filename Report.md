# Udacity Deep Reinforcement Learning Nanodegree

## Collaboration and Competition project Report

In this project report I tried to briefly summarize the learnings and final modeling decisions taken as part of the Collaboration and Competition project. After several attempts with different hyperparameters and models, I was able to find a setup that solves the environment with around 1350 steps. I implemented several optimizations but the main criterion to solve the environment in my experience was a long enough training time (sufficient number of episodes).

## Multi-Agent Deep Deterministic Policy Gradient
In this project I use to train two agents whose action space is continuous using an reinforcement learning method called Multi-Agent Deep Deterministic Policy Gradient (MADDPG).Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm is a new population DRL algorithm, which is proposed by [Lowe et al.](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). It can find the global optimization solution and can easily defeat various DRL methods, including DQN (Deep Q-Network), TRPO (Trust region policy optimization) and DDPG (Deep Deterministic Policy Gradient). MADDPG is a kind of "Actor-Critic" method. Unlike DDPG algorithm which trains each agent independantly, MADDPG trains actors and critics using all agents information (actions and states). However, the trained agent model (actor) can make an inference independentaly using its own state.

## Model architecture
 - Actor network

The actor network is a multi-layer perceptron (MLP) with 2 hidden layers, which maps states to actions.

```Input Layer —> 1st hidden layer (256 neurons, ReLu) —> Batch Normalization —> 2nd hidden layer (128 neurons, ReLu) —> Output layer (tanh)```

```class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
        
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
```

## Critic network

The critic network is also a multi-layer perceptron (MLP) with 2 hidden layers, which maps (state, action) pair to Q-value.
```
"""Critic (Value) Model."""

    def __init__(self, full_state_size, actions_size, seed=0, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(full_state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+actions_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.reset_parameters()
```
## Training algorithm
 - Actor Network and Critic Network of target agents are soft-updated respectively.
 - The agents using the current policy and exploration noise interact with the environment.
 - All the episodes are saved into the shared replay buffer.
 - For each agent, using a minibatch which is randomly selected from the reply buffer, the critic and the actor are trained using MADDPG training algorithm.

## Training Hyperparameters
After different configuration on hyper-parameters and noise for action exploration, the below set solves the environment. Firstly, I used Ornstein-Uhlenbeck for noise, but I switched to random noise with standard normal distribution, which works for my case.

```BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
UPDATE_FREQ = 1

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

NOISE_REDUCTION_RATE = 0.99
EPISODES_BEFORE_TRAINING = 500
NOISE_START=1.0
NOISE_END=0.1
```

## Solution Description:

Here i have created a python file called 'Tennis.py' to execute my solution and the model is trained on 'multi-core cpu'. To extract the
plot i have written a seperate script called 'parse.py' which extracts the average score values out of log file.

## Result: 
Average score reaches 0.5009000075235963 at 1603 episode, which is considered to solve the environment.
```
INFO: 2020-12-19 20:59:13,896 Episode 1596	Average Score: 0.43
INFO: 2020-12-19 20:59:36,731 Episode 1597	Average Score: 0.43
INFO: 2020-12-19 20:59:42,842 Episode 1598	Average Score: 0.43
INFO: 2020-12-19 21:01:54,325 Episode 1599	Average Score: 0.46
INFO: 2020-12-19 21:01:56,059 Episode 1600	Average Score: 0.46
INFO: 2020-12-19 21:04:08,523 Episode 1601	Average Score: 0.48
INFO: 2020-12-19 21:05:44,878 Episode 1602	Average Score: 0.50
INFO: 2020-12-19 21:06:13,519 Episode 1603	Average Score: 0.50
INFO: 2020-12-19 21:06:13,519 Environment solved in 1603 episodes!	Average Score: 0.50

```

## Plot of reward
![ ](/plot.png)

## Ideas for Future work
 - One of the most obvious directions for additional research would be the change of the model. In particular, one could try Proximal Policy Optimization (PPO) on this task, which seems to have worked for other people in the nanodegree. I could imagine it to work quite well on a problem like this, that is not very high-dimensional.
 - Fine tuning hyper-parameters is one of tasks to be done to stablize the policy to address the sudden drop of score after reaching the high score. For this, I'd like to trace changing parameters such as decayed noise to check if it is one of causes. Learning rate for actor and critic, and batch size are one to be tuned as well. Also, having different network architecture for actor and critic (deeper or wider) are something worth to be tried.

