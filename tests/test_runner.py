from abc import ABC

from env import runner as run
import numpy as np
import gym
import torch
from torch import nn


class LinearEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.state = np.array([0.0])
        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(1, ), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def reset(self):
        self.state[0] = 0.0
        return self.state.copy()

    def step(self, action):
        self.state[0] += action
        return self.state.copy(), self.reward(), self.done(), None

    def reward(self):
        return 1.0 if self.state[0].item() > 1.0 else 0.0

    def done(self):
        return self.state[0].item() < 0.0 or self.state[0].item() > 1.0

    def render(self, mode='human'):
        pass


def test_linear_env():
    env = LinearEnv()

    env.reset()
    assert env.state[0] == 0.0

    state, reward, done, info = env.step(0.5)
    assert state[0] == 0.5
    assert done is False
    assert reward == 0.0

    state, reward, done, info = env.step(0.5)
    assert state[0] == 1.0
    assert done is False
    assert reward == 0.0

    state, reward, done, info = env.step(0.1)
    assert state[0] == 1.1
    assert done is True
    assert reward == 1.0

    env.reset()
    state, reward, done, info = env.step(-0.1)
    assert state[0] == -0.1
    assert done is True
    assert reward == 0.0

    env.reset()
    state, reward, done, info = env.step(0.0)
    assert state[0] == 0.0
    assert done is False
    assert reward == 0.0

    runner = run.SubjectWrapper(env)

    def policy(state):
        dist = torch.distributions.normal.Normal(0, 0.5)
        return dist.rsample()

    replay_buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", replay_buffer)
    for i in range(5):
        run.episode(runner, policy)
    print('')
    for start, end in replay_buffer.trajectories:
        for state, action, reward, done, info in replay_buffer.buffer[start:end]:
            print(state, action, reward, done, info)


def test_REINFORCE():
    env = LinearEnv()
    buffer = run.ReplayBuffer()
    env = run.SubjectWrapper(env)
    env.attach_observer("replay_buffer", buffer)

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.1]))

        def forward(self, state):
            return torch.distributions.Normal(loc=state * self.w, scale=0.25)

    policy_net = PolicyNet()
    optim = torch.optim.SGD(policy_net.parameters(), lr=0.1)

    def policy(state):
        state = torch.from_numpy(state)
        action = policy_net(state)
        return action.rsample().item()

    for epoch in range(10):
        for ep in range(10):
            run.episode(env, policy)

        """ create dataset using naive value function """
        discount = 0.99
        state = []
        action = []
        value = []
        for start, end in buffer.trajectories:
            v = 0
            for i in reversed(range(start+1, end)):
                s, _, _, _, _ = buffer.buffer[i-1]
                s_prime, a, r, _, _ = buffer.buffer[i]
                v = r + v * discount
                state += [s]
                action += [a]
                value += [v]

        state, action, value = torch.tensor(state), torch.tensor(action), torch.tensor(value)
        optim.zero_grad()
        a_dist = policy_net(state)
        loss = - torch.mean(torch.exp(a_dist.log_prob(action) + torch.log(value)))
        loss.backward()
        optim.step()

        print(policy_net.w)





