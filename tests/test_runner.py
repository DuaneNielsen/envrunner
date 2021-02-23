from env import runner as run
import torch


class LinearEnv:
    def __init__(self):
        self.state = torch.tensor([0.0])

    def reset(self):
        self.state[0] = 0.0
        return self.state.clone()

    def step(self, action):
        self.state[0] += action
        return self.state.clone(), self.reward(), self.done(), None

    def reward(self):
        return 1.0 if self.state[0].item() > 1.0 else 0.0

    def done(self):
        return self.state[0].item() < 0.0 or self.state[0].item() > 1.0


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

    runner = run.EnvRunner(env)

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

    pass