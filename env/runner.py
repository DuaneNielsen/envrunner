import time
import numpy as np
import torch
from collections import OrderedDict
from torchvision.io import write_video, write_jpeg, write_png
from pathlib import Path


class EnvObserver:
    def reset(self):
        """ called before environment reset"""
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        """ called each environment step """
        pass

    def done(self):
        """ called when episode ends """
        pass


class StateCapture(EnvObserver):
    def __init__(self):
        self.trajectories = []
        self.traj = []
        self.index = []
        self.cursor = 0

    def reset(self):
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        self.traj.append(state)
        self.index.append(self.cursor)
        self.cursor += 1

    def done(self):
        self.trajectories += [self.traj]
        self.traj = []


class ReplayBuffer(EnvObserver):
    def __init__(self):
        """
        Replay buffer

        Attributes:
        buffer          [(s, a, r, d, i), ...]
        trajectories    [(start, end), ...]
        transitions     a flat index into buffer that points to the head of each transition
                        ie: to retrieve the n'th transition s, a, s_prime, r, done
                        transition_index = transitions[n]
                        s, _, _, _, _ = buffer[transition_index]
                        s_prime, a, r, done, _ = buffer[transtion_index + 1]
        """
        self.buffer = []
        self.trajectories = []
        self.transitions = []
        self.traj_start = 0

    def reset(self):
        pass

    def step(self, state, action, reward, done, info, **kwargs):

        self.buffer.append((state, action, reward, done, info))

        if done:
            """ terminal state, trajectory is complete """
            self.trajectories.append((self.traj_start, len(self.buffer)))
            self.traj_start = len(self.buffer)
        else:
            """ if not terminal, then by definition, this will be a transition """
            self.transitions.append(len(self.buffer)-1)

    def done(self):
        pass

    def get_trajectory(self, item):
        start, end = self.trajectories[item]
        return self.buffer[start:end]

    def len_trajectories(self):
        return len(self.trajectories)

    def get_transition(self, item):
        i = self.transitions[item]
        s, _, _, _, _ = self.buffer[i]
        s_p, a, r, d, _ = self.buffer[i+1]
        return s, a, s_p, r, d

    def len_transitions(self):
        _, _, _, done, _ = self.buffer[-1]
        """ if the final state is not done, then we are still writing """
        if not done:
            """ we cant use the transition at the end yet"""
            return len(self.transitions) - 1
        return len(self.transitions)


class TransitionDataset:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __getitem__(self, item):
        return self.replay_buffer.get_transtion(item)

    def __len__(self):
        return self.replay_buffer.len_transitions()


class TrajectoryDataset:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __getitem__(self, item):
        return self.replay_buffer.get_trajectory(item)

    def __len__(self):
        return self.replay_buffer.len_trajectories()


class VideoCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0

    def reset(self):
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        self.t.append(state)

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        write_video(f'{self.directory}/capture_{self.cap_id}.mp4', stream, 24.0)
        self.cap_id += 1


class JpegCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        self.t.append(state)

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_jpeg(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.jpg')
            self.image_id += 1


class PngCapture(EnvObserver):
    def __init__(self, directory):
        self.t = []
        self.directory = directory
        self.cap_id = 0
        self.image_id = 0

    def reset(self):
        pass

    def step(self, state, action, reward, done, info, **kwargs):
        self.t.append(state)

    def done(self):
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        stream = torch.from_numpy(np.stack(self.t))
        for image in stream:
            write_png(image.permute(2, 0, 1), f'{self.directory}/{self.image_id}.png')
            self.image_id += 1


class StepFilter:
    """
    Step filters are used to preprocess steps before handing them to observers

    if you want to pre-process environment observations before passing to policy, use a gym.Wrapper
    """
    def __call__(self, state, action, reward, done, info, **kwargs):
        return state, action, reward, done, info, kwargs


class RewardFilter(StepFilter):
    def __init__(self, state_prepro, R, device):
        self.state_prepro = state_prepro
        self.R = R
        self.device = device

    def __call__(self, state, action, reward, done, info, **kwargs):
        r = self.R(self.state_prepro(state, self.device))
        kwargs['model_reward'] = r.item()
        return state, action, reward, done, info, kwargs


class EnvRunner:
    """
    environment loop with pluggable observers

    to attach an observer implement EnvObserver interface and use attach()

    filters to process the steps are supported, and data enrichment is possible
    by adding to the kwargs dict
    """
    def __init__(self, env, seed=None, **kwargs):
        self.kwargs = kwargs
        self.env = env
        if seed is not None:
            env.seed(seed)
        self.observers = OrderedDict()
        self.step_filters = OrderedDict()

    def attach_observer(self, name, observer):
        self.observers[name] = observer

    def detach_observer(self, name):
        del self.observers[name]

    def observer_reset(self):
        for name, observer in self.observers.items():
            observer.reset()

    def append_step_filter(self, name, filter):
        self.step_filters[name] = filter

    def observe_step(self, state, action, reward, done, info, **kwargs):
        for name, filter in self.step_filters.items():
            state, action, reward, done, info, kwargs = filter(state, action, reward, done, info, **kwargs)

        for name, observer in self.observers.items():
            observer.step(state, action, reward, done, info, **kwargs)

    def observer_episode_end(self):
        for name, observer in self.observers.items():
            observer.done()

    def render(self, render, delay):
        if render:
            self.env.render()
            time.sleep(delay)

    def reset(self, **kwargs):
        self.observer_reset()
        state, reward, done, info = self.env.reset(), 0.0, False, {}
        self.observe_step(state, None, reward, done, info, **kwargs)
        return state

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        self.observe_step(state, action, reward, done, info, **kwargs)
        return state, reward, done, info


def episode(runner, policy, render=False, delay=0.01, **kwargs):
    """

    :param policy: takes state as input, and must output an Action
    :param render: if True will call environments render function
    :param delay: rendering delay
    :param kwargs: kwargs will be passed to policy, environment step, and observers
    """
    with torch.no_grad():
        state, reward, done, info = runner.reset(**kwargs), 0.0, False, {}
        action = policy(state, **kwargs)
        runner.render(render, delay)
        while not done:
            state, reward, done, info = runner.step(action, **kwargs)
            action = policy(state, **kwargs)
            runner.render(render, delay)
