from env import runner as run


class StaticEnv:
    def __init__(self):
        self.len = 3
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        self.state += 1
        done = self.len - 1 == self.state
        return self.state, 0.0, done, {}


class DummyEnv:
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.t = 0
        self.i = 0

    def reset(self):
        self.i = 0
        return self.trajectories[self.t][0][self.i]

    def step(self, action):
        self.i += 1
        s, r, d, i = self.trajectories[self.t][self.i]
        if d:
            self.t += 1
        return s, r, d, i


def transition_equal(t1, t2):
    s1, a1, sp_1, r1, d1 = t1
    s2, a2, sp_2, r2, d2 = t2
    assert (s1 == s2) and (sp_1 == sp_2)
    assert (a1.action == a2.action)
    assert (r1 == r2)
    assert (d1 == d2)


def test_buffer():

    t1 = [(0, 0.0, False, {}), (1, 0.0, False, {}), (2, 0.0, True, {})]
    t2 = [(0, 0.0, False, {}), (1, 0.0, True, {})]
    traj = [t1, t2]

    env = DummyEnv(traj)
    runner = run.EnvRunner(env)
    buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", buffer)

    def policy(state):
        return run.Action(action=0)

    run.episode(runner, policy)
    run.episode(runner, policy)

    print(buffer.trajectories)
    start, end = buffer.trajectories[0]
    assert len(buffer.buffer[start:end]) == 3
    start, end = buffer.trajectories[1]
    assert len(buffer.buffer[start:end]) == 2
    assert len(buffer.transitions) == 3

    transition = buffer.get_transition(0)
    expected = 0, run.Action(action=0), 1, 0.0, False
    transition_equal(transition, expected)

    transition = buffer.get_transition(1)
    expected = 1, run.Action(action=0), 2, 0.0, True
    transition_equal(transition, expected)

    transition = buffer.get_transition(2)
    expected = 0, run.Action(action=0), 1, 0.0, True
    transition_equal(transition, expected)


def test_load_before_trajectory_terminates():

    env = DummyEnv([])
    runner = run.EnvRunner(env)
    buffer = run.ReplayBuffer()
    runner.attach_observer("replay_buffer", buffer)

    """ first step, from env reset """
    step = 0, run.Action(action=0), 0.0, False, {}
    runner.observe_step(*step)
    assert buffer.len_transitions() == 0
    assert buffer.len_trajectories() == 0

    """ second step, intermediate step"""
    step = 1, run.Action(action=0), 0.0, False, {}
    runner.observe_step(*step)
    assert buffer.len_transitions() == 1
    assert buffer.len_trajectories() == 0
    expected_transition = 0, run.Action(action=0), 1, 0.0, False
    transition_equal(buffer.get_transition(0), expected_transition)

    """ third step, trajectory ends """
    step = 2, run.Action(action=0), 1.0, True, {}
    runner.observe_step(*step)
    assert buffer.len_transitions() == 2
    assert buffer.len_trajectories() == 1
    expected_transition = 0, run.Action(action=0), 1, 0.0, False
    transition_equal(buffer.get_transition(0), expected_transition)
    expected_transition = 1, run.Action(action=0), 2, 1.0, True
    transition_equal(buffer.get_transition(1), expected_transition)

    """ forth step, 2nd trajectory resets  """
    step = 3, run.Action(action=0), 0.0, False, {}
    runner.observe_step(*step)
    assert buffer.len_transitions() == 2
    assert buffer.len_trajectories() == 1
    expected_transition = 0, run.Action(action=0), 1, 0.0, False
    transition_equal(buffer.get_transition(0), expected_transition)
    expected_transition = 1, run.Action(action=0), 2, 1.0, True
    transition_equal(buffer.get_transition(1), expected_transition)

    """ fifth step, 2nd trajectory ends """
    step = 4, run.Action(action=0), 1.0, True, {}
    runner.observe_step(*step)
    assert buffer.len_transitions() == 3
    assert buffer.len_trajectories() == 2
    expected_transition = 0, run.Action(action=0), 1, 0.0, False
    transition_equal(buffer.get_transition(0), expected_transition)
    expected_transition = 1, run.Action(action=0), 2, 1.0, True
    transition_equal(buffer.get_transition(1), expected_transition)
    expected_transition = 3, run.Action(action=0), 4, 1.0, True
    transition_equal(buffer.get_transition(2), expected_transition)
