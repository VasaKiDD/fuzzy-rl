import gymnasium as gym


def make_continuing_pendulum(seed: int = 0, render: bool = False):
    """Strip TimeLimit so the task is truly continuing. .unwrapped exposes the
    raw PendulumEnv, whose step() never sets terminated/truncated."""
    env = gym.make(
        "Pendulum-v1", render_mode="human" if render else None
    ).unwrapped
    obs, _ = env.reset(seed=seed)
    return env, obs
