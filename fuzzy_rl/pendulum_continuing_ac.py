import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fuzzy_rl.agents import ActorCriticContinuing
from fuzzy_rl.env import make_continuing_pendulum
from fuzzy_rl.features import RBFFeaturesPendulum


def train(
    n_steps: int = 500_000,
    seed: int = 0,
    log_every: int = 5_000,
    render: bool = False,
):
    np.random.seed(seed)
    feat = RBFFeaturesPendulum(n_angle=12, n_vel=8, bandwidth=0.40)

    agent = ActorCriticContinuing(
        feat,
        alpha_w=0.10,  # value step size
        alpha_mu=0.02,  # policy-mean step size
        alpha_sigma=0.01,  # policy-std step size  (smaller: keep exploration)
        alpha_r=0.05,  # average-reward step size
        lam_w=0.80,
        lam_theta=0.80,
    )

    env, obs = make_continuing_pendulum(seed, render=render)
    x = feat(obs)

    reward_window = []
    history = []  # (step, running average reward)

    for t in range(1, n_steps + 1):
        a, mu, sigma = agent.act(x)
        a_env = np.clip(a, -2.0, 2.0)  # env expects torque in [-2,2]
        obs_next, R, terminated, truncated, _ = env.step([a_env])
        # terminated/truncated are always False on the unwrapped env -> continuing
        x_next = feat(obs_next)

        agent.update(x, a, mu, sigma, R, x_next)
        x = x_next

        reward_window.append(R)
        if t % log_every == 0:
            avg = float(np.mean(reward_window))
            reward_window.clear()
            history.append((t, avg))
            print(
                f"step {t:>8d} | avg reward/step ~ {avg:8.4f} | "
                f"R_bar = {agent.R_bar:8.4f} | sigma~{sigma:5.3f}"
            )

    return agent, history


if __name__ == "__main__":
    _, history = train(
        n_steps=50_000, seed=18101990, log_every=1_000, render=False
    )

    history = pd.DataFrame(history)
    history.columns = ["step", "avg_reward"]
    history = history.set_index("step")["avg_reward"]
    history.to_csv("results/pendulum_continuing.csv")
    history.plot()
    plt.show()
