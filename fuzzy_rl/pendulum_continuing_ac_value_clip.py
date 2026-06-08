import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fuzzy_rl.agents import ActorCriticContinuingClip
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

    agent = ActorCriticContinuingClip(
        feat,
        alpha_mu=0.02,
        alpha_sigma=0.01,
        alpha_r=0.01,  # reward-rate / variance EMA rate
        lam_w=0.80,
        lam_theta=0.80,
        m=2.0,  # fractional step  (m>=1 -> no overshoot)
        l=2.5,  # clip at +/- 2 sigma_r
        eps=1e-2,  # denominator floor
        use_bootstrap_denom=False,  # z.x : better conditioned on smooth Pendulum dynamics
    )

    env, obs = make_continuing_pendulum(seed, render=render)
    x = feat(obs)

    reward_window = []
    floored_at_log = 0
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
            floored_frac = (agent.n_floored - floored_at_log) / log_every
            floored_at_log = agent.n_floored
            history.append((t, avg))
            print(
                f"step {t:>8d} | avg R/step ~ {avg:8.4f} | R_bar = {agent.R_bar:8.4f} "
                f"| sigma_r = {agent.last_sigma_r:6.3f} | pi_sigma ~ {sigma:5.3f} "
                f"| floored = {floored_frac:5.1%}"
            )

    return agent, history


if __name__ == "__main__":
    _, history = train(
        n_steps=50_000, seed=18101990, log_every=1_000, render=False
    )

    history = pd.DataFrame(history)
    history.columns = ["step", "avg_reward"]
    history = history.set_index("step")["avg_reward"]
    history.to_csv("results/pendulum_continuing_trick.csv")
    history.plot()
    plt.show()
