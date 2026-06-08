import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fuzzy_rl.agents import ActionValueContinuingClip
from fuzzy_rl.env import make_continuing_pendulum
from fuzzy_rl.features import RBFFeaturesPendulum


def train(
    n_steps=500_000, seed=0, log_every=5_000, render=False, target_mode="q"
):
    np.random.seed(seed)
    feat = RBFFeaturesPendulum(n_angle=12, n_vel=8, bandwidth=0.40)

    agent = ActionValueContinuingClip(
        feat,
        n_actions=9,  # torque grid on [-2, 2], includes 0
        alpha_r=0.01,
        lam=0.80,
        m=2.0,
        l=2.5,
        eps=1e-2,
        epsilon=0.10,
        use_bootstrap_denom=False,
        target_mode=target_mode,
    )

    env, obs = make_continuing_pendulum(seed, render=render)
    x = feat(obs)
    j, greedy = agent.act(x)

    reward_window = []
    floored_at_log = 0
    history = []

    for t in range(1, n_steps + 1):
        obs_next, R, terminated, truncated, _ = env.step([agent.actions[j]])
        x_next = feat(obs_next)
        j_next, greedy_next = agent.act(x_next)

        agent.update(x, j, R, x_next, j_next, greedy)
        x, j, greedy = x_next, j_next, greedy_next

        reward_window.append(R)
        if t % log_every == 0:
            avg = float(np.mean(reward_window))
            reward_window.clear()
            floored_frac = (agent.n_floored - floored_at_log) / log_every
            floored_at_log = agent.n_floored
            history.append((t, avg))
            qa = agent.q_all(x)
            print(
                f"step {t:>8d} | avg R/step ~ {avg:8.4f} | R_bar = {agent.R_bar:8.4f} "
                f"| sigma_r = {agent.last_sigma_r:6.3f} | greedy_a ~ {agent.actions[int(np.argmax(qa))]:+.2f} "
                f"| floored = {floored_frac:5.1%}"
            )

    return agent, history


if __name__ == "__main__":
    _, history = train(
        n_steps=100_000,
        seed=18101990,
        log_every=1_000,
        render=False,
        target_mode="q",
    )

    history = pd.DataFrame(history, columns=["step", "avg_reward"]).set_index(
        "step"
    )["avg_reward"]
    history.to_csv("results/pendulum_continuing_qlearning.csv")
    history.plot()
    plt.show()
