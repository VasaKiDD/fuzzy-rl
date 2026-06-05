import numpy as np
import gymnasium as gym
import pandas as pd
from matplotlib import pyplot as plt

from fuzzy_rl.features import RBFFeatures
from fuzzy_rl.env import make_continuing_pendulum


class ActorCriticContinuing:
    def __init__(
        self,
        feat: RBFFeatures,
        alpha_w: float,
        alpha_mu: float,
        alpha_sigma: float,
        alpha_r: float,
        lam_w: float,
        lam_theta: float,
    ):
        d = feat.n
        self.feat = feat
        self.w = np.zeros(d)  # value weights
        self.theta_mu = np.zeros(d)  # policy mean weights
        self.theta_sigma = np.zeros(d)  # policy log-std weights  -> sigma0 = 1
        self.R_bar = 0.0  # average-reward estimate

        self.alpha_w = alpha_w
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.alpha_r = alpha_r
        self.lam_w = lam_w
        self.lam_theta = lam_theta

        self.z_w = np.zeros(d)  # value-function trace
        self.z_mu = np.zeros(d)  # policy-mean trace  ) together these
        self.z_sigma = np.zeros(d)  # policy-std  trace  ) are z^theta

    # -- value --
    def value(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    # -- policy --
    def policy(self, x: np.ndarray):
        mu = float(self.theta_mu @ x)
        log_sigma = float(
            np.clip(self.theta_sigma @ x, -20.0, 2.0)
        )  # stability
        sigma = np.exp(log_sigma)
        return mu, sigma

    def act(self, x: np.ndarray):
        mu, sigma = self.policy(x)
        a = np.random.normal(mu, sigma)
        return a, mu, sigma

    def reset_traces(self):
        self.z_w[:] = 0.0
        self.z_mu[:] = 0.0
        self.z_sigma[:] = 0.0

    # -- one step of the algorithm in the box --
    def update(self, x, a, mu, sigma, R, x_next):
        v_s = self.value(x)
        v_next = self.value(x_next)

        # delta <- R - R_bar + v(S') - v(S)
        delta = R - self.R_bar + v_next - v_s

        # R_bar <- R_bar + alpha_R * delta
        self.R_bar += self.alpha_r * delta

        # score function grad ln pi(A|S, theta)
        grad_mu = (a - mu) / (sigma**2) * x
        grad_sigma = ((a - mu) ** 2 / (sigma**2) - 1.0) * x

        # traces:  z <- lambda * z + grad
        self.z_w = self.lam_w * self.z_w + x  # grad v_hat = x
        self.z_mu = self.lam_theta * self.z_mu + grad_mu
        self.z_sigma = self.lam_theta * self.z_sigma + grad_sigma

        # weight updates:  param <- param + alpha * delta * z
        self.w += self.alpha_w * delta * self.z_w
        self.theta_mu += self.alpha_mu * delta * self.z_mu
        self.theta_sigma += self.alpha_sigma * delta * self.z_sigma

        return delta


def train(
    n_steps: int = 500_000,
    seed: int = 0,
    log_every: int = 5_000,
    render: bool = False,
):
    np.random.seed(seed)
    feat = RBFFeatures(n_angle=12, n_vel=8, bandwidth=0.40)

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
