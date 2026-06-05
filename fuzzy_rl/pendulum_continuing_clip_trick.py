import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fuzzy_rl.env import make_continuing_pendulum
from fuzzy_rl.features import RBFFeatures


class ActorCriticContinuing:
    def __init__(
        self,
        feat,
        alpha_mu,
        alpha_sigma,
        alpha_r,
        lam_w,
        lam_theta,
        m=2.0,
        l=2.0,
        eps=1e-2,
        use_bootstrap_denom=True,
    ):
        d = feat.n
        self.feat = feat
        self.w = np.zeros(d)
        self.theta_mu = np.zeros(d)
        self.theta_sigma = np.zeros(d)

        self.R_bar = 0.0  # reward rate (average reward)
        self.r_var = 0.0  # EMA of (R - R_bar)^2  ->  sigma_r^2
        self.t = 0  # step counter for EMA bias correction

        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.alpha_r = alpha_r
        self.lam_w = lam_w
        self.lam_theta = lam_theta

        self.m = m  # fractional divisor (m>=1 -> monotone, no overshoot)
        self.l = l  # clip multiplier (2 or 3 in practice)
        self.eps = eps  # denominator floor (caps the 1/D blow-up)
        self.use_bootstrap_denom = use_bootstrap_denom

        self.z_w = np.zeros(d)
        self.z_mu = np.zeros(d)
        self.z_sigma = np.zeros(d)

        # diagnostics
        self.n_floored = 0
        self.last_sigma_r = 1.0

    def value(self, x):
        return float(self.w @ x)

    def policy(self, x):
        mu = float(self.theta_mu @ x)
        log_sigma = float(np.clip(self.theta_sigma @ x, -20.0, 2.0))
        return mu, np.exp(log_sigma)

    def act(self, x):
        mu, sigma = self.policy(x)
        return np.random.normal(mu, sigma), mu, sigma

    def reset_traces(self):
        self.z_w[:] = 0.0
        self.z_mu[:] = 0.0
        self.z_sigma[:] = 0.0

    @staticmethod
    def _floor(d, eps):
        """Keep sign, floor magnitude at eps -> guards the 1/d singularity."""
        if abs(d) >= eps:
            return d
        return eps if d >= 0.0 else -eps

    def update(self, x, a, mu, sigma, R, x_next):
        self.t += 1
        v_s, v_next = self.value(x), self.value(x_next)
        delta = R - self.R_bar + v_next - v_s

        # --- reward rate and its dispersion (the clip scale sigma_r) ---
        self.R_bar += self.alpha_r * delta
        self.r_var += self.alpha_r * ((R - self.R_bar) ** 2 - self.r_var)
        bc = 1.0 - (1.0 - self.alpha_r) ** self.t  # EMA bias correction
        sigma_r = np.sqrt(max(self.r_var / bc, 0.0))
        sigma_r = sigma_r if sigma_r > 1e-8 else 1.0
        self.last_sigma_r = sigma_r

        # --- traces:  z <- lambda z + grad ---
        grad_mu = (a - mu) / sigma**2 * x
        grad_sigma = ((a - mu) ** 2 / sigma**2 - 1.0) * x
        self.z_w = self.lam_w * self.z_w + x
        self.z_mu = self.lam_theta * self.z_mu + grad_mu
        self.z_sigma = self.lam_theta * self.z_sigma + grad_sigma

        # === Trick 2: clip TD error to +/- l*sigma_r ===
        clipped = float(np.clip(delta, -self.l * sigma_r, self.l * sigma_r))

        # === Trick 1: fractional NLMS critic step ===
        #   bootstrap target moves with w -> z_w . (x - x')   [correct]
        #   ignore the moving target      -> z_w . x          [better conditioned]
        denom_raw = (
            self.z_w @ (x - x_next)
            if self.use_bootstrap_denom
            else self.z_w @ x
        )
        denom_raw = float(denom_raw)
        if abs(denom_raw) < self.eps:
            self.n_floored += 1
        denom = self._floor(denom_raw, self.eps)
        self.w += (
            clipped / (self.m * denom)
        ) * self.z_w  # w <- w + alpha_nlms * z_w

        # --- actor: ordinary PG step, clipped (robust) advantage ---
        self.theta_mu += self.alpha_mu * clipped * self.z_mu
        self.theta_sigma += self.alpha_sigma * clipped * self.z_sigma
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
        alpha_mu=0.02,
        alpha_sigma=0.01,
        alpha_r=0.01,  # reward-rate / variance EMA rate
        lam_w=0.80,
        lam_theta=0.80,
        m=5.0,  # fractional step  (m>=1 -> no overshoot)
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
