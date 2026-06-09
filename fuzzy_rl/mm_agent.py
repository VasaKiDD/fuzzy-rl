"""
Continuing market making with adverse selection (Cartea-Jaimungal style, MOs move
the mid).  Two agents on the same discretised environment:

  1. ActorCriticContinuingClip  -- average-reward linear AC with the two tricks
     (fractional NLMS critic step `m`, clipped TD `l`).
  2. OptimalMMAgent             -- closed-form DPE control: the ergodic HJB
     linearises (h = log(omega)/kappa) into the eigenproblem  A omega = (rho*kappa) omega.

Key modelling facts that shape the code:
  * The ansatz H = x + qS + h(q) makes the value linear in (x,S): sigma drops out
    and the optimal control depends ONLY on inventory q.  Hence the RL state is q
    (7 values) and one-hot features make the critic exactly tabular in h(q).
  * Per-step reward = realised marked-wealth change  d(X+qS) - phi q^2 dt.
    Summed, this telescopes to the continuing objective (no terminal alpha penalty).
  * The Brownian P&L  q*sigma*dW  is mean-zero and policy-irrelevant (sigma is absent
    from the optimal control), so we subtract it as a control variate: same optimum,
    far less reward variance.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# --------------------------------------------------------------------------- #
class MarketMakingAdverseSelectionEnv:
    def __init__(
        self,
        dt=0.01,
        kappa_p=100.0,
        kappa_m=100.0,
        lam_p=1.0,
        lam_m=1.0,
        q_max=3,
        q_min=-3,
        phi=0.01,
        sigma=0.01,
        S0=100.0,
        eps_p=0.005,
        eps_m=0.005,
        mark_to_market_cv=True,
        seed=None,
    ):
        self.dt = dt
        self.kappa_p, self.kappa_m = kappa_p, kappa_m
        self.lam_p, self.lam_m = lam_p, lam_m
        self.q_max, self.q_min = q_max, q_min
        self.phi, self.sigma, self.S0 = phi, sigma, S0
        self.eps_p, self.eps_m = eps_p, eps_m
        # Control variate: subtract the (mean-zero) inventory mark-to-market
        # q0 * (dS - E[dS]).  Action-independent -> optimum & rho preserved exactly,
        # leaves the actor the clean transaction P&L (delta - epsilon per fill).
        self.mark_to_market_cv = mark_to_market_cv
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.X, self.S, self.q, self.t = 0.0, self.S0, 0, 0
        return self.q, {}

    def step(self, action):
        delta_p, delta_m = float(action[0]), float(
            action[1]
        )  # half-spreads >= 0
        q0, V0, S_start = self.q, self.X + self.q * self.S, self.S

        # (1) Brownian mid move at current inventory q0
        dS_diff = self.sigma * np.sqrt(self.dt) * self.rng.standard_normal()
        self.S += dS_diff

        # (2) buy MO  -> may lift our SELL LO (q decreases), then jumps mid UP
        if self.rng.random() < self.lam_p * self.dt:
            S_pre = self.S
            if self.q > self.q_min and self.rng.random() < np.exp(
                -self.kappa_p * delta_p
            ):
                self.X += S_pre + delta_p  # sell at ask = S + delta_p
                self.q -= 1
            self.S += self.eps_p  # permanent impact (fill or not)

        # (3) sell MO -> may lift our BUY LO (q increases), then jumps mid DOWN
        if self.rng.random() < self.lam_m * self.dt:
            S_pre = self.S
            if self.q < self.q_max and self.rng.random() < np.exp(
                -self.kappa_m * delta_m
            ):
                self.X -= S_pre - delta_m  # buy at bid = S - delta_m
                self.q += 1
            self.S -= self.eps_m

        V1 = self.X + self.q * self.S
        reward = (V1 - V0) - self.phi * (q0**2) * self.dt
        if self.mark_to_market_cv:
            # subtract q0 * (dS - E[dS]); mean-zero -> optimum & rho unchanged,
            # leaves the clean transaction P&L  (delta - epsilon) per fill, 0 on a miss.
            dS_total = self.S - S_start
            mean_dS = (
                self.lam_p * self.eps_p - self.lam_m * self.eps_m
            ) * self.dt
            reward -= q0 * dS_total - q0 * mean_dS
        self.t += 1
        return (
            self.q,
            reward,
            False,
            False,
            {"q": self.q, "S": self.S, "X": self.X},
        )


# --------------------------------------------------------------------------- #
class InventoryFeatures:
    """One-hot over inventory -> critic/actor are tabular in q (exact for h(q))."""

    def __init__(self, q_min=-3, q_max=3):
        self.q_min, self.q_max = q_min, q_max
        self.n = q_max - q_min + 1

    def __call__(self, q):
        x = np.zeros(self.n)
        x[int(q) - self.q_min] = 1.0
        return x


def softplus(u):
    return np.logaddexp(0.0, u)  # log(1+e^u), numerically stable, > 0


# --------------------------------------------------------------------------- #
class ActorCriticContinuingClip:
    """2-D Gaussian policy over (u_p, u_m); depth = (1/kappa)*softplus(u) so the
    action sits at the natural 1/kappa scale (sigma_init=1 then explores fill
    probs ~ (0.1, 0.9)).  Critic uses the fractional-NLMS + clipped-TD tricks.
    """

    def __init__(
        self,
        feat,
        d_scale_p,
        d_scale_m,
        alpha_mu=0.05,
        alpha_sigma=0.02,
        alpha_r=0.01,
        lam_w=0.8,
        lam_theta=0.8,
        m=2.0,
        l=8.0,
        eps=1e-2,
        reward_scale=100.0,
        use_bootstrap_denom=False,
    ):
        d = feat.n
        self.feat = feat
        self.d_scale_p, self.d_scale_m = d_scale_p, d_scale_m
        self.w = np.zeros(d)
        self.th_mu = np.zeros((2, d))  # rows: [+ , -]
        self.th_sig = np.zeros((2, d))
        self.z_w = np.zeros(d)
        self.z_mu = np.zeros((2, d))
        self.z_sig = np.zeros((2, d))

        self.R_bar, self.r_var, self.t = 0.0, 0.0, 0
        self.alpha_mu, self.alpha_sigma, self.alpha_r = (
            alpha_mu,
            alpha_sigma,
            alpha_r,
        )
        self.lam_w, self.lam_theta = lam_w, lam_theta
        self.m, self.l, self.eps = m, l, eps
        self.reward_scale = reward_scale
        self.use_bootstrap_denom = use_bootstrap_denom
        self.n_floored, self.last_sigma_r = 0, 1.0

    def value(self, x):
        return float(self.w @ x)

    def _policy(self, x):
        mu = self.th_mu @ x  # (2,)
        log_sig = np.clip(self.th_sig @ x, -7.0, 2.0)
        return mu, np.exp(log_sig)

    def act(self, x):
        mu, sig = self._policy(x)
        u = np.random.normal(mu, sig)  # (2,)
        dp = self.d_scale_p * softplus(u[0])
        dm = self.d_scale_m * softplus(u[1])
        return (dp, dm), {"u": u, "mu": mu, "sig": sig}

    def greedy_depths(self, q):
        x = self.feat(q)
        mu, _ = self._policy(x)
        return self.d_scale_p * softplus(mu[0]), self.d_scale_m * softplus(
            mu[1]
        )

    @staticmethod
    def _floor(d, eps):
        if abs(d) >= eps:
            return d
        return eps if d >= 0.0 else -eps

    def update(self, x, pinfo, R, x_next):
        self.t += 1
        R = (
            self.reward_scale * R
        )  # scale P&L to O(1); optimum is scale-invariant
        v_s, v_next = self.value(x), self.value(x_next)
        delta = R - self.R_bar + v_next - v_s

        self.R_bar += self.alpha_r * delta
        self.r_var += self.alpha_r * ((R - self.R_bar) ** 2 - self.r_var)
        bc = 1.0 - (1.0 - self.alpha_r) ** self.t
        sigma_r = np.sqrt(max(self.r_var / bc, 0.0))
        sigma_r = sigma_r if sigma_r > 1e-12 else 1.0
        self.last_sigma_r = sigma_r

        clipped = float(np.clip(delta, -self.l * sigma_r, self.l * sigma_r))

        # --- critic: trace + fractional NLMS step ---
        self.z_w = self.lam_w * self.z_w + x
        denom_raw = float(
            self.z_w @ (x - x_next)
            if self.use_bootstrap_denom
            else self.z_w @ x
        )
        if abs(denom_raw) < self.eps:
            self.n_floored += 1
        denom = self._floor(denom_raw, self.eps)
        self.w += (clipped / (self.m * denom)) * self.z_w

        # --- actor: two Gaussian heads, score in u-space, clipped advantage ---
        u, mu, sig = pinfo["u"], pinfo["mu"], pinfo["sig"]
        grad_mu = ((u - mu) / sig**2)[:, None] * x[None, :]  # (2,d)
        grad_sig = (((u - mu) ** 2 / sig**2) - 1.0)[:, None] * x[None, :]
        self.z_mu = self.lam_theta * self.z_mu + grad_mu
        self.z_sig = self.lam_theta * self.z_sig + grad_sig
        self.th_mu += self.alpha_mu * clipped * self.z_mu
        self.th_sig += self.alpha_sigma * clipped * self.z_sig
        return delta


# --------------------------------------------------------------------------- #
class OptimalMMAgent:
    """Closed-form ergodic DPE control.  Ergodic HJB + ansatz H=x+qS+h(q) and the
    log transform h=log(omega)/kappa give the eigenproblem  A omega = (rho*kappa) omega.
    Perron eigenpair -> gain rho, relative value h(q), feedback depths from the FOC.
    """

    def __init__(self, env):
        e = env
        kappa = e.kappa_p  # symmetric kappa assumed for the transform
        qs = np.arange(e.q_min, e.q_max + 1)
        n = len(qs)
        lin = e.eps_p * e.lam_p - e.eps_m * e.lam_m
        lt_p = e.lam_p * np.exp(-1.0 - kappa * e.eps_p)
        lt_m = e.lam_m * np.exp(-1.0 - kappa * e.eps_m)

        A = np.zeros((n, n))
        for i, q in enumerate(qs):
            A[i, i] = q * kappa * lin - e.phi * kappa * q * q
            if q > e.q_min:
                A[i, i - 1] = lt_p  # lambda+ couples to q-1 (sell fill)
            if q < e.q_max:
                A[i, i + 1] = lt_m  # lambda- couples to q+1 (buy fill)

        vals, vecs = np.linalg.eig(A)
        k = int(np.argmax(vals.real))
        omega = np.abs(vecs[:, k].real)  # Perron eigenvector is single-signed
        self.mu_max = float(vals[k].real)
        self.rho = self.mu_max / kappa  # average reward per unit time
        h = np.log(omega) / kappa

        BIG = 1e6
        self.dp = np.full(n, BIG)  # sell depth (no quote at q_min)
        self.dm = np.full(n, BIG)  # buy  depth (no quote at q_max)
        for i, q in enumerate(qs):
            if q > e.q_min:
                self.dp[i] = max(1.0 / kappa + e.eps_p + h[i] - h[i - 1], 0.0)
            if q < e.q_max:
                self.dm[i] = max(1.0 / kappa + e.eps_m + h[i] - h[i + 1], 0.0)
        self.q_min, self.h, self.qs = e.q_min, h, qs

    def act(self, q):
        i = int(q) - self.q_min
        return self.dp[i], self.dm[i]


# --------------------------------------------------------------------------- #
def evaluate(env_kwargs, policy_fn, n_steps=200_000, seed=0):
    """Run a (q -> (dp,dm)) policy and return mean reward/step and inventory hist."""
    env = MarketMakingAdverseSelectionEnv(**env_kwargs)
    q, _ = env.reset(seed=seed)
    tot, hist = 0.0, np.zeros(env.q_max - env.q_min + 1)
    for _ in range(n_steps):
        q, R, _, _, _ = env.step(policy_fn(q))
        tot += R
        hist[int(q) - env.q_min] += 1
    return tot / n_steps, hist / n_steps


def train(n_steps=400_000, seed=0, log_every=10_000, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    np.random.seed(seed)
    env = MarketMakingAdverseSelectionEnv(seed=seed, **env_kwargs)
    feat = InventoryFeatures(env.q_min, env.q_max)
    agent = ActorCriticContinuingClip(
        feat, 1.0 / env.kappa_p, 1.0 / env.kappa_m
    )

    q, _ = env.reset(seed=seed)
    x = feat(q)
    rw, history = [], []
    for t in range(1, n_steps + 1):
        action, pinfo = agent.act(x)
        q_next, R, _, _, _ = env.step(action)
        x_next = feat(q_next)
        agent.update(x, pinfo, R, x_next)
        x = x_next
        rw.append(R)
        if t % log_every == 0:
            avg = float(np.mean(rw))
            rw.clear()
            history.append((t, avg))
            dp0, dm0 = agent.greedy_depths(0)
            print(
                f"step {t:>8d} | avg R/step = {avg:+.3e} | R_bar = {agent.R_bar:+.3e} "
                f"| sigma_r = {agent.last_sigma_r:.3e} | depths@q0 = ({dp0:.4f},{dm0:.4f})"
            )
    return agent, history, env


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    env_kwargs = dict(
        dt=0.01,
        kappa_p=100.0,
        kappa_m=100.0,
        lam_p=1.0,
        lam_m=1.0,
        q_max=3,
        q_min=-3,
        phi=0.01,
        sigma=0.01,
        S0=100.0,
        eps_p=0.005,
        eps_m=0.005,
        mark_to_market_cv=True,
    )

    # --- optimal benchmark (closed-form DPE) ---
    opt = OptimalMMAgent(MarketMakingAdverseSelectionEnv(**env_kwargs))
    print(
        f"[optimal]  mu_max = {opt.mu_max:.4f} | rho/time = {opt.rho:.4e} "
        f"| rho*dt/step = {opt.rho * env_kwargs['dt']:.4e}"
    )
    for i, q in enumerate(opt.qs):
        print(
            f"  q={q:+d}  delta+={opt.dp[i]:.4f}  delta-={opt.dm[i]:.4f}  h={opt.h[i]:+.4f}"
        )
    opt_avg, opt_hist = evaluate(
        env_kwargs, opt.act, n_steps=300_000, seed=123
    )
    print(f"[optimal]  empirical avg R/step = {opt_avg:+.4e}")

    # --- train AC ---
    agent, history, _ = train(
        n_steps=400_000, seed=18101990, log_every=10_000, env_kwargs=env_kwargs
    )
    ac_avg, ac_hist = evaluate(
        env_kwargs, lambda q: agent.greedy_depths(q), n_steps=300_000, seed=123
    )
    print(
        f"[AC greedy] empirical avg R/step = {ac_avg:+.4e}  "
        f"({100 * ac_avg / opt_avg:.1f}% of optimal)"
    )

    h = pd.DataFrame(history, columns=["step", "avg_reward"]).set_index(
        "step"
    )["avg_reward"]
    h.to_csv("results/mm_adverse_selection_ac.csv")
    ax = h.plot(label="AC (behaviour)")
    ax.axhline(opt_avg, ls="--", color="k", label="optimal (DPE)")
    ax.legend()
    plt.show()
