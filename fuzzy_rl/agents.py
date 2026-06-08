import numpy as np

from fuzzy_rl.features import RBFFeatures


class ActorCriticContinuingClip:
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
