import numpy as np

from fuzzy_rl.features import RBFFeaturesPendulum


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
        """Actor-critic for the continuing (no episode resets) setting with two
        variance-reduction tricks: fractional NLMS critic updates (Trick 1) and
        TD-error clipping scaled by reward standard deviation (Trick 2).

        Args:
            feat: Feature extractor (e.g. RBFFeaturesPendulum); its `.n` attribute
                gives the feature-vector dimension used to size all weight vectors.
            alpha_mu: Step size for the policy mean weights (theta_mu).
            alpha_sigma: Step size for the policy log-std weights (theta_sigma).
            alpha_r: EMA rate for the reward-rate estimate R_bar and its variance
                r_var; also controls the bias-correction denominator.
            lam_w: Eligibility-trace decay for the critic (z_w). 0 = TD(0), 1 = MC.
            lam_theta: Eligibility-trace decay for both actor traces (z_mu, z_sigma).
            m: Fractional divisor in the NLMS critic step (w += delta / (m * denom) * z_w).
                m >= 1 ensures the update is a contraction and prevents overshoot.
            l: Clip multiplier; the TD error is clipped to ±l * sigma_r before
                any weight update. Typical values: 2–3.
            eps: Floor on the NLMS denominator magnitude to prevent division by near-zero.
            use_bootstrap_denom: If True the NLMS denominator uses z_w · (x − x′),
                which accounts for the moving bootstrap target. If False it uses z_w · x,
                which is better-conditioned but ignores the bootstrap correction.
        """
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
        feat: RBFFeaturesPendulum,
        alpha_w: float,
        alpha_mu: float,
        alpha_sigma: float,
        alpha_r: float,
        lam_w: float,
        lam_theta: float,
    ):
        """Vanilla differential actor-critic (continuing tasks, no tricks).

        Args:
            feat: Feature extractor; `.n` gives the feature-vector dimension.
            alpha_w: Step size for the critic (value) weights w.
            alpha_mu: Step size for the policy mean weights theta_mu.
            alpha_sigma: Step size for the policy log-std weights theta_sigma.
            alpha_r: EMA rate for the average-reward estimate R_bar.
            lam_w: Eligibility-trace decay for the critic trace z_w.
            lam_theta: Eligibility-trace decay for the actor traces z_mu and z_sigma.
        """
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


class ActionValueContinuingClip:
    def __init__(
        self,
        feat,
        n_actions=9,
        a_low=-2.0,
        a_high=2.0,
        alpha_r=0.01,
        lam=0.8,
        m=2.0,
        l=2.5,
        eps=1e-2,
        epsilon=0.1,
        use_bootstrap_denom=False,
        target_mode="q",
    ):
        """Differential action-value agent (continuing tasks) with NLMS and
        TD-error clipping.  Actions are a finite grid; the policy is ε-greedy.

        Args:
            feat: Feature extractor; `.n` gives the feature-vector dimension.
            n_actions: Number of discrete actions, evenly spaced over [a_low, a_high].
            a_low: Lower bound of the action grid.
            a_high: Upper bound of the action grid.
            alpha_r: EMA rate for the reward-rate estimate R_bar and its variance r_var.
            lam: Eligibility-trace decay λ shared by all action traces.
            m: Fractional divisor in the NLMS update (W += delta / (m * denom) * Z).
                m >= 1 prevents overshoot.
            l: Clip multiplier; the TD error is clipped to ±l * sigma_r.
            eps: Floor on the NLMS denominator to guard against division by near-zero.
            epsilon: ε-greedy exploration probability.
            use_bootstrap_denom: If True the NLMS denominator uses Z[j]·x − Z[j*]·x′
                (bootstrap-corrected); if False uses Z[j]·x only.
            target_mode: "q" for Q-learning (greedy max over next-state actions with
                Watkins trace cutting) or "sarsa" for on-policy Sarsa(λ).
        """
        self.feat = feat
        d = feat.n
        self.actions = np.linspace(
            a_low, a_high, n_actions
        )  # includes 0 if n odd
        self.N = n_actions
        self.W = np.zeros((self.N, d))  # one value-weight block per action
        self.Z = np.zeros((self.N, d))  # eligibility trace (same shape)

        self.R_bar = 0.0  # reward rate
        self.r_var = 0.0  # EMA of (R - R_bar)^2 -> sigma_r^2
        self.t = 0

        self.alpha_r = alpha_r  # ONLY surviving step size (scalar EMAs)
        self.lam = lam
        self.m = m
        self.l = l
        self.eps = eps
        self.epsilon = epsilon
        self.use_bootstrap_denom = use_bootstrap_denom
        self.target_mode = (
            target_mode  # "q" (max target) or "sarsa" (on-policy)
        )

        self.n_floored = 0
        self.last_sigma_r = 1.0

    def q_all(self, x):
        return self.W @ x  # length-N action-value vector

    def act(self, x):
        """eps-greedy. Returns (action_index, was_greedy)."""
        greedy_j = int(np.argmax(self.q_all(x)))
        if np.random.rand() < self.epsilon:
            j = np.random.randint(self.N)
            return j, (j == greedy_j)
        return greedy_j, True

    def reset_traces(self):
        self.Z[:] = 0.0

    @staticmethod
    def _floor(d, eps):
        if abs(d) >= eps:
            return d
        return eps if d >= 0.0 else -eps

    def update(self, x, j, R, x_next, j_next, greedy_action):
        """One differential TD step.
        j             : action index taken at S
        j_next        : action index taken at S'  (Sarsa target only)
        greedy_action : was the action at S greedy? (Watkins trace cut)
        """
        self.t += 1
        q_sa = float(self.W[j] @ x)
        q_next_all = self.W @ x_next
        j_star = (
            int(np.argmax(q_next_all)) if self.target_mode == "q" else j_next
        )
        q_next = float(q_next_all[j_star])

        delta = R - self.R_bar + q_next - q_sa

        # --- reward rate + dispersion (the clip scale) ---
        self.R_bar += self.alpha_r * delta
        self.r_var += self.alpha_r * ((R - self.R_bar) ** 2 - self.r_var)
        bc = 1.0 - (1.0 - self.alpha_r) ** self.t
        sigma_r = np.sqrt(max(self.r_var / bc, 0.0))
        sigma_r = sigma_r if sigma_r > 1e-8 else 1.0
        self.last_sigma_r = sigma_r

        # --- trace:  Z <- lam Z ; Z[j] += phi(S)  (grad of q(S,A) wrt W) ---
        self.Z *= self.lam
        self.Z[j] += x

        # === Trick 2: clip TD error ===
        clipped = float(np.clip(delta, -self.l * sigma_r, self.l * sigma_r))

        # === Trick 1: fractional NLMS step ===
        if self.use_bootstrap_denom:
            denom_raw = float(self.Z[j] @ x - self.Z[j_star] @ x_next)
        else:
            denom_raw = float(self.Z[j] @ x)
        if abs(denom_raw) < self.eps:
            self.n_floored += 1
        denom = self._floor(denom_raw, self.eps)
        self.W += (clipped / (self.m * denom)) * self.Z

        # --- Watkins Q(lambda): cut traces after a non-greedy action ---
        if self.target_mode == "q" and not greedy_action:
            self.Z[:] = 0.0

        return delta
