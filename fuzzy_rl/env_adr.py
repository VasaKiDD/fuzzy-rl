"""
ADR arbitrage / lead-lag market-making model as a gym-style RL environment.

Economics (Cartea-Jaimungal-style, two cointegrated venues)
-----------------------------------------------------------
  * F = foreign mid (LEAD): martingale + permanent jumps from foreign order flow.
  * A = ADR mid   (LAG)  : error-correction toward F (dA has -kappa_A (A-F) dt)
                           + its own diffusion + permanent jumps from ADR flow.
  * Basis  Z := A - F  is a stationary OU process, half-life ln2 / kappa_A.

  Impact split (per venue k):  temporary b^k (walks the book, gates the fill),
  permanent beta^k = pi^k b^k  (moves the mid).  Order sizes eps ~ Exp(rho^k).
  A size-eps MO walks the book by b^k eps, so it lifts our LO at depth delta iff
  b^k eps >= delta  ->  P(fill) = P(eps >= delta/b^k) = e^{-kappa^k delta},
  kappa^k = rho^k / b^k.  Conditional on a fill eps is large => big permanent jump
  => adverse selection is *intrinsic and size-correlated* (no add-on needed).

Agent / control
---------------
  Each step the agent posts an ADR bid at A-delta^- and ask at A+delta^+ (pegged
  around the ADR mid for the next dt).  Action = (delta^+, delta^-) >= 0.
  The foreign leg is an *automatic* delta-hedge: a peg-at-touch order that fires
  only on the side that shrinks the net delta  Delta := q^A + q^F, with intensity
  lambda^F nu^F g(|Delta|/V^F),  g(x)=e^{-eta x}  (participation throttle).

Reward  (per-step increment of the performance functional H)
------------------------------------------------------------
  reward_t = d( X + q^A A + q^F F )  -  phi * Delta^2 * dt
  i.e. mark-to-market wealth change minus the running net-delta penalty.  Summed,
  this telescopes to the continuing analogue of
      H = E[ X_T + q^A(A-alpha^A q^A) + q^F(F-alpha^F q^F) - phi int Delta^2 ].
  The mean-zero Brownian inventory P&L is removed as a control variate (optimum
  unchanged) so the convergence/spread signal is not buried in mark-to-market noise.
"""

import numpy as np

try:  # use real gym if present, else shim
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
except Exception:  # minimal gym-compatible fallback

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = self.low.shape if shape is None else tuple(shape)
            self.dtype = dtype

        def sample(self):
            lo = np.where(np.isfinite(self.low), self.low, -1.0)
            hi = np.where(np.isfinite(self.high), self.high, 1.0)
            return (lo + (hi - lo) * np.random.rand(*self.shape)).astype(
                self.dtype
            )

        def contains(self, x):
            x = np.asarray(x)
            return (
                x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )

    class _Spaces:
        Box = _Box

    _spaces = _Spaces()

    class _Base:  # tiny Env stand-in
        metadata = {}

        def reset(self, *a, **k):
            raise NotImplementedError

        def step(self, a):
            raise NotImplementedError


# --------------------------------------------------------------------------- #
class ADRArbitrageEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=1.0,  # seconds per step
        steps_per_day=23400,  # 6.5h * 3600
        # --- diffusion ---
        sigma0_A=0.010,
        sigma0_F=0.010,
        rho_AF=0.90,
        # --- error correction (lag speed) ---
        kappa_A=0.03,  # half-life ln2/kappa_A ~ 23 s
        # --- order flow / impact (per venue) ---
        lam0_A=2.0,
        lam0_F=2.0,  # MO arrival rate per side at nu=1
        rho_A=1.0,
        rho_F=1.0,  # size ~ Exp(rho): mean size 1/rho
        kappa_fill_A=100.0,
        kappa_fill_F=100.0,  # fill decay = rho/b
        pi_A=0.30,
        pi_F=0.30,  # permanent/temporary impact ratio
        # --- activity factors (CIR) ---
        nu_bar=1.0,
        theta_nu=0.01,
        xi_nu=0.10,
        Vbar_A=100.0,
        Vbar_F=100.0,
        # --- foreign hedge ---
        xi_F=0.010,
        eta=1.0,  # touch half-spread, throttle steepness
        # --- MM order size / penalty ---
        L0=1.0,
        phi=0.01,
        S0=100.0,
        subtract_diffusion_pnl=True,
        max_inventory=500.0,  # safety clamp on |q^A|
        seed=None,
    ):
        self.dt, self.steps_per_day = dt, steps_per_day
        self.sigma0_A, self.sigma0_F, self.rho_AF = sigma0_A, sigma0_F, rho_AF
        self.kappa_A = kappa_A
        self.lam0_A, self.lam0_F = lam0_A, lam0_F
        self.rho_A, self.rho_F = rho_A, rho_F
        self.kf_A, self.kf_F = kappa_fill_A, kappa_fill_F
        self.b_A, self.b_F = (
            rho_A / kappa_fill_A,
            rho_F / kappa_fill_F,
        )  # temp impact
        self.beta_A, self.beta_F = (
            pi_A * self.b_A,
            pi_F * self.b_F,
        )  # perm impact
        self.nu_bar, self.theta_nu, self.xi_nu = nu_bar, theta_nu, xi_nu
        self.Vbar_A, self.Vbar_F = Vbar_A, Vbar_F
        self.xi_F, self.eta = xi_F, eta
        self.xi_F_eff = (
            xi_F + pi_F / kappa_fill_F
        )  # effective foreign half-spread
        self.L0, self.phi, self.S0 = L0, phi, S0
        self.subtract_diffusion_pnl = subtract_diffusion_pnl
        self.max_inventory = max_inventory
        self.rng = np.random.default_rng(seed)

        big = np.float32(1e6)
        self.action_space = _spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([0.5, 0.5]),
            dtype=np.float32,
        )
        # obs = [Z, q^A, q^F, nu^A, nu^F]
        self.observation_space = _spaces.Box(
            low=np.array([-big, -big, -big, 0.0, 0.0]),
            high=np.array([big, big, big, big, big]),
            dtype=np.float32,
        )

    # ---- helpers ------------------------------------------------------------
    def _obs(self):
        return np.array(
            [self.A - self.F, self.q_A, self.q_F, self.nu_A, self.nu_F],
            dtype=np.float32,
        )

    def _cir_step(self, nu):
        z = self.rng.standard_normal()
        nu = (
            nu
            + self.theta_nu * (self.nu_bar - nu) * self.dt
            + self.xi_nu * np.sqrt(max(nu, 0.0) * self.dt) * z
        )
        return max(nu, 1e-8)  # full truncation

    # ---- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.X = 0.0
        self.F = self.S0
        self.A = self.S0
        self.q_A = 0.0
        self.q_F = 0.0
        self.nu_A = self.nu_bar
        self.nu_F = self.nu_bar
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        dp, dm = float(action[0]), float(action[1])  # ask / bid depths >= 0
        dp, dm = max(dp, 0.0), max(dm, 0.0)
        qA0, qF0 = self.q_A, self.q_F
        Delta0 = qA0 + qF0
        M0 = self.X + qA0 * self.A + qF0 * self.F

        # (0) activity factors -> vols, intensities, volumes
        self.nu_A, self.nu_F = self._cir_step(self.nu_A), self._cir_step(
            self.nu_F
        )
        sigA, sigF = self.sigma0_A * np.sqrt(
            self.nu_A
        ), self.sigma0_F * np.sqrt(self.nu_F)
        lamA, lamF = self.lam0_A * self.nu_A, self.lam0_F * self.nu_F
        V_F = self.Vbar_F * self.nu_F
        sdt = np.sqrt(self.dt)

        # (1) correlated diffusion + ADR error correction toward F
        zF = self.rng.standard_normal()
        zA = (
            self.rho_AF * zF
            + np.sqrt(1.0 - self.rho_AF**2) * self.rng.standard_normal()
        )
        dF_diff = sigF * sdt * zF
        dA_diff = sigA * sdt * zA
        self.F += dF_diff
        self.A += -self.kappa_A * (self.A - self.F) * self.dt + dA_diff

        # (2) foreign order flow -> permanent jumps in F (Poisson-many per interval)
        n_Fp = self.rng.poisson(lamF * self.dt)
        n_Fm = self.rng.poisson(lamF * self.dt)
        if n_Fp:
            self.F += (
                self.beta_F
                * self.rng.exponential(1.0 / self.rho_F, n_Fp).sum()
            )
        if n_Fm:
            self.F -= (
                self.beta_F
                * self.rng.exponential(1.0 / self.rho_F, n_Fm).sum()
            )

        # (3) ADR order flow -> permanent jump in A and possible MM fill, per MO.
        #     Quote is pegged to the (moving) mid: fill iff b*eps >= delta.
        for _ in range(
            self.rng.poisson(lamA * self.dt)
        ):  # buy MOs lift our ASK
            A_pre = self.A
            eps = self.rng.exponential(1.0 / self.rho_A)
            self.A += self.beta_A * eps  # permanent up-move
            if self.b_A * eps >= dp and self.q_A > -self.max_inventory:
                self.X += self.L0 * (A_pre + dp)  # we sell L0 @ ask
                self.q_A -= self.L0
        for _ in range(
            self.rng.poisson(lamA * self.dt)
        ):  # sell MOs hit our BID
            A_pre = self.A
            eps = self.rng.exponential(1.0 / self.rho_A)
            self.A -= self.beta_A * eps  # permanent down-move
            if self.b_A * eps >= dm and self.q_A < self.max_inventory:
                self.X -= self.L0 * (A_pre - dm)  # we buy L0 @ bid
                self.q_A += self.L0

        # (4) automatic foreign delta-hedge: shrink |Delta| = |q^A + q^F| toward 0
        Delta = self.q_A + self.q_F
        if Delta != 0.0:
            g = np.exp(-self.eta * abs(Delta) / max(V_F, 1e-8))
            n_h = min(
                self.rng.poisson(lamF * g * self.dt), int(np.floor(abs(Delta)))
            )
            if n_h > 0:
                if Delta > 0:  # sell foreign at touch
                    self.X += n_h * (self.F + self.xi_F_eff)
                    self.q_F -= n_h
                else:  # buy foreign at touch
                    self.X -= n_h * (self.F - self.xi_F_eff)
                    self.q_F += n_h

        # (5) reward = d(marked wealth) - phi Delta^2 dt  [- mean-zero diffusion P&L]
        M1 = self.X + self.q_A * self.A + self.q_F * self.F
        reward = (M1 - M0) - self.phi * (Delta0**2) * self.dt
        if self.subtract_diffusion_pnl:
            reward -= qA0 * dA_diff + qF0 * dF_diff  # control variate

        self.t += 1
        info = {
            "A": self.A,
            "F": self.F,
            "Z": self.A - self.F,
            "Delta": Delta,
            "q_A": self.q_A,
            "q_F": self.q_F,
            "X": self.X,
            "wealth": M1,
        }
        return self._obs(), reward, False, False, info


# --------------------------------------------------------------------------- #
def analytic_targets(env):
    """Closed-form daily vol and stationary basis stats from the parameters."""
    T = env.steps_per_day * env.dt
    var_rate = lambda s0, lam, beta, rho: s0**2 + 4.0 * lam * (beta / rho) ** 2
    #   diffusion var-rate s0^2*nu_bar  +  jump var-rate 2 sides * lam * beta^2 * E[eps^2],
    #   E[eps^2]=2/rho^2  ->  4*lam*(beta/rho)^2.  (nu_bar=1)
    vrF = (
        var_rate(env.sigma0_F, env.lam0_F, env.beta_F, env.rho_F) * env.nu_bar
    )
    vrA = (
        var_rate(env.sigma0_A, env.lam0_A, env.beta_A, env.rho_A) * env.nu_bar
    )
    daily_vol_F = np.sqrt(vrF * T)
    daily_vol_A = np.sqrt(vrA * T)  # A tracks F => ~ same daily vol
    # basis innovation rate: diffusions partly cancel (rho_AF), jumps independent add
    diff_rate = (
        env.sigma0_A**2 * env.nu_bar
        + env.sigma0_F**2 * env.nu_bar
        - 2 * env.rho_AF * env.sigma0_A * env.sigma0_F * env.nu_bar
    )
    jump_rate = (
        4 * env.lam0_A * (env.beta_A / env.rho_A) ** 2 * env.nu_bar
        + 4 * env.lam0_F * (env.beta_F / env.rho_F) ** 2 * env.nu_bar
    )
    var_Z = (diff_rate + jump_rate) / (2.0 * env.kappa_A)
    return {
        "daily_vol_F_pct": 100 * daily_vol_F / env.S0,
        "daily_vol_A_pct": 100 * daily_vol_A / env.S0,
        "basis_std": np.sqrt(var_Z),
        "basis_std_bps": 1e4 * np.sqrt(var_Z) / env.S0,
        "basis_halflife_s": np.log(2)
        / env.kappa_A
        * env.dt
        / env.dt,  # = ln2/kappa_A (per s)
    }


def diagnose(env_kwargs, n_days=10, depth=0.05, seed=0):
    """Run passively (fixed wide depths) and measure realised daily vol & basis OU fit."""
    env = ADRArbitrageEnv(**env_kwargs)
    env.reset(seed=seed)
    n = n_days * env.steps_per_day
    A = np.empty(n)
    F = np.empty(n)
    act = np.array([depth, depth])
    for i in range(n):
        _, _, _, _, info = env.step(act)
        A[i], F[i] = info["A"], info["F"]
    spd = env.steps_per_day
    # daily vol from per-step realised variance scaled by sqrt(steps/day) (robust);
    # prices here are pure exogenous mids (MM fills/hedge do not move A,F).
    dvolA = np.std(np.diff(A)) * np.sqrt(spd) / env.S0 * 100
    dvolF = np.std(np.diff(F)) * np.sqrt(spd) / env.S0 * 100
    Z = A - F
    # OU half-life via AR(1):  Z_{t+1} = c + a Z_t ;  kappa = -ln(a)/dt
    z0, z1 = Z[:-1], Z[1:]
    a = np.polyfit(z0, z1, 1)[0]
    kappa_hat = -np.log(max(a, 1e-9)) / env.dt
    return (
        {
            "daily_vol_F_pct": dvolF,
            "daily_vol_A_pct": dvolA,
            "basis_std": np.std(Z),
            "basis_std_bps": 1e4 * np.std(Z) / env.S0,
            "basis_halflife_s": np.log(2) / kappa_hat,
            "basis_mean": np.mean(Z),
        },
        A,
        F,
    )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import pandas as pd
    from matplotlib import pyplot as plt

    env_kwargs = dict(
        dt=1.0,
        steps_per_day=23400,
        sigma0_A=0.010,
        sigma0_F=0.010,
        rho_AF=0.90,
        kappa_A=0.03,
        lam0_A=2.0,
        lam0_F=2.0,
        rho_A=1.0,
        rho_F=1.0,
        kappa_fill_A=100.0,
        kappa_fill_F=100.0,
        pi_A=0.30,
        pi_F=0.30,
        nu_bar=1.0,
        theta_nu=0.01,
        xi_nu=0.10,
        phi=0.01,
        S0=100.0,
    )

    env = ADRArbitrageEnv(**env_kwargs)
    print("=== analytic targets ===")
    for k, v in analytic_targets(env).items():
        print(f"  {k:22s} {v:10.4f}")

    print("=== simulated (10 days, passive) ===")
    stats, A, F = diagnose(env_kwargs, n_days=10, depth=0.05, seed=1)
    for k, v in stats.items():
        print(f"  {k:22s} {v:10.4f}")

    one = env.steps_per_day
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(A[:one], label="A (ADR, lag)")
    ax[0].plot(F[:one], label="F (foreign, lead)")
    ax[0].legend()
    ax[0].set_ylabel("price")
    ax[1].plot((A - F)[:one], color="purple")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("Z = A - F")
    ax[1].set_xlabel("step (s)")
    plt.tight_layout()
    plt.savefig("results/adr_calibration.png", dpi=110)
    print("saved results/adr_calibration.png")
