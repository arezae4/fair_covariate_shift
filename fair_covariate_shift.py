import numpy as np
from scipy.optimize import minimize
from math import isclose, sqrt


def _dot_intercept(w, X):
    c = 0
    if w.size == X.shape[1] + 1:
        c = w[-1]
        w = w[:-1]

    z = np.dot(X, w) + c
    return z


class eopp_fair_covariate_shift_logloss:
    def __init__(
        self,
        tol=1e-6,
        verbose=0,
        max_iter=10000,
        C=0.001,
        random_initialization=True,
        trg_grp_marginal_matching=True,
        trg_grg_marginal_matching_covariate_shift=False,
    ):
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.C = C
        self.random_start = random_initialization
        self.theta = None
        self.lb_prob = 1e-9
        self.trg_grp_marginal_matching = trg_grp_marginal_matching
        self.max_epoch = 5
        self.trg_group_estimator = None
        self.trg_grp_marginal_matching_covariate_shift = (
            trg_grg_marginal_matching_covariate_shift
        )
        self.lambdas = np.zeros((2,))

    def f_value(self, A):
        if self.trg_group_estimator(1) >= self.trg_group_estimator(0):
            f_val = np.where(
                A == 1,
                1,
                -self.trg_group_estimator(1) / self.trg_group_estimator(0),
            )
        else:
            f_val = np.where(
                A == 1,
                self.trg_group_estimator(0) / self.trg_group_estimator(1),
                -1,
            )
        return f_val

    def build_trg_grp_estimator(
        self, X_src, A_src, Y_src, src_ratio, X_trg, A_trg, trg_ratio
    ):
        h = eopp_fair_covariate_shift_logloss(
            tol=self.tol,
            max_iter=self.max_iter,
            C=self.C,
            random_initialization=False,
            verbose=False,
            trg_grp_marginal_matching=False,
        )
        h.trg_group_estimator = lambda a: 1  # dummy estimator, not used because mu = 0
        h.fit(
            X_src, Y_src, A_src, src_ratio, X_trg, A_trg, trg_ratio, mu_range=0
        )  # build a covaritae shift model with ignored fairness
        p = h.predict_proba(
            X_trg, A_trg, trg_ratio
        )  # A is ignored as attribute but is included in X
        p1 = np.dot(p, A_trg.astype("int")) / A_trg.shape[0]
        p0 = np.dot(p, 1 - A_trg.astype("int")) / A_trg.shape[0]
        if self.verbose >= 1:
            print("p1 : {:.4f}, p0 : {:.4f}".format(p1, p0))
        estimator = lambda a: p1 if (a == 1) else p0 if (a == 0) else 0
        return estimator

    def fit(
        self,
        X_src,
        Y_src,
        A_src,
        src_st_ratio,
        X_trg,
        A_trg,
        trg_st_ratio,
        mu_range=[-1, 1],
    ):

        if not self.trg_group_estimator:
            self.trg_group_estimator = self.build_trg_grp_estimator(
                X_src,
                A_src,
                Y_src,
                (
                    src_st_ratio
                    if self.trg_grp_marginal_matching_covariate_shift
                    else np.ones_like(src_st_ratio)
                ),
                X_trg,
                A_trg,
                (
                    trg_st_ratio
                    if self.trg_grp_marginal_matching_covariate_shift
                    else np.ones_like(trg_st_ratio)
                ),
            )

        def _fit_given_mu(
            X_src, Y_src, A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu
        ):
            X_src = np.hstack((X_src, np.ones((X_src.shape[0], 1))))
            X_trg = np.hstack((X_trg, np.ones((X_trg.shape[0], 1))))

            def init_theta(mu, X_src, Y_src, A_src, src_st_ratio):
                m = X_src.shape[1]
                theta = self.theta
                if theta is None:
                    if self.random_start:
                        theta = np.random.random_sample((m,)) - 0.5
                    else:
                        theta = np.zeros((m,))
                res = minimize(
                    compute_theta_loss_grad,
                    theta,
                    args=(self.lambdas, mu, X_src, Y_src, A_src, src_st_ratio),
                    method="L-BFGS-B",
                    jac=True,
                    tol=1e-12,
                    options={
                        "maxiter": self.max_iter,
                        "disp": self.verbose >= 3,
                        "gtol": self.tol,
                    },
                )
                theta = res.x
                return theta

            """
            This function computes loss and grad for passing to optimization function.
            Lambdas are concated to the returned grad
            """

            def compute_theta_loss_grad(
                theta, lambdas, mu, X_src, Y_src, A_src, src_st_ratio
            ):
                # The gradient is computed on src data
                p, q = self._compute_p_and_q(
                    theta, lambdas, mu, X_src, A_src, src_st_ratio
                )
                n = X_src.shape[0]
                z = _dot_intercept(theta, X_src)
                loss = 1 / src_st_ratio * (
                    -q * np.log(p)
                    - (1 - q) * np.log(1 - p)
                    + mu * p * q * self.f_value(A_src)
                ) + z * (q - Y_src)
                loss = np.mean(loss) + 0.5 * self.C * np.dot(theta, theta)

                grad = np.reshape(q - Y_src, (-1, 1)) * X_src  # todo no ratio

                grad = np.sum(grad, axis=0) / n + self.C * theta

                return loss, grad

            def compute_lambda_loss_grad(lambdas, theta, mu, X_trg, A_trg, st_ratio):
                if (
                    not self.trg_grp_marginal_matching
                ):  # or self.mu == 0: # inactive fairness
                    return 0, 0
                C = 0
                p, q = self._compute_p_and_q(theta, lambdas, mu, X_trg, A_trg, st_ratio)
                n = q.shape[0]
                g1 = np.dot(q, A_trg) / n - self.trg_group_estimator(1)
                g0 = np.dot(q, 1 - A_trg) / n - self.trg_group_estimator(0)
                loss = (
                    g1 * lambdas[0]
                    + g0 * lambdas[1]
                    + 0.5 * C * np.dot(lambdas, lambdas)
                )
                grad = np.array([g1, g0]).reshape((-1,))
                grad = grad + C * lambdas
                return loss, grad

            def find_theta_lambda_with_grad(
                mu, X_src, Y_src, A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio
            ):
                lambdas = self.lambdas
                theta = self.theta
                lambda_rate = 1  # learning rate
                theta_rate = 0.01
                max_itr = self.max_iter
                min_val_t = self.tol
                min_val_l = 1e-6

                S_g_t = np.ones_like(theta) * 1e-8  # prevent dividing by zero
                S_g_l = np.ones_like(lambdas) * 1e-8  # prevent dividing by zero
                l_0 = 0
                l_1 = (1 + sqrt(1 + 4 * l_0 ** 2)) / 2
                delta_1_l, delta_1_t = 0, 0
                t = 1
                while True:
                    t = t + 1
                    decay = sqrt(1000 / (1000 + t))
                    l_2 = (1 + sqrt(1 + 4 * l_1 ** 2)) / 2
                    l_3 = (1 - l_1) / l_2
                    _, G_l = compute_lambda_loss_grad(
                        lambdas, theta, mu, X_trg, A_trg, trg_st_ratio
                    )
                    _, G_t = compute_theta_loss_grad(
                        theta, lambdas, mu, X_src, Y_src, A_src, src_st_ratio
                    )
                    if self.verbose >= 2:
                        if t % 1000 == 0:
                            print(
                                "Lambda gnorm {:.7f}, Theta norm {:.7f}".format(
                                    np.linalg.norm(G_l), np.linalg.norm(G_t)
                                )
                            )
                    if (
                        np.linalg.norm(G_t) < min_val_t
                        and np.linalg.norm(G_l) < min_val_l
                    ):  # convergence threshold
                        if self.verbose >= 2:
                            print(
                                "-> GD epoch: converged. |grad_theta|:\t{:.9f}, |grad_lambda|:\t{:.9f}".format(
                                    np.linalg.norm(G_t), np.linalg.norm(G_l)
                                )
                            )
                        break
                    elif t > max_itr:
                        if self.verbose >= 2:
                            print(
                                "-> GD epoch: max iteration stopped. |grad_theta|:\t{:.9f}, |grad_lambda|:\t{:.9f}".format(
                                    np.linalg.norm(G_t), np.linalg.norm(G_l)
                                )
                            )
                        break
                    S_g_t = S_g_t + np.square(G_t)  # for adaptive gradient
                    S_g_l = S_g_l + np.square(G_l)  # for adaptive gradient
                    delta_2_l = lambdas - decay * lambda_rate * G_l / np.sqrt(
                        S_g_l
                    )  # adaptive gradient and Nesterov's Accelerated Gradient Descent
                    delta_2_t = theta - decay * theta_rate * G_t / np.sqrt(
                        S_g_t
                    )  # adaptive gradient and Nesterov's Accelerated Gradient Descent
                    lambdas = (1 - l_3) * delta_2_l + l_3 * delta_1_l
                    theta = (1 - l_3) * delta_2_t + l_3 * delta_1_t
                    delta_1_l = delta_2_l
                    delta_1_t = delta_2_t
                    l_1 = l_2
                return theta, lambdas, G_l, G_t

            self.theta = init_theta(mu, X_src, Y_src, A_src, src_st_ratio)

            epoch = 1
            while True:
                self.theta, self.lambdas, l_g, t_g = find_theta_lambda_with_grad(
                    mu, X_src, Y_src, A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio
                )
                if np.linalg.norm(l_g) < self.tol and np.linalg.norm(t_g) < self.tol:
                    if self.verbose >= 1:
                        print(
                            "Mu = {:.3f} - Epoch {:d}: converged, error\t{:.2f}, |grad_theta|:\t{:.7f}, |grad_lambda|:\t{:.7f}".format(
                                mu,
                                epoch,
                                1
                                - self._score(
                                    X_src,
                                    Y_src,
                                    A_src,
                                    src_st_ratio,
                                    mu,
                                    self.lambdas,
                                    self.theta,
                                ),
                                np.linalg.norm(t_g),
                                np.linalg.norm(l_g),
                            )
                        )
                    break
                elif epoch >= self.max_epoch:
                    if self.verbose >= 1:
                        print(
                            "Mu = {:.3f} - Epoch {:d} NOT converged, error\t{:.2f}, |grad_theta|:\t{:.7f}, |grad_lambda|\t{:.7f}".format(
                                mu,
                                epoch,
                                1
                                - self._score(
                                    X_src,
                                    Y_src,
                                    A_src,
                                    src_st_ratio,
                                    mu,
                                    self.lambdas,
                                    self.theta,
                                ),
                                np.linalg.norm(t_g),
                                np.linalg.norm(l_g),
                            )
                        )
                    break
                epoch += 1
            return self.theta, self.lambdas

        def q_violation_given_mu(mu):
            _fit_given_mu(
                X_src, Y_src, A_src, src_st_ratio, X_trg, A_trg, trg_st_ratio, mu
            )
            return self.q_fairness_violation(X_trg, A_trg, trg_st_ratio, mu)

        def _binary_search_mu(mu0, mu1):
            a, b = mu0, mu1
            fa = q_violation_given_mu(a)
            fb = q_violation_given_mu(b)
            if fa > fb:
                a, b = b, a
                fa, fb = fb, fa
            if isclose(fa, 0, abs_tol=1e-4):
                return a
            elif isclose(fb, 0, abs_tol=1e-4):
                return b
            elif fa * fb > 0:
                print(
                    "mu: [{:.3f}, {:.3f}],f(mu): [{:.4f}, {:.4f}]".format(a, b, fa, fb)
                )
                raise ValueError(
                    "Mu range boundary is both positive or negative. Try a wider range!."
                )
            else:
                self.zero_regions = [[a, b]]
                print(
                    "Binary search for zero violation in Mu ranges:", self.zero_regions
                )
                for r in self.zero_regions:
                    a, b = r[0], r[1]
                    while not isclose(a - b, 0, abs_tol=1e-4):
                        c = (a + b) / 2
                        fc = q_violation_given_mu(c)
                        print(
                            "Mu range: [{:.3f}, {:.3f}] => c = {:.4f}, q-violation = {:.4f}".format(
                                a, b, c, fc
                            )
                        )
                        if isclose(abs(fc), 0, abs_tol=1e-3):
                            return c
                        elif fc < 0:
                            a = c
                        elif fc > 0:
                            b = c

        # end _binary_search_mu()

        def find_valid_mu_range(mu_range, step=0.5):
            if np.isscalar(mu_range):
                return mu_range
            if len(mu_range) < 2:
                return mu_range[0]

            mu0 = mu_range[0]
            mu1 = mu_range[1]
            if mu1 < mu0:
                mu0, mu1 = mu1, mu0

            lo = q_violation_given_mu(mu0)
            hi = q_violation_given_mu(mu1)

            mu_lo = mu0
            for mu_ in np.arange(mu0 + step, mu1 + step, step):
                if self.verbose >= 1:
                    print("Grid search: mu range({:.2f},{:.2f})".format(mu_lo, mu_))
                v_ = q_violation_given_mu(mu_)
                if v_ * lo < 0:
                    if self.verbose >= 1:
                        print("Found zero point range.")
                    return [mu_lo, mu_]
                else:
                    lo = v_
                    mu_lo = mu_

            raise Exception(
                "Error: no Mu range found to cross zero violation. Try wider Mu range, or smaller grid step."
            )

        mu_range = find_valid_mu_range(mu_range, step=0.5)
        if np.isscalar(mu_range):
            self.mu = mu_range
        # elif len(mu_range) < 2:
        #    self.mu = mu_range[0]
        else:
            mu0 = mu_range[0]
            mu1 = mu_range[1]
            self.mu = _binary_search_mu(mu0, mu1)
            return self
        _fit_given_mu(
            X_src,
            Y_src,
            A_src,
            src_st_ratio,
            X_trg,
            A_trg,
            trg_st_ratio,
            self.mu,
        )
        return self

    def _compute_p_and_q(self, theta, lambdas, mu, X, A, st_ratio):
        z = _dot_intercept(theta, X)
        n = X.shape[0]

        def _solve_p_binary_search(mu, A, st_ratio, z, lambdas):
            eps = self.lb_prob
            a, b = np.zeros_like(z) + eps, np.ones_like(z) - eps
            f = self.f_value(A)
            mu_f = mu * f
            b[mu_f > 1] = 1 / mu_f[mu_f > 1]
            fun = (
                lambda x: -np.log(1 / (1 - x))
                + np.log(1 / x)
                + mu * x * f
                + st_ratio * z
                + lambdas[0] * A
                + lambdas[1] * (1 - A)
            )
            while np.any(abs(a - b) > self.tol):
                m = (a + b) / 2
                fc = fun(m)
                a = np.where(fc >= 0, m, a)
                b = np.where(fc <= 0, m, b)

            return (a + b) / 2

        p = _solve_p_binary_search(mu, A, st_ratio, z, lambdas)
        q = p / (1 - mu * self.f_value(A) * p + mu * self.f_value(A) * np.square(p))
        assert np.all(q >= 0)
        assert np.all(q <= 1)

        if not (np.all(q <= 1)):
            print(" Q <= 1 FAILED!")
        q = np.maximum(0, np.minimum(q, 1))
        return p, q

    def _sanitize_check_model(self):
        if self.theta is None:
            raise ValueError("Missing model parameters for feature matching (theta).")
        if self.lambdas is None:
            raise ValueError("Missing model parameters for group matching (lambdas).")
        if self.mu is None:
            raise ValueError("Missing model parameters for fairness penalty (mu).")

    def _q_fairness_violation(self, p, q, A):
        return (
            (np.sum((p * q)[A == 1]) / self.trg_group_estimator(1))
            - (np.sum((p * q)[A == 0]) / self.trg_group_estimator(0))
        ) / p.shape[0]

    def q_fairness_violation(self, X, A, st_ratio, mu=None):
        if mu is None:
            mu = self.mu
        p, q = self._compute_p_and_q(self.theta, self.lambdas, mu, X, A, st_ratio)
        return self._q_fairness_violation(p, q, A)

    def _score(self, X, Y, A, st_ratio, mu, lambdas, theta):
        p, _ = self._compute_p_and_q(theta, lambdas, mu, X, A, st_ratio)
        return 1 - np.mean(abs(np.round(p) - Y))

    def score(self, X, Y, A, st_ratio):
        self._sanitize_check_model()
        return self._score(X, Y, A, st_ratio, self.mu, self.lambdas, self.theta)

    def expected_error(self, X, Y, A, ratio):
        proba = self.predict_proba(X, A, ratio)
        return np.mean(np.where(Y == 1, 1 - proba, proba))

    def predict_proba(self, X, A, st_ratio):
        self._sanitize_check_model()
        p, _ = self._compute_p_and_q(self.theta, self.lambdas, self.mu, X, A, st_ratio)
        return p

    def predict(self, X, A, st_ratio):
        return np.round(self.predict_proba(X, A, st_ratio))

    def fairness_violation(self, X, Y, A, ratio):
        proba = self.predict_proba(X, A, ratio)
        return np.mean(proba[np.logical_and(Y == 1, A == 1)]) - np.mean(
            proba[np.logical_and(Y == 1, A == 0)]
        )
