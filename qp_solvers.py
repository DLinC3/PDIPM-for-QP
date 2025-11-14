import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from typing import Tuple

class PDIPSolver():
    """
    Follows the implementation of the primal-dual interior point algorithm as found in
        "CVXGEN: a code generator for embedded convex optimization", J. Mattingley and S. Boyd (2012).
    """
    def __init__(self, max_qp_iter: int = 50, tol: float = 1e-15):
        """
        Attributes:
            max_qp_iter (int): maximum number of QP iterations
            tol (float): positive value for measuring QP residual convergence

            nx (int): number of primal variables
            n_eq (int): number of equality constraints
            n_ineq (int): number of inequality constraints

            x0 (jnp.array): primal solution x
            s0 (jnp.array): slack variable s
            z0 (jnp.array): dual variables for inequality constraints z
            y0 (jnp.array): dual variables for equality constraints y
        """
        self.max_qp_iter = max_qp_iter
        self.tol = tol

        self.nx = None
        self.n_eq = None
        self.n_ineq = None

        self.x0 = None
        self.s0 = None
        self.z0 = None
        self.y0 = None

        self._Q = None
        self._q = None
        self._A = None
        self._b = None
        self._G = None
        self._h = None

    def init_problem(self, Q: jax.Array, q: jax.Array, A: jax.Array, b: jax.Array, G: jax.Array, h: jax.Array) -> None:
        self._Q = 0.5*(Q + Q.T)
        self._q = q
        self.nx = q.size
        assert q.size**2 == Q.size

        self._A = A
        self._b = b
        self.n_eq = b.size
        assert A.shape == (self.n_eq, self.nx)

        self._G = G
        self._h = h
        self.n_ineq = h.size
        assert G.shape == (self.n_ineq, self.nx)

    def init_soln(self):
        """
        Follows initialization procedure from Section 5.2 in CVXGEN paper

        Returns:
            x0 (jnp.array): init. soln. for primary variable x
            s0 (jnp.array): init. soln. for slack variable s
            z0 (jnp.array): init. soln. for dual variables associated with inequality constraints
            y0 (jnp.array): init. soln. for dual variables associated with equality constraints
        """
        # raise NotImplementedError("PDIPSolver.init_soln has yet to be implemented")

        # Solve the coupled system, then set z := Gx - h and build s^(0), z^(0) via alpha_p, alpha_d rules.
        nx, ne, ni = self.nx, self.n_eq, self.n_ineq
        Q, A, G = self._Q, self._A, self._G
        q, b, h = self._q, self._b, self._h

        K = jnp.block([
            [Q,                       G.T,                      A.T],
            [G,             -jnp.eye(ni),           jnp.zeros((ni, ne))],
            [A,              jnp.zeros((ne, ni)),   jnp.zeros((ne, ne))]
        ])
        rhs = jnp.concatenate([-q, h, b])

        sol = jsp.linalg.solve(K, rhs, assume_a='gen')
        x = sol[:nx]
        # z_lin = sol[nx:nx+ni]  # not used further per the specified initialization
        y = sol[nx+ni:]

        # z := Gx - h  (note: s = h - Gx = -z)
        z = G @ x - h

        # alpha_p = inf{alpha | -z + alpha*1 >= 0} = max(z)
        alpha_p = jnp.max(z)
        s = jnp.where(alpha_p < 0.0, -z, -z + (1.0 + alpha_p) * jnp.ones_like(z))

        # alpha_d = inf{alpha | z + alpha*1 >= 0} = max(-z)
        alpha_d = jnp.max(-z)
        z = jnp.where(alpha_d < 0.0, z, z + (1.0 + alpha_d) * jnp.ones_like(z))

        self.x0, self.s0, self.z0, self.y0 = x, s, z, y
        return x, s, z, y


    def compute_residuals(self, xbar: jax.Array, sbar: jax.Array, zbar: jax.Array, ybar: jax.Array):
        """
        Implements the residuals associated with the right hand side of the KKT system.

        Parameters:
            xbar (jnp.array): Current primal variable x
            sbar (jnp.array): Current slack variable s
            zbar: (jnp.array): Current dual variable z for inequality constraints
            sbar: (jnp.array): Current dual variable s for equality constraints

        Returns:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
        """
        # raise NotImplementedError("PDIPSolver.compute_residuals has yet to be implemented")
        # r1 = -(Qx + q + A^T y + G^T z)
        r1 = -(self._Q @ xbar + self._q + self._A.T @ ybar + self._G.T @ zbar)
        # r2 = -diag(s) z
        r2 = -(jnp.diag(sbar) @ zbar)
        # r3 = -(Gx + s - h)
        r3 = -(self._G @ xbar + sbar - self._h)
        # r4 = -(Ax - b)
        r4 = -(self._A @ xbar - self._b)
        return r1, r2, r3, r4

    def compute_centering_plus_corrector(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array) -> None:
        """
	Computes the sigma and mu terms used for the centering-plus-corrector step from Section 5.2 in CVXGEN paper.
	Hint: you should be able to reuse self.compute_line_search here.

        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.

        Returns:
            mu (float): mu term for centering-corrector step
            sigma (float): sigma term for centering-corrector step
        """
        # raise NotImplementedError("PDIPSolver.compute_centering_plus_corrector has yet to be implemented")
        p = self.n_ineq

        # mu = (s^T z)/p
        mu = (s0.T @ z0) / p

        # alpha from simple backtracking along affine direction
        alpha_aff, _ = self.compute_line_search(s0, ds, z0, dz, tol=1e-16)

        # numerator = ((s + a ds)^T (z + a dz))^2 ? square or not
        num = (s0 + alpha_aff*ds).T @ (z0 + alpha_aff*dz)
        denom = s0.T @ z0
        # sigma = ( (num**2) / denom )**3
        sigma = (num/ denom )**3
        # # build RHS
        # u1 = jnp.zeros(self.nx)
        # u3 = jnp.zeros(self.n_ineq)
        # u4 = jnp.zeros(self.n_eq)
        # u2 = sigma * mu * jnp.ones(self.n_ineq) - (jnp.diag(ds) @ dz)
        # # solve KKT with current bars (s0,z0)
        # dx_cc, ds_cc, dz_cc, dy_cc = self.solve_kkt_system(u1, u2, u3, u4, s0, z0)
        return mu, sigma

    def compute_line_search(self, s0: jax.Array, ds: jax.Array, z0: jax.Array, dz: jax.Array, n_steps: int = 500, tol: float = 1e-15):
        """
	The line search procedure that ensures s0+alpha*ds >= 0 and z0+alpha*ds >= 0

        Parameters:
            s0 (jnp.array): current iterate for slack variable s.
            ds (jnp.array): descent direction for slack variable s.
            z0 (jnp.array): current iterate for dual variable z.
            dz (jnp.array): descent direction for dual variable s.
            n_steps (int): max number of iterations for line search.
            tol (float): positive number tolerance for passing line search (i.e., s+ds >= tol)

        Returns:
            alpha (float): step size value
            line_search_successful (bool): indicates if line search found valid solution
        """
        # raise NotImplementedError("PDIPSolver.compute_line_search has yet to be implemented")
        def max_step(v, dv):
            mask = dv < 0.0
            if jnp.any(mask):
                return jnp.min(-v[mask] / dv[mask])
            else:
                return jnp.inf

        step_s = max_step(s0, ds)
        step_z = max_step(z0, dz)
        alpha_sup = jnp.minimum(1.0, jnp.minimum(step_s, step_z))
        ok = jnp.isfinite(alpha_sup)
        return float(alpha_sup), bool(ok)

    def has_converged(self) -> bool:
        """
        Given current solution iterate (x0, s0, z0, y0), computes whether the
        current QP residuals have been sufficiently minimized.

        Returns:
            converged (bool): true if QP solver has minimized residuals.
        """
        # raise NotImplementedError("PDIPSolver.has_converged has yet to be implemented")
        r1, r2, r3, r4 = self.compute_residuals(self.x0, self.s0, self.z0, self.y0)
        n1 = jnp.linalg.norm(r1, 2)
        n2 = jnp.linalg.norm(r2, 2)
        n3 = jnp.linalg.norm(r3, 2)
        n4 = jnp.linalg.norm(r4, 2)
        res_ok = bool(jnp.max(jnp.array([n1, n2, n3, n4])) <= self.tol)

        p = self.n_ineq if self.n_ineq > 0 else 1
        gap = (self.s0 @ self.z0) / p
        gap_ok = bool(gap <= self.tol)
        return res_ok and gap_ok


    def solve_kkt_system(self, r1: jax.Array, r2: jax.Array, r3: jax.Array, r4: jax.Array, sbar: jax.Array, zbar: jax.Array):
        """
        Solves the KKT system at the current iteration using the residual values.
	Note that sbar and zbar is passed in as the KKT matrix depends on the current value of those variables.

        Parameters:
            r1 (jnp.array): Residual for KKT derivative
            r2 (jnp.array): Residual for complementary slackness
            r3 (jnp.array): Residual for inequality constraint
            r4 (jnp.array): Residual for equality constraint
            sbar (jnp.array): Current value for slack value s
            zbar (jnp.array): Current value for dual variable z

        Returns:
            dx (jnp.array): descent direction for primal x
            ds (jnp.array): descent direction for slack variable s
            dz (jnp.array): descent direction for dual variable z
            dy (jnp.array): descent direction for dual variable y
        """
        # raise NotImplementedError("PDIPSolver.solve_kkt_system has yet to be implemented")

        # Solve:
        #   [ Q  0   G^T  A^T ][dx]   = [u1]
        #   [ 0  Z    S    0  ][ds]     [u2]
        #   [ G  I    0    0  ][dz]     [u3]
        #   [ A  0    0    0  ][dy]     [u4]
        # The inputs u1..u4 are the current RHS
        nx, ni, ne = self.nx, self.n_ineq, self.n_eq
        Q, A, G = self._Q, self._A, self._G

        S = jnp.diag(sbar)
        Z = jnp.diag(zbar)

        K = jnp.block([
            [Q,                  jnp.zeros((nx, ni)),    G.T,                 A.T],
            [jnp.zeros((ni, nx)),         Z,             S,                   jnp.zeros((ni, ne))],
            [G,                  jnp.eye(ni),            jnp.zeros((ni, ni)), jnp.zeros((ni, ne))],
            [A,                  jnp.zeros((ne, ni)),    jnp.zeros((ne, ni)), jnp.zeros((ne, ne))]
        ])
        rhs = jnp.concatenate([r1, r2, r3, r4])

        sol = jsp.linalg.solve(K, rhs, assume_a='gen')
        dx = sol[:nx]
        ds = sol[nx:nx+ni]
        dz = sol[nx+ni:nx+2*ni]
        dy = sol[nx+2*ni:]
        return dx, ds, dz, dy

    def solve_qp(self, verbose: bool = False):
        """
        Solves the QP by:
        1. Initialize the primal, slack, and dual variables using self.init_soln
        2. Carry out max_qp_iter number of iterations of solving the KKT system
            2a. At each iteration, use self.compute_residuals and self.solve_kkt_system
            2b. Use self.compute_centering_plus_corrector for correcting dz and ds corrections
		Hint: you should be able to use self.solve_kkt_system for both the affine and centering-corrector step
            2c. Use self.compute_line_search to compute the step size
            2d. Check for convergence

        Returns
            costs (list): a list of the objective function across iterations
        """
        # raise NotImplementedError("PDIPSolver.solve_qp has yet to be implemented")

        # init
        x, s, z, y = self.init_soln()
        costs = [float(0.5 * (x @ (self._Q @ x)) + self._q @ x)]

        for k in range(1, self.max_qp_iter + 1):
            # 1) stopping criteria
            self.x0, self.s0, self.z0, self.y0 = x, s, z, y
            if self.has_converged():
                if verbose:
                    print(f"Converged at iter {k-1}")
                break

            # 2) affine-scaling directions
            r1, r2, r3, r4 = self.compute_residuals(x, s, z, y)
            dx_aff, ds_aff, dz_aff, dy_aff = self.solve_kkt_system(r1, r2, r3, r4, s, z)

            # 3) centering-plus-corrector directions
            mu, sigma = self.compute_centering_plus_corrector(s, ds_aff, z, dz_aff)
            u1_cc = jnp.zeros(self.nx)
            u3_cc = jnp.zeros(self.n_ineq)
            u4_cc = jnp.zeros(self.n_eq)
            u2_cc = sigma * mu * jnp.ones(self.n_ineq) - (jnp.diag(ds_aff) @ dz_aff)
            dx_cc, ds_cc, dz_cc, dy_cc = self.solve_kkt_system(u1_cc, u2_cc, u3_cc, u4_cc, s, z)

            # 4) combine
            dx = dx_aff + dx_cc
            ds = ds_aff + ds_cc
            dz = dz_aff + dz_cc
            dy = dy_aff + dy_cc

            # 4) step size (0.99 shrink)
            alpha_sup, ok = self.compute_line_search(s, ds, z, dz, tol=0.0)
            alpha = float(min(1.0, 0.99 * alpha_sup)) if ok else 1e-3

            # 5) update variables
            x = x + alpha * dx
            s = s + alpha * ds
            z = z + alpha * dz
            y = y + alpha * dy

            # cost
            cost = float(0.5 * (x @ (self._Q @ x)) + self._q @ x)
            costs.append(cost)

            if verbose:
                r1n, r2n, r3n, r4n = self.compute_residuals(x, s, z, y)
                res2 = float(jnp.linalg.norm(jnp.concatenate([r1n, r2n, r3n, r4n]), 2))
                gap = float(s @ z)
                mu_now = float((s @ z) / max(self.n_ineq, 1))
                r1n, r2n, r3n, r4n = self.compute_residuals(x, s, z, y)
                rdual_norm = float(jnp.linalg.norm(r1n, 2))
                rpri_vec   = jnp.concatenate([r3n, r4n])
                rpri_norm  = float(jnp.linalg.norm(rpri_vec, 2))
                r_feas     = float(jnp.sqrt(rdual_norm**2 + rpri_norm**2))

                print(
                    f"it={k:02d} "
                    f"alpha={alpha:.3e} "
                    f"cost={cost:.3e} "
                    f"mu={mu_now:.3e} "
                    f"gap={gap:.15e} "
                    f"r_feas={r_feas:.15e} "
                    f"res2={res2:.3e}"
                )
            self.x0, self.s0, self.z0, self.y0 = x, s, z, y

        return costs

