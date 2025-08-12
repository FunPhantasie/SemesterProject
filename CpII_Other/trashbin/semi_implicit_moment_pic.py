import numpy as np
from scipy.constants import c, epsilon_0, e, m_e
from scipy.sparse.linalg import gmres

class SemiImplicitPIC1D:
    def __init__(self, Nx=64, Lx=1.0, Np=1000, dt=0.05, theta=0.5):
        self.Nx = Nx
        self.Lx = Lx
        self.dx = Lx / Nx
        self.Np = Np
        self.dt = dt
        self.theta = theta
        self.x_grid = np.linspace(0, Lx, Nx, endpoint=False)
        self.t=0
        xp1 = 2 * Lx / Np * np.arange(Np // 2)
        xp2 = 2 * Lx / Np * np.arange(Np // 2)
        vp1 = 1 + 0.1 * np.sin(2 * np.pi / Lx * xp1)
        vp2 = -1 - 0.1 * np.sin(2 * np.pi / Lx * xp2)
        self.xp = np.concatenate([xp1, xp2])
        self.vp = np.zeros((3, Np))
        self.vp[0, :] = np.concatenate([vp1, vp2])
        self.qp = -e * np.ones(Np)
        self.mp = m_e * np.ones(Np)
        self.beta = (self.qp * dt / self.mp)[0]
        self.E = np.zeros((3, Nx))
        self.B = np.zeros((3, Nx))
        self.B[2] = 0.01

    def interpolate_field(self, xp, F):
        Fp = np.zeros((3, len(xp)))
        for p in range(len(xp)):
            i = int(xp[p] / self.dx) % self.Nx
            w = (xp[p] - i * self.dx) / self.dx
            Fp[:, p] = (1 - w) * F[:, i % self.Nx] + w * F[:, (i + 1) % self.Nx]
        return Fp

    def compute_alpha(self, Bp):
        alpha = np.zeros((3, 3, Bp.shape[1]))
        Bmag2 = np.sum(Bp**2, axis=0)
        denom = 1 + (self.beta**2) * Bmag2
        for p in range(Bp.shape[1]):
            B = Bp[:, p]
            I = np.identity(3)
            bx = np.array([[0, -B[2], B[1]], [B[2], 0, -B[0]], [-B[1], B[0], 0]])
            outer = np.outer(B, B)
            alpha[:, :, p] = (I - self.beta * bx + self.beta**2 * outer) / denom[p]
        return alpha

    def step(self):
        Ep = self.interpolate_field(self.xp, self.E)
        Bp = self.interpolate_field(self.xp, self.B)
        alpha = self.compute_alpha(Bp)
        v_hat = np.einsum("ijk,jk->ik", alpha, self.vp)

        rho = np.zeros(self.Nx)
        for p in range(self.Np):
            i = int(self.xp[p] / self.dx) % self.Nx
            w = (self.xp[p] - i * self.dx) / self.dx
            rho[i % self.Nx] += self.qp[p] * (1 - w)
            rho[(i + 1) % self.Nx] += self.qp[p] * w
        rho /= self.dx

        rhs = np.zeros(3 * self.Nx)
        A = np.eye(3 * self.Nx)
        E_flat, _ = gmres(A, rhs)
        E_theta = E_flat.reshape(3, self.Nx)
        E_theta_p = self.interpolate_field(self.xp, E_theta)
        v_new = v_hat + self.beta * np.einsum("ijk,jk->ik", alpha, E_theta_p)
        self.xp = (self.xp + self.dt * v_new[0, :]) % self.Lx
        self.vp = v_new
        return self.xp.copy()


    def compute_moment_sources(self, alpha, v_hat):
        J_hat = np.zeros((3, self.Nx))
        Pi_hat = np.zeros((3, 3, self.Nx))
        rho = np.zeros(self.Nx)

        for p in range(self.Np):
            i = int(self.xp[p] / self.dx) % self.Nx
            w = (self.xp[p] - i * self.dx) / self.dx

            for d in range(3):
                J_hat[d, i % self.Nx] += self.qp[p] * v_hat[d, p] * (1 - w)
                J_hat[d, (i + 1) % self.Nx] += self.qp[p] * v_hat[d, p] * w
                for e in range(3):
                    Pi_hat[d, e, i % self.Nx] += self.qp[p] * v_hat[d, p] * v_hat[e, p] * (1 - w)
                    Pi_hat[d, e, (i + 1) % self.Nx] += self.qp[p] * v_hat[d, p] * v_hat[e, p] * w

            rho[i % self.Nx] += self.qp[p] * (1 - w)
            rho[(i + 1) % self.Nx] += self.qp[p] * w

        return J_hat / self.dx, Pi_hat / self.dx, rho / self.dx

    def build_system_matrix_and_rhs(self, mu, J_hat, Pi_hat):
        A = np.zeros((3 * self.Nx, 3 * self.Nx))
        rhs = np.zeros(3 * self.Nx)

        for i in range(self.Nx):
            for d in range(3):
                row = 3 * i + d
                rhs[row] = J_hat[d, i]

                for e in range(3):
                    col = 3 * i + e
                    A[row, col] = epsilon_0 / self.dt + 0.5 * mu[d, e, i]

        return A, rhs

    def step(self):
        Ep = self.interpolate_field(self.xp, self.E)
        Bp = self.interpolate_field(self.xp, self.B)
        alpha = self.compute_alpha(Bp)
        v_hat = np.einsum("ijk,jk->ik", alpha, self.vp)

        J_hat, Pi_hat, rho = self.compute_moment_sources(alpha, v_hat)

        mu = np.zeros((3, 3, self.Nx))
        for i in range(self.Nx):
            mu[:, :, i] = -self.qp[0] * rho[i] / self.mp[0] * alpha[:, :, 0]  # Annahme: alle α gleich

        A, rhs = self.build_system_matrix_and_rhs(mu, J_hat, Pi_hat)
        E_flat, _ = gmres(A, rhs)
        E_theta = E_flat.reshape(3, self.Nx)

        E_theta_p = self.interpolate_field(self.xp, E_theta)
        v_new = v_hat + self.beta * np.einsum("ijk,jk->ik", alpha, E_theta_p)
        self.xp = (self.xp + self.dt * v_new[0, :]) % self.Lx
        self.vp = v_new
        self.t+=self.dt

