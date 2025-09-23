import numpy as np
from scipy import sparse
from scipy.sparse import linalg

class TwoStreamPIC1D:
    """
    1D electrostatic PIC (normalized explicit) for the electron two‑stream case.

    ✅ Uses the *original* calculation formulas from your `Normalized_Explicit.py`:
      - Field energy:  E_field = 0.5 * sum(Eg**2) * dx
      - Kinetic:       E_kin   = 0.5 * (Q/QM) * sum(vp**2)
      - Momentum:      Pmom    = (Q/QM) * sum(vp)

    ⚙️ Initialization stays outside. You assign `xp` and `vp` after construction.

    Behavior per your request:
      - Arrays (rho, J, P_exx, rho_e, Ep) are saved in self.
      - Only `step()` returns a diagnostics dict.
    """

    def __init__(self, L, NG, PPC, DT, ES=True):
        self.L  = float(L)
        self.NG = int(NG)
        self.PPC = int(PPC)
        self.Np = self.NG * self.PPC
        self.DT = float(DT)
        self.ES = bool(ES)

        self.WP = 1.0
        self.QM = -1.0
        self.dx = self.L / self.NG

        self.Eg = np.zeros(self.NG)
        self.E  = self.Eg.copy()
        self.Q = self.WP**2 / (self.QM * self.Np / self.L)
        self.rho_back = -self.Q * self.Np / self.L

        self.xp = None
        self.vp = None

        self.rho   = np.zeros(self.NG)
        self.J     = np.zeros(self.NG)
        self.P_exx = np.zeros(self.NG)
        self.rho_e = np.zeros(self.NG)
        self.Ep    = None

        if self.ES:
            self.calc_E=self._field_from_poisson
        else:
            self.calc_E=self._ampere_update

        _Poisson = sparse.spdiags(([1, -2, 1] * np.ones((1, NG - 1), dtype=int).T).T, \
                         [-1, 0, 1], NG - 1, NG - 1)
        self.Poisson = _Poisson.tocsc()

        self.xg = np.linspace(0, self.L - self.dx, self.NG) + 0.5 * self.dx# grid at cell centers
        self.it = 0
        self.t = 0.0

    # ---------------- internal helpers ----------------
    def _wrap_positions(self):
        self.xp %= self.L



    # ---------------- deposition (original calc) ----------------
    def _deposit_CIC(self):
        NG = self.NG
        dx = self.dx
        Np = self.Np

        xp = self.xp
        vp=self.vp
        g1 = np.floor(xp / dx - 0.5).astype(int)
        g = np.concatenate((g1, g1 + 1))

        fraz1 = 1 - np.abs(xp / dx - g1 - 0.5)
        fraz = np.concatenate((fraz1, 1 - fraz1))

        g[g < 0] += NG
        g[g > NG - 1] -= NG

        p = np.arange(Np)
        prow = np.concatenate((p, p))

        mat = sparse.csc_matrix((fraz, (prow, g)), shape=(Np, NG))


        rho_e = self.Q / self.dx * mat.toarray().sum(axis=0)

        mat2 = mat.multiply(vp.reshape(Np, 1))
        mat3 = mat2.multiply(vp.reshape(Np, 1))

        self.J     = (self.Q / dx) * mat2.toarray().sum(axis=0)
        self.P_exx  = (self.Q / dx) * mat3.toarray().sum(axis=0)
        self.rho   = rho_e + self.rho_back

    # ---------------- fields (original calc) ----------------
    def _field_from_poisson(self):
        NG = self.NG
        dx = self.dx
        b = -self.rho[:-1] * dx * dx
        phi_red = linalg.spsolve(self.Poisson, b)
        phi = np.empty(NG)
        phi[:-1] = phi_red
        phi[-1] = 0.0
        self.Eg = -(np.roll(phi, -1) - np.roll(phi, 1)) / (2.0 * dx)
        self.E = self.Eg

    def _ampere_update(self):
        self.Eg = self.Eg - self.DT * self.J
        self.E = self.Eg

    # ---------------- interpolation & push (original calc) ----------------
    def _interp_E_to_particles(self):
        NG = self.NG
        dx = self.dx
        xr = self.xp / dx
        i0 = np.floor(xr).astype(int) % NG
        i1 = (i0 + 1) % NG
        w1 = xr - np.floor(xr)
        w0 = 1.0 - w1
        self.Ep = w0 * self.Eg[i0] + w1 * self.Eg[i1]

    def _push_v(self):
        self.vp += (self.Q / self.QM) * self.Ep * self.DT

    def _push_x(self):
        self.xp += self.vp * self.DT
        self._wrap_positions()

    # ---------------- main step ----------------
    def step(self):
        self._deposit_CIC()
        self.calc_E()
        self._interp_E_to_particles()
        self._push_v()
        self._push_x()
        self.t += self.DT
        self.it += 1



    # ---------------- diagnostics (original calc) ----------------
    def calcEnergy(self):
        return 0.5 * np.sum(self.Eg ** 2) * self.dx

    def calcPotEnergy(self):
        return 0.5 * np.sum(self.E ** 2) * self.dx

    def calcKinEnergy(self):
        return 0.5 * (self.Q / self.QM) * np.sum(self.vp ** 2)

    def calcMomentum(self):
        return (self.Q / self.QM) * np.sum(self.vp)