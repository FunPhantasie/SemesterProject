import numpy as np
from scipy import sparse
class PIC1D:
    def __init__(self, Lx=1.0, Nx=64,Np=200):
        self.Lx = float(Lx)
        self.Nx = int(Nx)
        self.Np = int(Np)
        self.dx = self.Lx / self.Nx
        self.xg = (np.arange(self.Nx) + 0.5) * self.dx  # cell centers
        self.WP = 1.0
        self.QM = -1.0
        charge_el = self.WP ** 2 / (self.QM * self.Np / self.Lx)
        self.rho_back = -charge_el * self.Np / self.Lx
        self.omega_p=1
        el = dict(name="e", q=-1, qDm=-1, Np=Np)
        pr = dict(name="p", q=1, qDm=1. / 1836., Np=Np, vp=np.zeros(Np))
        el["charge"] = self.omega_p ** 2 / (el["qDm"] * el["Np"] / self.Lx)
        sc_faktor = -pr["qDm"] / el["qDm"] * pr["Np"] / el["Np"] * el["Np"] / self.Lx * self.dx / el["Np"]
        self.charge = self.omega_p ** 2 / (pr["qDm"] * pr["Np"] / self.Lx) * sc_faktor
        self.xp = np.linspace(0, Lx, pr["Np"], endpoint=False)
        species = [el, pr]

    def _deposit_CIC(self):
        NG = self.Nx
        dx = self.dx
        xp = self.xp
        Np = self.Np

        g1 = np.floor(xp / dx - 0.5).astype(int)
        g = np.concatenate((g1, g1 + 1))

        fraz1 = 1 - np.abs(xp / dx - g1 - 0.5)
        fraz = np.concatenate((fraz1, 1 - fraz1))

        g[g < 0] += NG
        g[g > NG - 1] -= NG

        p = np.arange(Np)
        prow = np.concatenate((p, p))

        mat = sparse.csc_matrix((fraz, (prow, g)), shape=(Np, NG))


        rho_e = self.charge / self.dx * mat.toarray().sum(axis=0)

        print(rho_e)
        print(self.rho_back)



a = PIC1D()
a._deposit_CIC()