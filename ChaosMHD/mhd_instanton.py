import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import os
import shutil
import csv
import matplotlib.pyplot as plt
import argparse

class TimeGrid:
    def __init__(self, Nt, t_begin=-10.0, talpha=1.0):
        self.Nt = Nt
        tb, te = t_begin, 0.0

        # uniform grid in NumPy first
        t = np.linspace(tb, te, Nt)

        # exponential clustering toward te=0
        t_A = (te - tb) / (np.exp(-talpha * te) - np.exp(-talpha * tb))
        t_B = tb - t_A * np.exp(-talpha * tb)
        t = t_A * np.exp(-talpha * t) + t_B

        # force exact endpoints
        t[0] = tb
        t[-1] = te

        # compute dt and drop last element
        dt = np.roll(t, -1) - t
        dt = dt[:-1]

        # store as JAX arrays
        self.t = jnp.array(t)
        self.dt = jnp.array(dt)

# =============================================================
# 2D grids (real and Fourier) — pseudo‑spectral on [0,2π)^2
# =============================================================

class RealSpaceGrid2D:
    def __init__(self, N, L=2*jnp.pi):
        self.N = N
        self.L = L
        self.dx = L / N
        x1d = jnp.linspace(0.0, L, num=N, endpoint=False)
        self.x, self.y = jnp.meshgrid(x1d, x1d, indexing='ij')

class FourierSpaceGrid2D:
    def __init__(self, N):
        self.N = N
        kk = jnp.fft.fftfreq(N, d=1.0/N)
        kx, ky = jnp.meshgrid(kk, kk, indexing='ij')
        # real wavenumbers (no i); we multiply by 1j during ops
        self.kx = (2*jnp.pi/N) * kx
        self.ky = (2*jnp.pi/N) * ky
        k2 = -( (1j*self.kx)**2 + (1j*self.ky)**2 )
        self.k2 = k2
        self.k2 = self.k2.at[0,0].set(-1.0)
        self.k2inv = 1.0 / self.k2
        self.k2inv = self.k2inv.at[0,0].set(0.0)
        # 2/3 rule dealias mask
        kabs = jnp.sqrt(self.kx**2 + self.ky**2) * (N/(2*jnp.pi))
        kcut = (2.0/3.0)*(N/2.0)
        self.mask = (kabs <= kcut).astype(jnp.float32)

class MHDInstanton2D(TimeGrid, RealSpaceGrid2D, FourierSpaceGrid2D):
    def __init__(self, N=128, Nt=400, nu=1e-3, talpha=1.0):
        TimeGrid.__init__(self, Nt=Nt, t_begin=-10.0, talpha=talpha)
        RealSpaceGrid2D.__init__(self, N=N)
        FourierSpaceGrid2D.__init__(self, N=N)
        self.nu = nu
        self._chiF = self._build_chiF()

    # ---------------- FFT helpers ----------------
    def fft2(self, f):
        return jnp.fft.fft2(f)

    def ifft2(self, F):
        return jnp.fft.ifft2(F).real

    def dealias(self, F):
        return F * self.mask

    # ---------------- Biot–Savart (ω → z) ----------------
    def z_from_omegaF(self, omF):
        zxF = - self.k2inv * ((1j * self.ky) * omF)
        zyF = self.k2inv * ((1j * self.kx) * omF)
        zxF = zxF.at[0, 0].set(0.0 + 0.0j)
        zyF = zyF.at[0, 0].set(0.0 + 0.0j)
        return zxF, zyF

    # ---------------- Covariance χ(k) ----------------
    def _build_chiF(self, k0=4.0, width=1.0, amp=1.0):
        kabs = jnp.sqrt(self.kx ** 2 + self.ky ** 2) * (self.N / (2 * jnp.pi))
        chi = amp * jnp.exp(-0.5 * ((kabs - k0) / width) ** 2)
        chi = chi.at[0, 0].set(0.0)
        return chi

    def cov_applyF(self, pF):
        return self._chiF * pF

    # ---------------- Nonlinearity for ω± ----------------
    def nonlinearF(self, om_pF, om_mF):
        zpxF, zpyF = self.z_from_omegaF(self.dealias(om_pF))
        zmxF, zmyF = self.z_from_omegaF(self.dealias(om_mF))
        zpx, zpy = self.ifft2(zpxF), self.ifft2(zpyF)
        zmx, zmy = self.ifft2(zmxF), self.ifft2(zmyF)
        F_xx = self.fft2(zpx * zmx)
        F_xy = self.fft2(zpx * zmy)
        F_yx = self.fft2(zpy * zmx)
        F_yy = self.fft2(zpy * zmy)
        kx, ky = self.kx, self.ky
        NpF = (1j * kx) * (1j * ky) * F_xx + (1j * ky) * (1j * ky) * F_xy - (1j * kx) * (1j * kx) * F_yx - (
                    1j * kx) * (1j * ky) * F_yy
        NmF = (1j * kx) * (1j * ky) * F_xx + (1j * ky) * (1j * ky) * F_yx - (1j * kx) * (1j * kx) * F_xy - (
                    1j * kx) * (1j * ky) * F_yy
        return self.dealias(NpF), self.dealias(NmF)

    # ---------------- Forward RHS (ω±) ----------------
    def RHS_forwardF(self, p_pF, p_mF, om_pF, om_mF):
        NpF, NmF = self.nonlinearF(om_pF, om_mF)
        diff_p = self.k2 * self.nu * om_pF
        diff_m = self.k2 * self.nu * om_mF
        fp = self.cov_applyF(p_pF)
        fm = self.cov_applyF(p_mF)
        return NpF + diff_p + fp, NmF + diff_m + fm

    # ---------------- Adjoint RHS (p±) ----------------
    def RHS_adjointF(self, p_pF, p_mF, om_pF, om_mF):
        zpxF, zpyF = self.z_from_omegaF(om_pF)
        zmxF, zmyF = self.z_from_omegaF(om_mF)
        zpx, zpy = self.ifft2(zpxF), self.ifft2(zpyF)
        zmx, zmy = self.ifft2(zmxF), self.ifft2(zmyF)
        pp = self.ifft2(p_pF);
        pm = self.ifft2(p_mF)

        def dxf(f): return self.ifft2((1j * self.kx) * self.fft2(f))

        def dyf(f): return self.ifft2((1j * self.ky) * self.fft2(f))

        adv_p = zmx * dxf(pp) + zmy * dyf(pp)
        pmxF = - self.k2inv * ((1j * self.ky) * p_mF)
        pmyF = self.k2inv * ((1j * self.kx) * p_mF)
        pmx, pmy = self.ifft2(pmxF), self.ifft2(pmyF)
        Gp = dxf(zpx) * pmx + dyf(zpx) * pmy + dxf(zpy) * pmx + dyf(zpy) * pmy
        diff_p = self.k2 * self.nu * p_pF
        rhs_p = -(self.fft2(adv_p - Gp)) + diff_p
        adv_m = zpx * dxf(pm) + zpy * dyf(pm)
        ppxF = - self.k2inv * ((1j * self.ky) * p_pF)
        ppyF = self.k2inv * ((1j * self.kx) * p_pF)
        ppx, ppy = self.ifft2(ppxF), self.ifft2(ppyF)
        Gm = dxf(zmx) * ppx + dyf(zmx) * ppy + dxf(zmy) * ppx + dyf(zmy) * ppy
        diff_m = self.k2 * self.nu * p_mF
        rhs_m = -(self.fft2(adv_m - Gm)) + diff_m
        return self.dealias(rhs_p), self.dealias(rhs_m)

    # ---------------- ETDRK2 stepping ----------------
    def step_forward(self, p_pF, p_mF, om_pF, om_mF, dt):
        R1p, R1m = self.RHS_forwardF(p_pF, p_mF, om_pF, om_mF)
        Ep = jnp.exp(self.k2 * self.nu * dt)
        om_pF_tilde = Ep * (om_pF + dt * R1p)
        om_mF_tilde = Ep * (om_mF + dt * R1m)
        R2p, R2m = self.RHS_forwardF(p_pF, p_mF, om_pF_tilde, om_mF_tilde)
        om_pF_new = Ep * om_pF + 0.5 * dt * (Ep * R1p + R2p)
        om_mF_new = Ep * om_mF + 0.5 * dt * (Ep * R1m + R2m)
        return self.dealias(om_pF_new), self.dealias(om_mF_new)

    def step_adjoint(self, p_pF, p_mF, om_pF, om_mF, dt):
        R1p, R1m = self.RHS_adjointF(p_pF, p_mF, om_pF, om_mF)
        Ep = jnp.exp(self.k2 * self.nu * dt)
        p_pF_tilde = Ep * (p_pF + dt * R1p)
        p_mF_tilde = Ep * (p_mF + dt * R1m)
        R2p, R2m = self.RHS_adjointF(p_pF_tilde, p_mF_tilde, om_pF, om_mF)
        p_pF_new = Ep * p_pF + 0.5 * dt * (Ep * R1p + R2p)
        p_mF_new = Ep * p_mF + 0.5 * dt * (Ep * R1m + R2m)
        return self.dealias(p_pF_new), self.dealias(p_mF_new)

    # ---------------- Propagate on full time grid ----------------
    def solve_forward(self, p_histF, om_histF):
        Nt = om_histF.shape[0]
        omp, omm = om_histF[0, 0], om_histF[0, 1]
        for n in range(1, Nt):
            dt = self.dt[n - 1]
            omp, omm = self.step_forward(p_histF[n - 1, 0], p_histF[n - 1, 1], omp, omm, dt)
            om_histF = om_histF.at[n, 0].set(omp)
            om_histF = om_histF.at[n, 1].set(omm)
        return om_histF

    def terminal_adjoint(self, om_TpF, om_TmF, x0=0.0, y0=0.0, sigma=None, weight_plus=1.0, weight_minus=1.0):
        if sigma is None:
            sigma = 2 * self.dx
        X, Y = self.x, self.y
        blob = jnp.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma ** 2)) / (2 * jnp.pi * sigma ** 2)
        pTp = self.fft2(weight_plus * blob)
        pTm = self.fft2(weight_minus * blob)
        return pTp, pTm

    def solve_adjoint(self, om_histF, p_histF):
        Nt = om_histF.shape[0]
        p_pF, p_mF = self.terminal_adjoint(om_histF[-1, 0], om_histF[-1, 1])
        p_histF = p_histF.at[-1, 0].set(p_pF)
        p_histF = p_histF.at[-1, 1].set(p_mF)
        for n in range(Nt - 1, 0, -1):
            dt = -self.dt[n - 1]
            omp_prev, omm_prev = om_histF[n - 1, 0], om_histF[n - 1, 1]
            p_pF, p_mF = self.step_adjoint(p_pF, p_mF, omp_prev, omm_prev, dt)
            p_histF = p_histF.at[n - 1, 0].set(p_pF)
            p_histF = p_histF.at[n - 1, 1].set(p_mF)
        return p_histF

    # ---------------- Action & observable ----------------
    def action_density_t(self, pF_pm):
        chi = self._chiF
        S = 0.0
        for s in (0, 1):
            cp = self.ifft2(chi * pF_pm[s])
            p = self.ifft2(pF_pm[s])
            S = S + 0.5 * jnp.mean(p * cp) * (self.L ** 2)
        return S

    def action_trapz(self, p_histF):
        Nt = p_histF.shape[0]
        S = 0.0
        for n in range(Nt):
            dt = self.dt[n - 1] if n > 0 else self.dt[0]
            w = 0.5 if (n == 0 or n == Nt - 1) else 1.0
            S = S + w * dt * self.action_density_t(p_histF[n])
        return jnp.real(S)

    def observable(self, om_histF, kind="omega_plus_point", x0=0.0, y0=0.0):
        if kind == "omega_plus_point":
            omx = self.ifft2(om_histF[-1, 0])
            ix = int(jnp.round(x0 / self.dx)) % self.N
            iy = int(jnp.round(y0 / self.dx)) % self.N
            return float(omx[ix, iy])
        raise NotImplementedError

# =============================================================
# Outer loop (kept minimal, close to common 1D patterns)
# =============================================================

def run_CS(sim: MHDInstanton2D, p_histF, om_histF, max_iter=50, sigma=1.0, tol=1e-6):
    A_prev = jnp.inf
    for it in range(max_iter):
        # backward (adjoint path) for current forward trajectory
        p_histF = sim.solve_adjoint(om_histF, p_histF)
        # forward with updated optimal control p
        om_histF = sim.solve_forward(p_histF, om_histF)
        # diagnostics
        S = sim.action_trapz(p_histF)
        A = sim.observable(om_histF)
        print(f"it {it:02d} | A={A:+.3e}  S={float(S):.6e}")
        if jnp.isfinite(A_prev) and abs(A - A_prev) < tol:
            break
        A_prev = A
    return p_histF, om_histF

# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2D MHD instanton (Elsässer, ω±) [JAX]")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--Nt", type=int, default=400)
    ap.add_argument("--nu", type=float, default=1e-3)
    ap.add_argument("--talpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim = MHDInstanton2D(N=args.N, Nt=args.Nt, nu=args.nu, talpha=args.talpha)

    # allocate paths (complex in k)
    om_histF = jnp.zeros((args.Nt, 2, args.N, args.N), dtype=jnp.complex64)
    p_histF = jnp.zeros_like(om_histF)

    # tiny random seed for ω to break symmetry (JAX)
    key = jrandom.PRNGKey(args.seed)
    key, k2 = jrandom.split(key)
    om0p = jrandom.normal(key, (args.N, args.N), dtype=jnp.float32) * 1e-6
    om0m = jrandom.normal(k2, (args.N, args.N), dtype=jnp.float32) * 1e-6
    om_histF = om_histF.at[0, 0].set(jnp.fft.fft2(om0p))
    om_histF = om_histF.at[0, 1].set(jnp.fft.fft2(om0m))

    p_histF, om_histF = run_CS(sim, p_histF, om_histF, max_iter=20)

    # optional: save to .npz via numpy (host)
    import numpy as _np
    _np.savez_compressed(
        "instanton_mhd2d_out.npz",
        om_histF=_np.array(om_histF).astype(_np.complex64),
        p_histF=_np.array(p_histF).astype(_np.complex64),
        t=_np.array(sim.t).astype(_np.float32),
        dt=_np.array(sim.dt).astype(_np.float32),
    )
    print("saved → instanton_mhd2d_out.npz")