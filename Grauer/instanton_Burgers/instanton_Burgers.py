import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import os
import shutil
import csv
import matplotlib.pyplot as plt

class TimeGrid:
    def __init__(self, Nt, t_begin=-10):
        self.Nt = Nt
        tb = t_begin
        te = 0
        t = jnp.linspace(tb, te, Nt)
        talpha = 1 # exponential decay rate
        t_A = (te - tb)/(np.exp(-talpha* te ) - np.exp(-talpha * tb))
        t_B = tb - t_A*np.exp(-talpha*tb)
        t = t_A * np.exp(-talpha * t) + t_B
        t[-1] = te
        t[0] = tb
        dt = np.roll(t, -1) - t
        dt = dt[:-1]  # remove last element which is not needed
        self.t = jnp.array(t)
        self.dt = jnp.array(dt)

class RealSpaceGrid:
    def __init__(self, Nx):
        self.Nx = Nx
        L = 2 * jnp.pi
        self.dx = L / Nx
        self.x_val = jnp.linspace(0.0, L, num=Nx, endpoint=False)

class FourierSpaceGrid:
    def __init__(self, Nx):
        # self.kx = 2*jnp.pi*jnp.fft.rfftfreq(Nx, self.dx)
        self.kx = jnp.arange(0, Nx//2 + 1)
    
        self.k2 = self.kx ** 2
        self.k2 = self.k2.at[0].set(1.0)
        self.k2_inv = 1.0 / self.k2
        self.k2 = self.k2.at[0].set(0.0)
        self.k4 = self.k2 * self.k2

class Burgers(TimeGrid, RealSpaceGrid, FourierSpaceGrid):
    def __init__(self, Nx, nu, F, mu, a, out_dir):
        RealSpaceGrid.__init__(self, Nx)
        FourierSpaceGrid.__init__(self, Nx)
        self.nu = nu                                # viscosity
        self.F = F                                  # linear penalty
        self.mu = mu                                # quadratic penalty
        self.a = a                                  # terminal constraint target value
        self.out_dir = out_dir
        self.init_output()
    
    def init_output(self):
        self.out_num = 0
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def setup_turbulence(self):
        x = self.x_val
        u = -jnp.cos(x + 1.4) - 1./3. * 4 * jnp.cos(2 * x + 2.3)
        return jnp.fft.rfft(u)

    def delta_F(self):
        dx = self.dx
        xb = 0
        return 1./dx * jnp.exp(1.j*self.kx*xb)
    
    def delta_x_F(self):
        return self.grad_F(self.delta_F())
    
    def chi_F(self):
        if not hasattr(self, "_chi_F_first_call_done"):
            print("chi_F needs to be tested!")
            self._chi_F_first_call_done = True
        
        kx = self.kx
        Nx = self.Nx
        lam = 8.**2.
        chi_F = lam*jnp.sqrt(2*jnp.pi)*kx*kx*jnp.exp(-lam*kx*kx/2.)
        # chi_F = jnp.where(kx == 0.0, 0.0, kx ** (-3))
        cutoff = int(kx.size * 2.0 / 3.0)
        sigma = jnp.sum(chi_F[1:cutoff])
        eps = 1
        chi_F = chi_F * eps*2./3.*Nx/sigma
        chi_F = chi_F.at[cutoff:].set(0.0)
        chi_F = chi_F.at[0].set(0.0)
        xb = 0.0
        chi_F = jnp.exp(1j*kx*xb)*chi_F
        return chi_F

    def dealias(self, IN_F):
        mask = (jnp.sqrt(jnp.abs(self.k2)) <= float(self.Nx) / 3.)
        return IN_F * mask

    def calc_dt(self, u_F, cfl):

        ux = jnp.fft.irfft(u_F).real
        umax = jnp.abs(ux).max()
        # print("umax = ", umax)

        # cfl = umax*dt/dx --> dt = cfl*self.dx/umax
        print("should be: dt =",cfl*self.dx/umax)
        dt = 0.0002  # hard-coded as per original
        return dt

    def multiply_F(self, f_F, g_F):
        return np.fft.rfft(np.fft.irfft(f_F)*np.fft.irfft(g_F))

    def grad_F(self, f_F):
        return 1.j*self.kx*f_F
    
    def convolution_F(self, f_F, g_F):
        kx = self.kx
        Nx = self.Nx
        xb = 0.0
        return 2.*jnp.pi*f_F*g_F/Nx*jnp.exp(1.j*kx*xb)

    def prop_F(self, dt):
        return jnp.exp(-self.nu*self.k2*dt)

    def RHS_u_F(self, p_F, u_F):
        return -0.5 * self.grad_F(jnp.fft.rfft(jnp.fft.irfft(u_F)**2)) + self.convolution_F(self.chi_F(), p_F)
        
    def RHS_p_F(self, p_F, u_F):
        RHS_p_F = +self.multiply_F(u_F, self.grad_F(p_F))
        return RHS_p_F
        
    def step_forward_F(self, p_F, u_F, dt):
        RHS = self.RHS_u_F(p_F, u_F)
        ret = self.prop_F(dt) * (u_F + RHS * dt)
        return ret
    
    def step_adjoint_F(self, p_F, u_F, dt):
        RHS = self.RHS_p_F(p_F, u_F)
        ret = self.prop_F(dt) * (p_F + RHS * dt)
        return ret

    def force_symmetry(self, f_F):
        return 1.j*f_F.imag

    def solve_forward_F(self, p_F, u_F, dt):
        Nt = u_F.shape[0]
        u_F_star = u_F[0]
        for i in range(1, Nt):
            delta_t = dt[i-1]
            u_F_star = self.force_symmetry(self.step_forward_F(p_F[i-1], u_F_star, delta_t))
            u_F = u_F.at[i].set(u_F_star)
        return u_F
    
    def gradObsT_F(self):
        gradObsT_F = -(self.F - self.mu * self.observable(u_F)) * self.delta_x_F()
        return gradObsT_F
    
    def solve_adjoint_F(self, z_F, u_F, dt):
        Nt = u_F.shape[0]
        z_F_star = self.gradObsT_F()
        z_F = z_F.at[-1].set(z_F_star)
        for i in range(1, Nt):
            delta_t = dt[Nt-i-1]
            z_F_star = self.step_adjoint_F(z_F_star, u_F[Nt-i], delta_t)
            z_F = z_F.at[Nt-i-1].set(z_F_star) 
        return z_F
    
    def compute_grad_F(self, p_F, z_F):
        return p_F - z_F
    
    def observable(self, u_F):
        u_x = jnp.fft.irfft(1j*self.kx*u_F)
        return u_x[-1][0]
    
    def actionDensity(self, p_F):
        chi_p = jnp.fft.irfft(self.convolution_F(self.chi_F(), p_F))
        p = jnp.fft.irfft(p_F)
        p_chi_p = p*chi_p
        actionDensity = 0.5*self.integrate(p_chi_p)
        return actionDensity
    
    def integrate(self, f):
        ret = 0.
        for val in f:
            ret += val
        ret *= self.dx
        return ret
        
    def compute_L_A(self, z_F, u_F, dt):
        action = 0.
        Nt = z_F.shape[0]
        for i in range(0, Nt):
            delta_t = dt[i]
            if i == 0:
                action += 0.5*delta_t*self.actionDensity(z_F[0])
            elif i == Nt-1:
                action += 0.5*dt[Nt-2]*self.actionDensity(z_F[-1])
            else:
                action += delta_t*self.actionDensity(z_F[i])
            loss = action + 1. # wrong
        return action, loss    
    
def run_CS(sim: Burgers, p_F, u_F, dt):
    A = 0.
    epsilon = 1.e-8
    sigma = 1.
    delta_A = epsilon+1
    while(jnp.abs(delta_A) > epsilon):
        z_F = sim.solve_adjoint_F(p_F, u_F, dt)
        grad_F = sim.compute_grad_F(p_F, z_F)
        maxabsgrad_F = jnp.max(jnp.abs(grad_F))

        p_F = p_F - sigma*grad_F
        u_F = sim.solve_forward_F(p_F, u_F, dt)

        S, loss = sim.compute_L_A(z_F, u_F, dt)
        A_bckp = A
        A = sim.observable(u_F)
        print("F = ", sim.F, "A =", A, "S = ", S, "max grad", maxabsgrad_F, "sigma = ", sigma)
        delta_A_bckp = delta_A
        delta_A = (A_bckp - A)/sigma/A_bckp
        if(delta_A_bckp*delta_A < 0.):
            sigma *= 0.8 #0.95

    with open(sim.out_dir + '/FAS.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sim.F, A, S])

    print(sim.F, A, S)
    return p_F, u_F

time_grid = TimeGrid(Nt=1000)
burgers = Burgers(Nx=1024, nu=0.1, F = -1.2, mu = 0, a = 10, out_dir="output_burgers")

u_F = jnp.zeros((time_grid.Nt, burgers.kx.size), dtype=jnp.complex64)
p_F = jnp.zeros_like(u_F)

dF = -0.02
burgers.F = -0.001
while burgers.F < 100:
    p_F, u_F = run_CS(burgers, p_F, u_F, dt=time_grid.dt)
    burgers.F += dF

# plt.plot(np.fft.irfft(u_F[-1])); plt.show()
