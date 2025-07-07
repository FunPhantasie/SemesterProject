# implements a 1-D stochastically forced numerical Burgers simulation
# base structure based on code by Rainer Grauer

import numpy as np

from helper import *

class Burgers:
    def __init__(self, n, x_min, x_max, dt, nu, chihat, strength):
        # n:        number of cells
        # x:        cell coordinates
        # u:        burgers velocity
        # k:        wave numbers
        # nu:       viscosity constant
        # chi:      forcing correlation
        # eta:      calculated forcing for the current timestep
        # strength: forcing scale factor

        # dudx:     derivative of u

        # <>hat:    quantity in k-space
        # lmbda:    sqrt of chihat

        self.n = n

        self.x_min = x_min
        self.x_max = x_max
        self.x_len = (self.x_max - self.x_min)
        self.dx = self.x_len / n
        self.x = np.linspace(self.x_min, self.x_max - self.dx, n)

        self.u = np.zeros(len(self.x))
        self.dudx = self.u
        self.uhat = rft(self.u, self.x_min, self.x_max)
        self.k = rft_k(len(self.uhat), self.x_min, self.x_max)
        self.k2 = np.square(self.k)
        self.dk = rft_dk(self.x_min, self.x_max)

        self.dt = dt
        self.t = 0.0

        self.nu = nu

        self.chihat = chihat(self.k)
        self.lmbda = np.sqrt(self.chihat)
        self.etahat = np.zeros(len(self.k))
        self.eta = np.zeros(len(self.x))
        self.strength = strength
        #Forward BAck Ward Sheme
        self.scheme = self.heun

    def time_step(self):
        # performs one time step

        #DIfferent Kind of Force the Convolution this smaller
        self.gen_force()
        self.scheme()

        self.t += self.dt

        # calculates $u$ and $\partial_x u$ in spacial coordinates
        self.u = irft(self.uhat, self.x_min, self.x_max)
        self.dudx = 0.5 * (np.roll(self.u, -1) - np.roll(self.u, 1)) / self.dx

    def gen_force(self):
        # generates the current forcing eta
        # $\hat{w}(k, t) = \mathcal{FFT}[\mathcal{N}(0, \frac{1}{\Delta x} \frac{1}{\Delta t})]$
        # $\hat{\eta} = \sqrt[4]{2 \pi} \sqrt{\hat{\chi}(k_n)} \hat{w}(k, t)$
        variance = 1.0 / (self.dx * self.dt)
        # note that w scales with the inverse square root of dx and dt
        w = np.random.normal(0.0, np.sqrt(variance), size=self.n)
        what = rft(w, self.x_min, self.x_max)
        self.etahat = (2.0 * np.pi) ** 0.25 * what * self.lmbda
        self.eta = irft(self.etahat, self.x_min, self.x_max)

    def rhs(self, uhat):
        # calculates the Right Hand Side (rhs) used in Euler and Heun steps
        # $r_n = \hat{u}_n + \Delta t \left( - \frac{i k_n}{2} \widehat{(u)^2}_n + \hat{\eta}_n \right)$
        uhat[int(2.0 * len(self.k) / 3.0):len(self.k)] = 0 # aliasing
        u2hat = rft(irft(uhat, self.x_min, self.x_max) ** 2.0, self.x_min, self.x_max)
        return uhat + self.dt * (-0.5j * self.k * u2hat + self.strength * self.etahat)

    def prop(self):
        # calculates the prop factor used in Euler and Heun Steps
        # $p_n = e^{-k_n^2 \nu \Delta t}$
        return np.exp(- self.k2 * self.nu * self.dt)

    def euler(self):
        # Euler Step
        # an Euler Step for the Burgers Equation separated into rhs (r) and prop (p)
        # $\hat{u}^{(i+1)}_n = r(\hat{u}^{(i)})_n \, p_n$
        self.uhat = self.rhs(self.uhat) * self.prop()

    def heun(self):
        # Heun Step
        # a Heun Step for the Burgers Equation separated into rhs (r) and prop (p)
        # $\hat{u}'^{(i+1)}_n = r(\hat{u}^{(i)}_n) \, p_n$
        # $\hat{u}^{(i+1)}_n = \frac{1}{2} (r(\hat{u}^{(i)})_n \, p_n + r(\hat{u}'^{(i+1)})_n)$
        uone = self.rhs(self.uhat) * self.prop()
        self.uhat = 0.5 * (self.uhat * self.prop() + self.rhs(uone))
