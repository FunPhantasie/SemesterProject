import numpy as np
from integrator_module import Integrator
from fourier_module import FourierSolver
from animationstudio_module import AnimatedScatter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments

class BurgerSolver:
    """
        Namespace:
        u: Speed Field
        u0: Initial Value Before Integration
        u1:  Value after Evolving
        X_hat: Fourier of Variable
        x_inv_hat: Inverse/Back Fourier of Variable
        u to u_hat to ou
    """

    # Parameters
    L = 25.0 * np.pi  # Border length
    N = 1024 * 32  # Number of grid points
    nu = 0.001  # Viscosity


    dt = 0.001  # Time step
    dx = L / N





    # Grid and wavenumbers
    x = np.linspace(0, L, N, endpoint=False)  # Periodic domain [0, 2pi) Grid
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers




    k2=k**2

    # Forcing (Mexican Hat)
    l = 1.0
    strength = 0.25




    def __init__(self,dimension="1d",step_method="EulerForward"):
        self.calculus=Integrator(step_method,self.dgl_eq)
        self.fourious=FourierSolver(dimension)

        self.u = np.sin(self.x) + 0.5 * np.sin(3 * self.x)+np.random.normal(0,0.5, self.N)
        self.u2 = np.zeros_like(self.u)  # temp
        self.u_hat = np.zeros_like(self.u)
        self.u_hat_dealised =np.zeros_like(self.u)
        self.u2_hat = np.zeros_like(self.u)  # temp

        # Define forcing correlation functions
        self.chi = lambda x: (1.0 - x ** 2.0 / self.l ** 2.0) * np.exp(- 0.5 * x ** 2.0 / self.l ** 2.0)
        self.chihat = lambda k: self.l ** 3.0 * k ** 2.0 * np.exp(- 0.5 * k ** 2.0 * self.l ** 2.0)
        self.lmbda = np.sqrt(self.chihat(self.k))  # Square root of chi_hat for forcing


        self.f_hat = np.zeros(self.N, dtype=complex)  # Forcing term in Fourier space


    def dgl_eq(self,u_hat):
        self.u_hat = self.fourious.forward(self.u)
        self.u_hat_dealised = self.fourious.dealise(self.u_hat, self.N)
        self.u2 = (self.fourious.inverse(self.u_hat_dealised)) ** 2
        self.u2_hat = self.fourious.forward(self.u2)
        return (-0.5j*self.k *(u2_hat)+self.strength*f_hat)

    def gen_force(self):
        # Generates the current forcing eta
        # $\hat{w}(k, t) = \mathcal{FFT}[\mathcal{N}(0, \frac{1}{\Delta x} \frac{1}{\Delta t})]$
        # $\hat{\eta} = \sqrt[4]{2 \pi} \sqrt{\hat{\chi}(k_n)} \hat{w}(k, t)$
        variance = 1.0 / (self.dx * self.dt)
        w = np.random.normal(0.0, np.sqrt(variance), size=self.N)
        w_hat = self.fourious.forward(w)
        self.f_hat = (2.0 * np.pi) ** 0.25 * w_hat * self.lmbda


    def runStep(self,i):
        self.gen_force()





        self.u_hat=self.calculus.EulerForward(self.dt,self.u_hat,self.f_hat)*np.exp(-self.nu*self.k2*self.dt)

        self.u=self.fourious.inverse(self.u_hat)
        return self.x,self.u,i*self.dt



test=BurgerSolver()
pixar_studio=AnimatedScatter(test.runStep,xlim=(0, 2*np.pi), ylim=(-10, 10))
pixar_studio.start()