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
    L = 2 * np.pi  # Border length
    N = 256  # Number of grid points
    nu = 0.05  # Viscosity


    dt = 0.01  # Time step
    dx = L / N

    # Grid and wavenumbers
    x = np.linspace(0, L, N, endpoint=False)  # Periodic domain [0, 2pi) Grid
    k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers
    f_hat = np.zeros(N, dtype=complex)
    f_hat[1:5] = (np.random.randn(4) + 1j * np.random.randn(4)) * 0.1
    f_hat[-5:-1] = np.conj(f_hat[1:5][::-1])  # ensure real-valued f(x)
    f = np.fft.ifft(f_hat).real
    k2=k**2

    def __init__(self,dimension="1d",step_method="EulerForward"):
        self.calculus=Integrator(step_method,self.dgl_eq)
        self.fourious=FourierSolver(dimension)

        self.u = np.sin(self.x) + 0.5 * np.sin(3 * self.x)+np.random.normal(0,0.5, self.N)
        self.u2 = np.zeros_like(self.u)  # temp
        self.u_hat = np.zeros_like(self.u)
        self.u_hat_dealised =np.zeros_like(self.u)
        self.u2_hat = np.zeros_like(self.u)  # temp

    def dgl_eq(self,u_hat, u2_hat, f_hat):
        return (-1j*self.k *(u2_hat)+f_hat)

    def runStep(self,i):
        self.u_hat = self.fourious.forward(self.u)
        self.u_hat_dealised = self.fourious.dealise(self.u_hat, self.N)
        self.u2=0.5*(self.fourious.inverse(self.u_hat_dealised))**2
        self.u2_hat = self.fourious.forward(self.u2)




        self.u_hat=self.calculus.EulerForward(self.dt,self.u_hat,self.u2_hat,self.f_hat)*np.exp(-self.nu*self.k2*self.dt)

        self.u=self.fourious.inverse(self.u_hat)
        return self.x,self.u,i*self.dt



test=BurgerSolver()
pixar_studio=AnimatedScatter(test.runStep,xlim=(0, 2*np.pi), ylim=(-10, 10))
pixar_studio.start()