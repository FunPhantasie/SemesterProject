import numpy as np
from integrator_module import Integrator
from fourier_module import FourierSolver
from animationstudio_module import Animater
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
    L = 25.0   # Border length
    N = 1024 * 32  # Number of grid points
    nu = 0.01  # Viscosity


    dt = 0.01  # Time step
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
        # mass muss zu den gitter punkten passen kraft mit amplitude eins wenn start 1
        # änderung nicht aus der zelle bewegen
        # rauschen correlation erfüllt
        # MExanischer hut ?MIttelwert????
        self.t=0
        self.f_hat = np.zeros(self.N, dtype=complex)  # Forcing term in Fourier space


    def dgl_eq(self,u_hat,t):

        self.u_hat_dealised = self.fourious.dealise(u_hat, self.N) #Overwriteing is faster than creating new one
        self.u2 = (self.fourious.inverse(self.u_hat_dealised)) ** 2
        self.u2_hat = self.fourious.forward(self.u2)
        return (-0.5j*self.k *(self.u2_hat)+self.strength*self.f_hat)

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
        self.u_hat = self.fourious.forward(self.u)
        self.t+=self.dt



        self.u_hat=self.calculus.EulerForward(self.t,self.u_hat,self.dt)*np.exp(-self.nu*self.k2*self.dt)

        self.u=self.fourious.inverse(self.u_hat)
        return self.x,self.u,self.t



test=BurgerSolver()
pixar_studio=Animater(test.runStep,xlim=(0, test.L), ylim=(-10, 10)) #(x, u, k, E, pdf_x, pdf_y, t)
pixar_studio.start()