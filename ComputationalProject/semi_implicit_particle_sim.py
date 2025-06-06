import numpy as np

from ..ChaosSim.integrator_module import Integrator
from ..ChaosSim.fourier_module import FourierSolver
from ..ChaosSim.animationstudio_module import AnimatedScatter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments

class PIC_2D_Solver:
    """
        Namespace:

    """

    # Parameters
    Lx = 25.0   # Border length
    Ly= 25.0
    Nx = 1024   # Number of grid points
    Ny = 1024   # Number of grid points



    dx = Lx / Nx
    dy = Ly / Ny





    def __init__(self,dimension="2d",step_method="EulerForward"):
        # Grid and wavenumbers
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)  # Periodic domain [0, 2pi) Grid
        self.y = np.linspace(0, self.Ly, self.Ny, endpoint=False)  # Periodic domain [0, 2pi) Grid
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij') #Meshgrid

        self.dt = 0.01
        self.t = 0.0

        # Particles
        self.Np = 100
        self.q = 1.0  # Charge
        self.m = 1.0  # Mass
        self.theta = 0.5  # Implicitness parameter


        #Initialize the Particles Positions and Velocities
        self.xp = np.random.uniform(0, self.Lx, self.Np)
        self.yp = np.random.uniform(0, self.Ly, self.Np)
        self.vxp = np.random.normal(0, 1, self.Np)
        self.vyp = np.random.normal(0, 1, self.Np)

        # Initialize the Fields
        self.rho = np.zeros((self.Nx, self.Ny))
        self.E = np.zeros((self.Nx, self.Ny, 2))  # Ex, Ey
        self.B = np.zeros((self.Nx, self.Ny, 1))  # Bz (2D)

        # Solve Method
        self.calculus = Integrator(step_method, self.dgl_eq)
        self.fourious = FourierSolver(dimension)
        self.particle_pusher=self.BorisPusher



    def deposit_charge(self):
        """Deposit particle charge onto the grid using CIC scheme"""
        pass

    def solve_fields(self):
        """Compute electric field using Fourier Poisson solver"""
        # rho -> potential -> E
        pass

    def interpolate_fields_to_particles(self):
        """Interpolate E and B fields to particle positions"""
        pass


    def apply_boundary_conditions(self):
        """Apply periodic or absorbing BCs to particles and fields"""
        pass

    def dgl_eq(self, t, y):
        """Optional ODE interface for integrator"""
        pass


    def FieldEvolver(self):
        pass
    def ParticleMover(self):
        pass
    def RHS(self):
        pass
    def BorisPusher(self):
        pass
    def step(self):
        """Advance one full PIC cycle"""
        self.deposit_charge()
        self.solve_fields()
        self.interpolate_fields_to_particles()
        self.particle_pusher()
        self.apply_boundary_conditions()
        self.t += self.dt

