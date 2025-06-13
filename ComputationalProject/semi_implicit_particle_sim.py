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

    # Borders
    Lx = 25.0   # Border length
    Ly= 25.0  # box size (fits fastest growing mode of two-stream); space step (normalized to u_d/omega_p)
    Lz = 25.0
    #Grid Points alongs Axis
    Nx = 10   # Number of grid points
    Ny = 10   # Number of grid points
    Nz = 10
    #Number per Cell
    NPpCell=20


    # Constants
    c=1
    pi=np.pi
    q=-1.0
    m=1.0
    omega_p = 1.  # plasma frequency
    epsilon_0 = 1.  # convenient normalization




    def __init__(self,dimension="2d",step_method="EulerForward"):
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nx

        self.dt = 0.01 #Check Bedinung


        # Implicit Parameter
        self.theta = 0.5  # Implicitness parameter
        #Total Particles
        self.Np = self.NPpCell * self.Nz * self.Ny * self.Nz

        #Constants
        self.charge = self.omega_p ** 2 / (self.q / self.m) * self.epsilon_0 * self.Lx / self.NP  # particle charge
        self.combi = self.c * self.theta * self.dt

        # Grid and wavenumbers (Steps dx,dy,dz)
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        self.z =  np.linspace(0, self.Lz, self.Nz, endpoint=False)
        self.X, self.Y, self.Z= np.meshgrid(self.x, self.y,self.z, indexing='ij') #Meshgrid Discussion about Indexing

        # Fake Particels to Grid (Computational Particles)
        self.rho = np.zeros([self.Nx*self.Ny*self.Nz])
        self.E_theta = np.zeros([self.Nx,self.Ny,self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
        self.B = np.zeros([self.Nx,self.Ny,self.Nz])  # Bz (2D)

        #Initialize the Particles Global Positions and Velocities
        self.vp = np.zeros([3, self.NP])
        self.Fp = np.zeros([3, self.NP])
        self.Ep = np.zeros([3, self.NP])
        self.Bp = np.zeros([3, self.NP])
        self.xp = np.zeros([3, self.NP])
        #In Simulation wurde nur in x Ebene Gerechnet
        self.pos_p = np.random.uniform(0, self.Lx, self.Np)
        self.vstart_p = np.random.normal(0, 1, self.Np)

        self.vp[0,:]=self.vstart_p
        self.B[2,...]=1.



        # Solve Method
        self.calculus = Integrator(step_method, self.dgl_eq)
        self.fourious = FourierSolver(dimension)

        # Initial state of the simulation
        self.t = 0.0
        self.N_steps = 1

        self.Ekin0 = np.sum(self.vp ** 2) * 0.5
        self.Ekin = []
        self.times = []


    def deposit_charge(self,x_p,rho,ShapeFunctionn):
        """8 Volumes"""
        for particle_index in range(self.Np):
            x,y,z=x_p[:,particle_index]
            ix,iy,iz = int(x/self.dx),int(y/self.dy),int(z/self.dz)
            #Arround The World
            #Muss Rho Volumes zuordnen
            for ax in [0,1]:
                for by in [0,1]:
                    for cz in [0,1]:
                        grid_point_x = np.mod(ix + ax, self.self.Nx) #Entweder x0 oder x1
                        grid_point_y = np.mod(iy + by, self.self.Nx)
                        grid_point_z = np.mod(iz + cz, self.self.Nx)
                        Vz=(-1)**(ax+by+cz)
                        VolumeHelper=Vz *(x-grid_point_x)*(y-grid_point_y)*(z-grid_point_z)
                        VolumeIndex= np.mod((1-ix) + ax, self.self.Nx) , np.mod((1-iy) + by, self.self.Nx) , np.mod((1-iz) + cz, self.self.Nx)
                        rho[VolumeIndex]+=ShapeFunctionn(VolumeHelper)

    def solve_fields(self):
        """Compute electric field using Fourier Poisson solver"""
        # rho -> potential -> E
        pass

    def interpolate_fields_to_particles(self,field,x_p,fieldp,ShapeFunctionn):
        """Interpolate E and B fields to particle positions"""
        #Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
        for particle_index in range(self.Np):
            x, y, z = x_p[:, particle_index]
            ix, iy, iz = int(x / self.dx), int(y / self.dy), int(z / self.dz)
            # Arround The World
            # Muss Rho Volumes zuordnen
            for ax in [0, 1]:
                for by in [0, 1]:
                    for cz in [0, 1]:
                        grid_point_x = np.mod(ix + ax, self.self.Nx)  # Entweder x0 oder x1
                        grid_point_y = np.mod(iy + by, self.self.Nx)
                        grid_point_z = np.mod(iz + cz, self.self.Nx)
                        Vz = (-1) ** (ax + by + cz)

                        VolumeHelper = Vz * (x - grid_point_x) * (y - grid_point_y) * (z - grid_point_z)
                        VolumeIndex = np.mod((1 - ix) + ax, self.self.Nx), np.mod((1 - iy) + by, self.self.Nx), np.mod(
                            (1 - iz) + cz, self.self.Nx)
                        field[:,particle_index] += ShapeFunctionn(fieldp[:,VolumeIndex],VolumeHelper)


    def apply_boundary_conditions(self):
        """Apply periodic or absorbing BCs to particles and fields"""
        pass

    def dgl_eq(self, t, y):
        """Optional ODE interface for integrator"""
        pass

    def cross(self,a,b):
        return np.cross(a,b)
    def dot(self,a,b):
        return np.dot(a,b)
    def absolute(self,a):
        return np.abs(a)
    def FieldEvolver(self):
        pass
    def ParticleMover(self):
        pass
    def RHS(self):
        pass

    def calc_v_hat(self,v,beta,E_theta):
        return v+beta*E_theta
    def calc_v_midpoint(self,v_hat,beta,B):
        helper= v_hat+beta (self.cross(v_hat,B)+beta *B *self.dot(v_hat,B))
        helper*=1/(1+beta**2*self.absolute(B))
        return helper
    def E_theta_RHS(self,E,B,J,rho):
        return E+self.combi *(self.curl(B)-4*self.pi/self.c* J)-self.combi**2 *4 *self.pi *self.gradient(rho)
    def E_theta_LHS(self,E_theta,mu):
        helper=E_theta*mu
        return E_theta+helper-self.combi**2 *(self.laplace_vector(E_theta)+self.gradient(self.divergence(helper)))
    def laplace_vector(self,A):
        return A
    def divergence(self,A):
        return self.dot(A,A) #Wrong
    def gradient(self,scalar):
        return np.array([scalar]*self.Nx) #Wrong
    def curl(self,A):
        return A
    def Half_step(self,x_i):
        """Advance one full PIC cycle"""

        self.deposit_charge(x_i)
        self.solve_fields(x_i)
        self.interpolate_fields_to_particles(x_i)
        #Now obtained habe En_theta, Bn
        beta = self.q_p * self.dt / (2 * self.m_p * self.c)
        self.v_hat=self.calc_v_hat(self.v,beta,self.E_theta)

        self.calc_v_midpoint()
        self.particle_pusher(x_i,0.5* self.dt)
        self.apply_boundary_conditions(x_i)
        return
    def step(self):
        """Advance one full PIC cycle"""
        self.xp_iter=self.Half_step(self.xp) #Should be Grid

        for i in range(2):
            self.xp_iter=self.Half_step(self.xp_iter)

        self.xp+=self.xp_iter*self.dt
        #Könnte auch V_n1 bestimmen aber brauch man nicht. Wird beim nächsten neu Approximiert
        self.t += self.dt

