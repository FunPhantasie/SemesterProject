import numpy as np

from Simulation.explicit_particle_sim import Explicit_PIC_Solver
from Simulation.semi_implicit_particle_sim import PIC_Solver

from Analytics.AnalyticsOfNStep import run_save_steps as run_nstep

from Analytics.RenderManager import CallItRenderer
from Analytics.Animator import run_continuous
from Analytics.Animator import run_flipbook

"""
Initnitialisation of the Implicit Probelm of the electromagnetic two streams poblem.

Choose Params
"""



class twostream1D(PIC_Solver):
    def __init__(self,border=1,NG=1,PPC=20,dt=0.1,):

        #Parameter Conditions
        self.Lx = border  # Plasma Space/Borders
        self.Nx = NG  # Number of grid points
        self.totalN=3*self.Nx #Total Number of Gridppoints (3 Could be Wrong)



        # Resulting Connected Conditions
        self.dx = self.Lx / self.Nx

        self.E = np.zeros([3, self.Nx])  # E[0]
        self.B = np.zeros([3, self.Nx])  # B[2]
        self.E_theta = np.zeros([3, self.Nx])

        """
        All The Fields and Moments
        """
        Np = PPC * self.Nx  # Total Particles

        species=[{
                "name": "e",
                "q": -1.0,
                "m": 1.0,
                "beta_mag_par": 0,
                "beta_mag_perp": 0,
                "beta": None,
                "NPpCell": PPC,
                "Np":Np,
            },]


        super().__init__(dimension=1, dt=dt, stepssize=self.dx,border=(self.Lx,),gridNumbers=(self.Nx,),species=species )




    def ShaperParticle(self, x_p,Np, prefaktor, ShapeFunction,toParticle=False):
        # Validate prefaktor shape and assign helper
        if toParticle:
            helper = np.zeros([3, Np])

        # Initialize helper based on prefaktor type

        else:
            if np.isscalar(prefaktor):
                is_scalar = True
                is_vector = False
                is_single_value = True
            else:
                is_scalar = prefaktor.shape == (Np,)
                is_vector = prefaktor.shape == (3, Np)
                is_single_value = prefaktor.shape == (1,)
            if not (is_scalar or is_vector):
                raise ValueError(f"prefaktor shape {prefaktor.shape} is invalid. Expected (Np,) or (3, Np).")

            helper = (np.zeros([3, self.Nx]) if is_vector else np.zeros(self.Nx))



        # Process each particle
        for particle_index in range(Np):
            # Particle position in grid coordinates
            x = x_p[ particle_index]

            xn = (x / self.dx)
            ix= np.floor(xn) # int Verhalten bei Negativen Zahlen Falsch
            # Arround The World
            # Muss Rho Volumes zuordnen

            # Compute weights for all 8 grid points at once
            for ax in [0, 1]:
                # Periodic boundary conditions
                grid_x = np.mod(ix + ax, self.Nx)
                # Weight based on linear distance (CIC)
                wx = 1 - abs(xn - (ix + ax))

                weight = wx

                # Apply shape function and update grid
                if toParticle:
                    helper[:, particle_index] += prefaktor[:, grid_x] * ShapeFunction(weight)
                elif is_single_value:
                    helper[grid_x] += prefaktor * ShapeFunction(weight)
                elif is_scalar:
                    helper[grid_x] += prefaktor[particle_index] * ShapeFunction(weight)
                else:
                    helper[:, grid_x] += prefaktor[:, particle_index] * ShapeFunction(weight)

        return helper

def initialize_two_stream1D(Lx, Np,B,VT=0.005,V0=0.05, amplitude=0.01):
    """
    Initialize particle positions and velocities for a two-stream instability.

    Args:
        Lx (float): System length
        Np (int): Total number of particles
        amplitude (float): Amplitude of velocity perturbation

    Returns:
        tuple: (xp, vp_x) where xp is particle positions and vp_x is x-component of velocities
    """

    #xg = np.linspace(0, L - dx, NG) + dx / 2
    #Number of Grid Creates 0 bis L-dx
    #xp = np.linspace(0, L - L / Np, Np).T
    # Eavenly spaced bis L - L / Np
    vp = np.zeros([3, Np])
    xp1 = 2 * Lx / Np * np.arange(Np // 2)
    xp2 = 2 * Lx / Np * np.arange(Np // 2)


    vp1 = V0 + amplitude * np.sin(2 * np.pi / Lx * xp1)+sample_maxwellian_anisotropic(VT,Np//2)
    vp2 = -V0 - amplitude * np.sin(2 * np.pi / Lx * xp1)+sample_maxwellian_anisotropic(VT,Np//2)
    xp = np.concatenate([xp1, xp2])
    vp_x = np.concatenate([vp1, vp2])
    vp[0, :] = vp_x
    #B[2, ...] = 1
    return xp, vp,B
def normalized_init(Lx, Np,B,VT=0.005,V0=0.05, XP1=0.01,mode=1):
    """
    Initialize particle positions and velocities for a two-stream instability.
    Keydifference
    Xp ist Awechselnd postive/negativ velocity Keine Teilchen am Start am selber Position
    Pertubation Ist in Position Space not Velocity Space
    Difference of Momentum and Velocity Space

    :param Lx: Space in X_Direction
    :param Np: Number of particles
    :param B: MAgnetic Field
    :param VT: Thermal Energy Spread / Velocity
    :param V0: Base Velocity
    :param XP1: Amplitude of space perturbation
    :return:
    """
    vp = np.zeros([3, Np])
    xp = np.linspace(0, Lx - Lx / Np, Np)
    xp += XP1 * (Lx / Np) * np.sin(2 * np.pi * xp / Lx * mode) #Pertubation in Postion Space
    xp = np.mod(xp, L)

    vp_x = VT * (1 - VT ** 2) ** (-0.5) * np.random.randn(Np) #Pertubation Velocity
    #vx = np.random.normal(loc=0.0, scale=vth_par, size=Np) Should work same as randn*vp
    #Its done Relativistic momentum:  p=γmv m=1
    pm = np.arange(Np)
    pm = 1 - 2 * np.mod(pm + 1, 2) #-1,1,-1,1
    vp_x += pm * (V0 * (1 - V0 ** 2) ** (-0.5)) #Base Velocity One BAckwards One Forward
    vp[0, :] = vp_x
    return xp, vp,B
def sample_maxwellian_anisotropic(vth_par, Np):
    # Sampling für anisotrope Maxwell-Verteilung (par = x, perp = y/z)

    vx = np.random.normal(loc=0.0, scale=vth_par, size=Np)
    return vx


"""
# Simulation parameters
# original parameters -- long!!!
# L = 20*np.pi #20*np.pi # Domain size
# DT = 0.005 # Time step
# NT = 50000  # Number of time steps
# doPlots = True
# NG = 320  # Number of grid cells
# N = NG * 20 # Number of particles
#  endoriginal parameters -- long!!!

# this is fast, but does not conserve energy in the end
# change parameters for better energy conservation
L = 2.5 * np.pi  # 20*np.pi #20*np.pi # Domain size
DT = 0.005 * 10  # 0.005 # Time step
NT = 500  # 50000  # Number of time steps
doPlots = True
NG = 40  # 80 #320  # Number of grid cells
PPC = 20  # number of particles per cell
N = NG * PPC  # total number of particles
"""

L = 2.5 * np.pi
DT = 0.005 * 10
NT=500
doPlots = True
NG = 40  # 80 #320  # Number of grid cells /gridpoints
PPC = 20  # number of particles per cell


"""
Possion Laplace Ableitung
Auxilliary vectors / Hilfs-
p = np.concatenate([np.arange(Np), np.arange(Np)])  
 Some indices up to N 0 bis np-1 und dann nochmal
Poisson is a diagonal matrix with -2 on the diag; -1 above and below used for \nabla^2
Poisson = sparse.spdiags(([1, -2, 1] * np.ones((1, NG - 1), dtype=int).T).T, [-1, 0, 1], NG - 1, NG - 1)
diags=[1, -2, 1] * np.ones((1, NG - 1) Für Jede Gridzelle-1 wird die Ableitung gebildet

spdiags(data, diags, m, n)
Poisson = Poisson.tocsc()
Faster code

d^2/d^2x^2+d^2/d^2y^2 phi
1D bedeuted f(x+h)+f(x-h)-2f(x)


"""




# Mode:
# 1 = N-Step
# 2 = Continous
# 3 = Flipbook
mode = 3


"""
PLots 
histEnergy, histPotE, histKinE, histMomentum, t = [], [], [], [], []
"""



solver_ref = Explicit_PIC_Solver(L, NG, PPC, DT)
solver_ref.xp, solver_ref.vp, solver_ref.B = initialize_two_stream1D(solver_ref.Lx, solver_ref.Np, solver_ref.B)
# Referenzen aktualisieren
solver_ref.species[0]["xp"] = solver_ref.xp
solver_ref.species[0]["vp"] = solver_ref.vp
solver_ref.species[0]["rho"] = solver_ref.rho
solver_ref.Ekin0 = np.sum(solver_ref.vp ** 2) * 0.5
solver_ref.step()