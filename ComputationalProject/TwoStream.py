import numpy as np

from Simulation.explicit_particle_sim import Explicit_PIC_Solver
from Simulation.Semi_Implicit import IPIC_Solver as Semi_PIC_Solver
from Simulation.solution_explcit_pic import TwoStreamPIC1D as Solution_PIC
from scipy import sparse
from scipy.sparse import linalg
from Analytics.AnalyticsOfNStep import run_save_steps as run_nstep

#from Analytics.RenderManager import CallItRenderer
#from Analytics.Animator import run_continuous
#from Analytics.Animator import run_flipbook

import matplotlib.pyplot as plt

class Init_SemiImplicit(Semi_PIC_Solver):
    def __init__(self,L=1,NG=1,PPC=20,DT=0.1,ES=True):
        #Parameter Conditions
        self.Nx = NG  # gridpoints
        self.dt = DT
        self.Lx = L  # Border
        self.dx = self.Lx / self.Nx
        #Npp per Cell is Electons in Species
        self.t = 0.0

        self.xg = np.linspace(0, self.Lx - self.dx, self.Nx) + 0.5 * self.dx  # grid at cell centers

        """--------Fields--------"""

        self.totalN=self.Nx #Total Number of Gridppoints (3 Could be Wrong)
        self.NPpCell = PPC  # NPpCell
        self.Np = self.Nx * self.NPpCell


        # Resulting Connected Conditions

        self.Eg = np.zeros([ self.Nx])  # E[0]
        self.rhog =np.zeros([self.Nx])
        self.Jg=np.zeros([ self.Nx]) #Global
        self.Pg=np.zeros([ self.Nx])

        self.rhog_hat =np.zeros([self.Nx])
        self.Jg_hat =np.zeros([self.Nx])
        self.B = np.zeros([ self.Nx])  # B[2]
        self.E_theta = np.zeros([ self.Nx])

        """
        All The Fields and Moments
        """
        # Physical constants
        self.c = 1
        self.pi = np.pi
        self.omega_p = 1.  # Plasma Freq.
        self.epsilon_0 = 1.  # Copied Convenient normalization
        # Resulting Connected Conditions


        self.omega_p = 1



        #self.Volume = np.prod(border)
        #self.GridVolume = np.prod(gridNumbers)
        #self.weight = 1 / (self.Nx * self.dx)
        self.Np = PPC * self.Nx  # Total Particles

        #sp["beta"] = sp["q"] * self.dt / (2 * sp["m"] * self.c) * self.weight
        #sp["q"] *= self.weight
        el={
                "name": "e",
                "q":-1,
                "qDm": -1,
                "charge": 0,
                "NPpCell": PPC,
                "Np":self.Np,
                "xp":None,
                "vp": None,
            }
        el["charge"] = self.omega_p ** 2 / (el["qDm"] * el["Np"] / self.Lx)
        self.back_charge_density = -el["charge"] * self.Np / self.Lx
        species=[el,]

        super().__init__(dimension=1, stepssize=self.dx,border=(self.Lx,),gridNumbers=(self.Nx,),species=species )

    def ShapeFunction(self,x_p,Np):
        xn = (x_p / self.dx)
        ix = np.floor(xn-0.5).astype(int)

        wx = 1 - abs(xn - (ix + 0.5))

        indexes = np.concatenate((ix, ix+1))
        #ix = np.mod(ix, self.Nx)
        NG=self.Nx
        indexes[indexes < 0] += NG
        indexes[indexes > NG - 1] -= NG
        weights= np.concatenate((wx,1-wx))

        p = np.arange(Np)
        prow = np.concatenate((p, p))

        mat = sparse.csc_matrix((weights, (prow, indexes)), shape=(Np, self.Nx))


        return mat





def normalized_initialize_two_stream1D(Lx, Np,VT=0.005,V0=0.05, XP1=0.01,mode=1,seed=42):
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
    B  = np.zeros([3, Np])
    vp = np.zeros([ Np])
    xp = np.linspace(0, Lx - Lx / Np, Np)
    xp += XP1 * (Lx / Np) * np.sin(2 * np.pi * xp / Lx * mode) #Pertubation in Postion Space
    xp = np.mod(xp, L)
    rng = np.random.default_rng(seed)
    vp_x = VT * (1 - VT ** 2) ** (-0.5) * rng.standard_normal(Np) #Pertubation Velocity
    #vx = np.random.normal(loc=0.0, scale=vth_par, size=Np) Should work same as randn*vp
    #Its done Relativistic momentum:  p=γmv m=1
    pm = np.arange(Np)
    pm = 1 - 2 * np.mod(pm + 1, 2) #-1,1,-1,1
    vp_x += pm * (V0 * (1 - V0 ** 2) ** (-0.5)) #Base Velocity One BAckwards One Forward
    vp = vp_x
    # RNG





    return xp, vp


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


L = 2.5 * np.pi
NG = 40  # 80 #320  # Number of grid cells /gridpoints
PPC = 20  # number of particles per cell

DT = 0.005 * 10
NT=500


params_class=dict(L=L, NG=NG, PPC=PPC, DT=DT,ES=False)
norm_inno = Solution_PIC( **params_class) #'Innocenti'
#own_solver = Explicit_PIC_Solver(**params_class) # Me
own_solver= Init_SemiImplicit(**params_class)
params_init = dict(VT=0.0000001, V0=0.5, XP1=1.0, mode=1,seed=42)
xp_helper,vp_helper = normalized_initialize_two_stream1D(L, NG*PPC,  **params_init)
#own_solver.xp, own_solver.vp= xp_helper,vp_helper
own_solver.species[0]["xp"],own_solver.species[0]["vp"]=xp_helper,vp_helper
norm_inno.xp, norm_inno.vp= xp_helper, vp_helper
own_solver.step()
norm_inno.step()

# ======== SETTINGS ========
stepview = 50   # show plots every 'stepview' steps
# ==========================



"""Plotting"""

times = []

# Historien
energy_sol, kin_sol, mom_sol = [], [], []
energy_ref, kin_ref, mom_ref = [], [], []

rho_sol_hist, rho_ref_hist = [], []
E_sol_hist,   E_ref_hist   = [], []
J_sol_hist,   J_ref_hist   = [], []
P_sol_hist,   P_ref_hist   = [], []

for n in range(NT):
    # Schritt machen
    norm_inno.step()
    own_solver.step()

    # Zeit
    times.append(norm_inno.t)

    # Energies & Momentum
    energy_sol.append(norm_inno.calcEnergy())
    kin_sol.append(norm_inno.calcKinEnergy())
    mom_sol.append(norm_inno.calcMomentum())

    energy_ref.append(own_solver.calcEnergy())
    kin_ref.append(own_solver.calcKinEnergy())
    mom_ref.append(own_solver.calcMomentum())

    rho_sol_hist.append(norm_inno.rho.copy())
    J_sol_hist.append(norm_inno.J.copy())
    E_sol_hist.append(norm_inno.Eg.copy())
    P_sol_hist.append(norm_inno.P_exx.copy())
    electrons = own_solver.species[0]

    #print(np.shape(electrons["rho"].copy))
    rho_ref_hist.append(electrons["rho"].copy())

    J_ref_hist.append(own_solver.Jg.copy())   # nur x-Komponente
    P_ref_hist.append(own_solver.Pg.copy())
    E_ref_hist.append(own_solver.Eg.copy())

    # ---- LIVE VIEW every 'stepview' steps ----
    if (n + 1) % stepview == 0 or n==0:
        # --------- PLOTTEN ---------
        fig, axs = plt.subplots(3, 2, figsize=(12, 13))
        """
        # Energie
        axs[0,0].plot(times, energy_sol, label="Solution Energy")
        axs[0,0].plot(times, energy_ref, "--", label="Implicit Own Energy")
        axs[0,0].set_title("Field Energy")
        axs[0,0].set_xlabel("time")
        axs[0,0].set_ylabel("Energy")
        axs[0,0].legend()
        axs[0,0].set_xlim(0,NT*DT)

        # Kinetische Energie
        axs[0,1].plot(times, kin_sol, label="Solution Kinetic")
        axs[0,1].plot(times, kin_ref, "--", label="Implicit Own Kinetic")
        axs[0,1].set_title("Kinetic Energy")
        axs[0,1].set_xlabel("time")
        axs[0,1].set_ylabel("Kinetic Energy")
        axs[0,1].legend()
        axs[0, 1].set_xlim(0, NT * DT)
        """

        # x–v Phase Space (Solution)
        axs[0,0].scatter(norm_inno.xp, norm_inno.vp, s=2, alpha=0.6, label="Solution")
        axs[0,0].set_title("Phase space (x–v) — Solution")
        axs[0,0].set_xlabel("x")
        axs[0,0].set_ylabel("v")
        axs[0,0].legend()

        # x–v Phase Space (Reference)

        axs[0,1].scatter(own_solver.species[0]["xp"], own_solver.species[0]["vp"], s=2, alpha=0.6, label="Implicit Own")
        axs[0,1].set_title("Phase space (x–v) — Implicit Own")
        axs[0,1].set_xlabel("x")
        axs[0,1].set_ylabel("v")
        axs[0,1].legend()








        # E-Feld (letzter Snapshot) — ersetzt Jx-Panel
        axs[1,0].plot(norm_inno.xg, E_sol_hist[-1], label="Solution E")
        axs[1,0].plot(own_solver.xg, E_ref_hist[-1], "--", label="Implicit Own E")
        axs[1,0].set_title("Electric field E (final)")
        axs[1,0].set_xlabel("x")
        axs[1,0].set_ylabel("E")
        axs[1,0].legend()
        # Momentum
        """
        axs[1,0].plot(times, mom_sol, label="Solution Momentum")
        axs[1,0].plot(times, mom_ref, "--", label="Implicit Own Momentum")
        axs[1,0].set_title("Momentum")
        axs[1,0].set_xlabel("time")
        axs[1,0].set_ylabel("Momentum")
        axs[1,0].legend()
        axs[1, 0].set_xlim(0, NT * DT)
        """
        # rho (letzter Snapshot)
        axs[1,1].plot(norm_inno.xg, rho_sol_hist[-1], label="Solution ρ")
        axs[1,1].plot(own_solver.xg, rho_ref_hist[-1], "--", label="Implicit Own ρ")
        axs[1,1].set_title("Charge density ρ (final)")
        axs[1,1].set_xlabel("x")
        axs[1,1].set_ylabel("ρ")
        axs[1,1].legend()

        #J (letzter Snapshot)
        axs[2,0].plot(norm_inno.xg, J_sol_hist[-1], label="Solution Jx")
        axs[2,0].plot(own_solver.xg, J_ref_hist[-1],  label="Implicit Own Jx")
        axs[2,0].set_title("Current J (final)")
        axs[2,0].legend()



        # P (letzter Snapshot)
        axs[2,1].plot(norm_inno.xg, P_sol_hist[-1], label="Solution Pxx")
        axs[2,1].plot(own_solver.xg, P_ref_hist[-1], "--", label="Implicit Own Pxx")
        axs[2,1].set_title("Stress Tensor P (final)")
        axs[2,1].set_xlabel("x")
        axs[2,1].set_ylabel("Pxx")
        axs[2,1].legend()

        plt.tight_layout()
        plt.show()
        plt.close(fig)   # close so the loop continues without piling windows
