import numpy as np

from Simulation.explicit_particle_sim import Explicit_PIC_Solver
from Simulation.semi_implicit_particle_sim import PIC_Solver

from Analytics.AnalyticsOfNStep import run as run_nstep

from Analytics.RenderManager import CallItRenderer
from Analytics.Animator import run_continuous
from Analytics.Animator import run_flipbook

"""
Initnitialisation of the Implicit Probelm of the electromagnetic two streams poblem.

Choose Params
"""



class twostream1D(PIC_Solver):
    def __init__(self,border=1,gridpoints=1,NPpCell=20,dt=0.1,):

        #Parameter Conditions
        self.Lx = border  # Plasma Space/Borders
        self.Nx = gridpoints  # Number of grid points
        self.totalN=3*self.Nx #Total Number of Gridppoints (3 Could be Wrong)



        # Resulting Connected Conditions
        self.dx = self.Lx / self.Nx

        self.E = np.zeros([3, self.Nx])  # E[0]
        self.B = np.zeros([3, self.Nx])  # B[2]
        self.E_theta = np.zeros([3, self.Nx])

        """
        All The Fields and Moments
        """
        Np = NPpCell * self.Nx  # Total Particles

        species=[{
                "name": "e",
                "q": -1.0,
                "m": 1.0,
                "beta_mag_par": 0,
                "beta_mag_perp": 0,
                "beta": None,
                "NPpCell": NPpCell,
                "Np":Np
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
            ix= int(xn)
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

def initialize_two_stream1D(Lx, Np,B, amplitude=0.01):
    """
    Initialize particle positions and velocities for a two-stream instability.

    Args:
        Lx (float): System length
        Np (int): Total number of particles
        amplitude (float): Amplitude of velocity perturbation

    Returns:
        tuple: (xp, vp_x) where xp is particle positions and vp_x is x-component of velocities
    """


    vp = np.zeros([3, Np])
    xp1 = 2 * Lx / Np * np.arange(Np // 2)
    xp2 = 2 * Lx / Np * np.arange(Np // 2)
    vth=0.005

    vp1 = 0.05 + amplitude * np.sin(2 * np.pi / Lx * xp1)+sample_maxwellian_anisotropic(vth,Np//2)
    vp2 = -0.05 - amplitude * np.sin(2 * np.pi / Lx * xp1)+sample_maxwellian_anisotropic(vth,Np//2)
    xp = np.concatenate([xp1, xp2])
    vp_x = np.concatenate([vp1, vp2])
    vp[0, :] = vp_x
    #B[2, ...] = 1
    return xp, vp,B
def sample_maxwellian_anisotropic(vth_par, Np):
    # Sampling für anisotrope Maxwell-Verteilung (par = x, perp = y/z)
    return 0
    vx = np.random.normal(loc=0.0, scale=vth_par, size=Np)
    return vx




border = 1
gridpoints = 64 #Dx is border/grdipoints
NPpCell = 20
dt = 0.05
t_end = 0.25
total_steps = int(t_end / dt)




# Mode:
# 1 = N-Step
# 2 = Continous
# 3 = Flipbook
mode = 3

nsteps = 2 #For N Step Debugging Attention No Saving


# Initialisiere Solver
solver_test = twostream1D(border, gridpoints, NPpCell, dt)
solver_test.species[0]["xp"], solver_test.species[0]["vp"], solver_test.B = initialize_two_stream1D(solver_test.Lx, solver_test.species[0]["Np"], solver_test.B)

solver_ref = Explicit_PIC_Solver(border, gridpoints, NPpCell, dt)
solver_ref.xp, solver_ref.vp, solver_ref.B = initialize_two_stream1D(solver_ref.Lx, solver_ref.Np, solver_ref.B)
# Referenzen aktualisieren
solver_ref.species[0]["xp"] = solver_ref.xp
solver_ref.species[0]["vp"] = solver_ref.vp
solver_ref.species[0]["rho"] = solver_ref.rho
solver_ref.Ekin0 = np.sum(solver_ref.vp ** 2) * 0.5




print("Chosen Parameters:")
print(f"dx: {border/gridpoints:.6f}")
print("Starting Parameters for Explicit Solver:")

solver_ref.weight_rho()
print(f"Normalization Charge (qp): {solver_ref.charge:.6f}")
print("Density (rho):")
print(solver_ref.rho)
print(solver_ref.species[0]["vp"][0])

# ---------------------
# Kompakte Parameterpakete
# ---------------------

# Simulation & Auflösung
Nx_test = solver_test.Nx
Np_test = solver_test.species[0]["Np"]
Np_ref = solver_ref.Np
sim_params = (total_steps, t_end, Nx_test, Np_test, Np_ref)

# Plot-Grenzen
plot_params = {
    "stream_limits": (-0.5, 0.5),
    "xlims_implicit": (0, solver_test.Lx),
    "xlims_explicit": (0, solver_ref.Lx),
    "moments_limits": (-1, 1),
    "energy_limits": (0, 5),
    "frame_duration_ms": 100
}

# Display Result / Calculating

if mode == 1:
    run_nstep(solver_test, solver_ref, nsteps, nsteps*dt)
elif mode == 2:
    data_test = CallItRenderer(solver_test, total_steps, "TwoStreamRender/Implicit",step=False)
    data_ref = CallItRenderer(solver_ref, total_steps,"TwoStreamRender/Explicit",step=True)

    run_continuous(data_test,data_ref,sim_params, plot_params)
elif mode == 3:
    data_test = CallItRenderer(solver_test, total_steps, "TwoStreamRender/Implicit",step=False)
    data_ref = CallItRenderer(solver_ref, total_steps,"TwoStreamRender/Explicit",step=True)

    run_flipbook(data_test,data_ref,sim_params, plot_params)
else:
    raise ValueError("No Valid Mode: Choose 1, 2 or 3")
