from .semi_implicit_particle_sim import  PIC_Solver
import numpy as np

"""
Initnitialisation of the Implicit Probelm of the electromagnetic two streams poblem.
"""

class twostream1D(PIC_Solver):
    def __init__(self,border=1,gridpoints=128,NPpCell=20,dt=0.1,):

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
                "name": "electron",
                "q": -1.0,
                "m": 1.0,
                "beta_mag_par": 0,
                "beta_mag_perp": 0,
                "beta": None,
                "NPpCell": NPpCell,
                "Np":Np
            },]


        super().__init__(dimension=1, dt=dt, stepssize=self.dx,border=(self.Lx,),gridNumbers=(self.Nx,),species=species )

        self.species[0]["xp"], self.species[0]["vp"], self.B = initialize_two_stream1D(self.Lx, self.species[0]["Np"], self.B)



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
    vp1 = 1 + amplitude * np.sin(2 * np.pi / Lx * xp1)
    vp2 = -1 - amplitude * np.sin(2 * np.pi / Lx * xp1)
    xp = np.concatenate([xp1, xp2])
    vp_x = np.concatenate([vp1, vp2])
    vp[0, :] = vp_x
    B[2, ...] = 1
    return xp, vp,B