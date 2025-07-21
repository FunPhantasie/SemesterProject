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
        self.NPpCell = NPpCell # Particle per Cell



        # Resulting Connected Conditions
        self.dx = self.Lx / self.Nx
        self.Np = self.NPpCell * self.Nx  # Total Particles

        """
        All The Fields and Moments
        """

        # Grid Fields and Densities
        # Only Fields in Scalar Direction non zero used, all updated
        self.rho_s = np.zeros(self.Nx)
        self.E = np.zeros([3, self.Nx])  #E[0]
        self.B = np.zeros([3, self.Nx])  # B[2]


        # Initialize the Particles Global Positions and Velocities
        self.Fp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])

        """Iteration Variabeln In Current Implementation not really Iterated"""
        self.J_hat_s = np.zeros([3, self.Nx])
        self.rho_hat_s = np.zeros([self.Nx])
        self.E_theta = np.zeros([3, self.Nx])
        self.E_theta_p = np.zeros([3, self.Np])
        self.xp_iter_s = np.zeros(self.Np)
        self.vp_iter_s = np.zeros([3, self.Np])
        self.rho_iter_s = np.zeros(self.Nx)
        self.E_prev = np.zeros([3, self.Nx])

        """
        Initiale t=0 Conditions
        """

        self.xp_s, self.vp_s, self.B = initialize_two_stream1D(self.Lx, self.Np, self.B)






        species=[{
                "name": "electron",
                "q": -1.0,
                "m": 1.0,
                "beta": None,
                "xp": self.xp_s.copy(),
                "vp": self.vp_s.copy(),
                "xp_iter":self.xp_iter_s.copy(),
                "vp_iter":self.vp_iter_s.copy(),
                "rho": self.rho_s.copy(),
                "rho_hat": self.rho_hat_s.copy(),
                "J_hat": self.J_hat_s.copy(),
            },]


        super().__init__(dimension=1, dt=dt, stepssize=self.dx,border=(self.Lx,),Np=self.Np,gridNumbers=(self.Nx,),species=species )



    def ShaperParticle(self, x_p, prefaktor, ShapeFunction,toParticle=False):
        # Validate prefaktor shape and assign helper
        if toParticle:
            helper = np.zeros([3, self.Np])

        # Initialize helper based on prefaktor type

        else:
            if np.isscalar(prefaktor):
                is_scalar = True
                is_vector = False
                is_single_value = True
            else:
                is_scalar = prefaktor.shape == (self.Np,)
                is_vector = prefaktor.shape == (3, self.Np)
                is_single_value = prefaktor.shape == (1,)
            if not (is_scalar or is_vector):
                raise ValueError(f"prefaktor shape {prefaktor.shape} is invalid. Expected (Np,) or (3, Np).")

            helper = (np.zeros([3, self.Nx]) if is_vector else np.zeros(self.Nx))



        # Process each particle
        for particle_index in range(self.Np):
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