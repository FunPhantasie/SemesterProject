from semi_implicit_particle_sim import  PIC_Solver
import numpy as np

class PIC1D(PIC_Solver):
    def __init__(self,border=1,gridpoints=128,NPpCell=20,dt=0.1,):

        # Borders
        self.Lx = border  # Border length

        # Grid Points alongs Axis
        self.Nx = gridpoints  # Number of grid points

        self.totalN=3*self.Nx

        # Number per Cell
        self.NPpCell = NPpCell



        self.dx = self.Lx / self.Nx






        # Total Particles
        self.Np = self.NPpCell * self.Nx



        super().__init__(dimension=1, dt=dt, steps=self.dx,border=(self.Lx,),Np=self.Np,gridNumbers=(self.Nx,) )

        # Grid and wavenumbers (Steps dx,dy,dz)
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)


        # Grid Fields and Densities
        self.rho = np.zeros(self.Nx)
        self.E = np.zeros([3, self.Nx])  # Ex, Ey Its E but bc only in forward time its used no one cares
        self.B = np.zeros([3, self.Nx])  # Bz (2D)
        """
        Using Yee Scheme E and B are Ofset E along the axis of 1/2 and B to the Center Face
        """
        # Initialize the Particles Global Positions and Velocities
        self.Fp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])

        """Iteration Variabeln Variabeln nicht imme rneu definiert"""
        self.J_hat = np.zeros([3, self.Nx])
        self.rho_hat = np.zeros([self.Nx])
        self.E_theta=np.zeros([3, self.Nx])
        self.E_theta_p = np.zeros([3, self.Np])




        #self.xp = np.zeros([3, self.Np])
        self.vp = np.zeros([3, self.Np])
        self.xp_iter = np.zeros(self.Np)
        self.vp_iter = np.zeros([3, self.Np])
        self.rho_iter = np.zeros(self.Nx)
        self.E_prev = np.zeros([3, self.Nx])

        self.xp, self.vp, self.B = initialize_two_stream1D(self.Lx, self.Np, self.vp, self.B)
        # Solve Method
        # self.calculus = Integrator(step_method, self.dgl_eq)
        # self.fourious = FourierSolver(dimension)

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



def initialize_two_stream1D(Lx, Np,vp,B, amplitude=0.01):
    """
    Initialize particle positions and velocities for a two-stream instability.

    Args:
        Lx (float): System length
        Np (int): Total number of particles
        amplitude (float): Amplitude of velocity perturbation

    Returns:
        tuple: (xp, vp_x) where xp is particle positions and vp_x is x-component of velocities
    """
    xp1 = 2 * Lx / Np * np.arange(Np // 2)
    xp2 = 2 * Lx / Np * np.arange(Np // 2)
    vp1 = 1 + amplitude * np.sin(2 * np.pi / Lx * xp1)
    vp2 = -1 - amplitude * np.sin(2 * np.pi / Lx * xp1)
    xp = np.concatenate([xp1, xp2])
    vp_x = np.concatenate([vp1, vp2])
    vp[0, :] = vp_x
    B[2, ...] = 1
    return xp, vp,B