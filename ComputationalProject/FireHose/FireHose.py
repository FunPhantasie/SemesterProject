import numpy as np
from semi_implicit_particle_sim import  PIC_Solver

class fireHose3D(PIC_Solver):
    def __init__(self,dt=0.01):
        B0 = np.array([0.07906, 0.0, 0.0])
        dt = 0.5  # Normalized time step





        # Constants
        k_B = 1.380649e-23  # Boltzmann constant [J/K]
        T = 1e5  # Temperature in Kelvin
        m = 1.67e-27  # Mass of a proton [kg]

        #\beta_{\parallel} = \frac{n k_B T_{\parallel}}{B^2 / (2\mu_0)} = \frac{2 \mu_0 n k_B T_{\parallel}}{B^2}



        # Borders
        self.Lx = 300  # Border length
        self.Ly = 2
        self.Lz = 2

        # Grid Points alongs Axis
        self.Nx = 300  # Number of grid points
        self.Ny = 1  # Number of grid points
        self.Nz = 1

        self.totalN = 3 * self.Nx* self.Ny* self.Nz
        # Number per Cell
        self.NPpCell_e = 200
        self.NPpCell_i = 200


        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz


        super().__init__(dimension=3, dt=dt,steps=(self.dx,self.dy,self.dz))


        # Total Particles
        self.Np = self.NPpCell * self.Nx * self.Ny * self.Nz

        # Constants
        self.charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Lx / self.Np  # particle charge
        # self.charge = self.epsilon_0 * self.omega_p**2 * self.Lx**3 / self.Np

        # Grid and wavenumbers (Steps dx,dy,dz)
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        self.z = np.linspace(0, self.Lz, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,indexing='ij')  # Meshgrid Discussion about Indexing

        # Grid Fields and Densities
        self.rho = np.zeros([self.Nx, self.Ny, self.Nz])
        self.E = np.zeros([3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
        self.B = np.zeros([3, self.Nx, self.Ny, self.Nz])  # Bz (2D)
        """
        Using Yee Scheme E and B are Ofset E along the axis of 1/2 and B to the Center Face
        """
        # Initialize the Particles Global Positions and Velocities
        self.vp = np.zeros([3, self.Np])
        self.Fp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])
        self.xp = np.zeros([3, self.Np])

        """Helper Variabeln nciht imme rneu definiert"""
        self.J_hat = np.zeros([3, self.Nx, self.Ny, self.Nz])
        self.rho_hat = np.zeros([self.Nx, self.Ny, self.Nz])
        self.E_theta=np.zeros([3, self.Nx, self.Ny, self.Nz])
        self.E_theta_p = np.zeros([3, self.Np])

        self.xp_iter = np.zeros([3, self.Np])
        self.vp_iter = np.zeros([3, self.Np])

        """Setting Up the Velocites and Energies"""


        # Thermal velocity
        v_th = np.sqrt(k_B * T / m)

        # Sample from 1D thermal (Maxwellian) distribution
        velocities = np.random.normal(loc=0.0, scale=v_th, size=10000)






        species = [{
            "name": "electron",
            "q": -1.0,
            "m": 1./25.0,
            "beta": None,
            "xp": self.xp.copy(),
            "vp": self.vp_s.copy(),
            "xp_iter": self.xp_iter_s.copy(),
            "vp_iter": self.vp_iter_s.copy(),
            "rho": self.rho_s.copy(),
            "rho_hat": self.rho_hat_s.copy(),
            "J_hat": self.J_hat_s.copy(),
        },
        {
            "name": "ions",
            "q": 1.0,
            "m": 1.,
            "beta": None,
            "xp": self.xp_s.copy(),
            "vp": self.vp_s.copy(),
            "xp_iter": self.xp_iter_s.copy(),
            "vp_iter": self.vp_iter_s.copy(),
            "rho": self.rho_s.copy(),
            "rho_hat": self.rho_hat_s.copy(),
            "J_hat": self.J_hat_s.copy(),
        },
        ]

    def initialize_positions(self, Np):
        # Gleichmäßig verteilt in der Box (random)
        return np.array([
            np.random.uniform(0, self.Lx, Np),
            np.random.uniform(0, self.Ly, Np),
            np.random.uniform(0, self.Lz, Np)
        ])

    def sample_maxwellian_anisotropic(self, vth_par, vth_perp, Np):
        # Sampling für anisotrope Maxwell-Verteilung (par = x, perp = y/z)
        vx = np.random.normal(loc=0.0, scale=vth_par, size=Np)
        vy = np.random.normal(loc=0.0, scale=vth_perp, size=Np)
        vz = np.random.normal(loc=0.0, scale=vth_perp, size=Np)
        return np.vstack((vx, vy, vz))

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

            helper = (np.zeros([3, self.Nx, self.Ny, self.Nz]) if is_vector else np.zeros([self.Nx, self.Ny, self.Nz]))



        # Process each particle
        for particle_index in range(self.Np):
            # Particle position in grid coordinates
            x, y, z = x_p[:, particle_index]

            xn, yn, zn = (x / self.dx), (y / self.dy), (z / self.dz)
            ix, iy, iz = int(xn), int(yn), int(zn)
            # Arround The World
            # Muss Rho Volumes zuordnen

            # Compute weights for all 8 grid points at once
            for ax in [0, 1]:
                for by in [0, 1]:
                    for cz in [0, 1]:
                        # Periodic boundary conditions
                        grid_x = np.mod(ix + ax, self.Nx)
                        grid_y = np.mod(iy + by, self.Ny)
                        grid_z = np.mod(iz + cz, self.Nz)

                        # Weight based on linear distance (CIC)
                        wx = 1 - abs(xn - (ix + ax))
                        wy = 1 - abs(yn - (iy + by))
                        wz = 1 - abs(zn - (iz + cz))
                        weight = wx * wy * wz

                        # Apply shape function and update grid
                        if toParticle:
                            helper[:, particle_index] += prefaktor[:, grid_x, grid_y,
                                                         grid_z] * ShapeFunction(weight)
                        elif is_single_value:
                            helper[grid_x, grid_y, grid_z] += prefaktor * ShapeFunction(weight)
                        elif is_scalar:
                            helper[grid_x, grid_y, grid_z] += prefaktor[particle_index] * ShapeFunction(weight)
                        else:
                            helper[:, grid_x, grid_y, grid_z] += prefaktor[:, particle_index] * ShapeFunction(weight)

        return helper




firhose_case=fireHose3D(dt=0.01)