import numpy as np
from semi_implicit_particle_sim import  PIC_Solver
from analytics_plotting import run

class fireHose3D(PIC_Solver):
    def __init__(self,dt=0.01):

        dt = 0.5  # Normalized time step

        B0 = ([0.07906, 0.0, 0.0])


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
        self.NPpCell_e = 200#(200,1,1)
        self.NPpCell_i = 200#(200,1,1)


        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz




        # Total Particles
        self.Np_e = self.NPpCell_e * self.Nx * self.Ny * self.Nz
        self.Np_i=self.NPpCell_i * self.Nx * self.Ny * self.Nz




        # Grid and wavenumbers (Steps dx,dy,dz)
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        self.z = np.linspace(0, self.Lz, self.Nz, endpoint=False)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,indexing='ij')  # Meshgrid Discussion about Indexing
        #Particle Independent
        self.E = np.zeros([3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
        self.E_theta = np.zeros([3, self.Nx, self.Ny, self.Nz])

        self.B = np.zeros([3, self.Nx, self.Ny, self.Nz])  # Bz (2D)

        #Species Abhängig











        species = [{
            "name": "electron",
            "q": -1.0,
            "m": 1./25.0,
            "beta_mag_par": 8,
            "beta_mag_perp": 0.8,
            "beta": None,
            "NPpCell": self.NPpCell_e,
            "Np":self.Np_e
        },
        {
            "name": "ions",
            "q": 1.0,
            "m": 1.,
            "beta_mag_par": 0,
            "beta_mag_perp": 0,
            "beta": None,
            "NPpCell": self.NPpCell_i,
            "Np": self.Np_i
        },
        ]

        stepssize=(self.dx,self.dy,self.dz)
        border=(self.Lx,self.Ly,self.Lx)
        Np=(self.Np_e,self.Np_i)
        gridNumbers=(self.Nx,self.Ny,self.Nz)

        super().__init__(dimension=3, dt=dt,stepssize=stepssize,border=border,gridNumbers=gridNumbers,species=species)

        """
        Initialising
        Setting Up the Velocites and Energies        
        #vx,vy,vz Thermal Electrons
        #ux,uy,uz Drift Velcctiy Electons
        
        uth = 0.158 0.0316 # Thermal velocity in X
        vth = 0.05 0.0316 # Thermal velocity in Y
        wth = 0.05 0.0316 # Thermal velocity in Z
        u0 = 0.0 0.0 # Drift velocity in X
        v0 = 0.0 0.0 # Drift velocity in y
        w0 = 0.0 0.0 # Drift velocity in z
        """
        self.B[:] = np.reshape(B0, (3, 1, 1, 1))
        self.species[0]["xp"]=self.initialize_positions(self.species[0]["Np"])
        self.species[1]["xp"]=self.initialize_positions(self.species[1]["Np"])
        vth_perp = 0.0316  # normalized, z.B. v_perp = sqrt(beta_perp * B^2 / (2n))
        vth_par_e = vth_perp * np.sqrt(10)
        self.species[0]["vp"] = self.sample_maxwellian_anisotropic(vth_par_e, vth_perp, self.species[0]["Np"])

        """
        Tpar/Tperp=10
        (v_th_e_par/v_th_i)**2 =1
        """

        # Thermal velocity
        #v_th = np.sqrt(k_B * T / m)

        # Sample from 1D thermal (Maxwellian) distribution
        #velocities = np.random.normal(loc=0.0, scale=v_th, size=10000)

    def initialize_positions(self, Np):
        # Uniformly Generateed all Same Propbality
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

            helper = (np.zeros([3, self.Nx, self.Ny, self.Nz]) if is_vector else np.zeros([self.Nx, self.Ny, self.Nz]))



        # Process each particle
        for particle_index in range(Np):
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





"""
Choose Params
"""

dt = 0.01
t_end = 0.01
nsteps = 1
total_steps = int(t_end / dt)

firhose_case=fireHose3D(dt=dt)
ShowSingleStep=None
run(firhose_case,  total_steps, dt,ShowSingleStep=ShowSingleStep)
