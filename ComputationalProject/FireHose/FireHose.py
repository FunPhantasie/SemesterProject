import numpy as np
from semi_implicit_particle_sim import  PIC_Solver

class PIC3D(PIC_Solver):
    def __init__(self,border=25.0,gridpoints=10,NPpCell=1,dt=0.01):

        # Borders
        self.Lx = border  # Border length
        self.Ly = border
        self.Lz = border

        # Grid Points alongs Axis
        self.Nx = gridpoints  # Number of grid points
        self.Ny = gridpoints  # Number of grid points
        self.Nz = gridpoints

        self.totalN = 3 * self.Nx* self.Ny* self.Nz
        # Number per Cell
        self.NPpCell = NPpCell



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

        # In Simulation wurde nur in x Ebene Gerechnet
        """
        self.pos_p = np.random.uniform(0, self.Lx, self.Np)
        self.vstart_p = np.random.normal(0, 1, self.Np)
        self.vp[0,:]=self.vstart_p
        self.B[2,...]=1.
        """

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


# Simulationsparameter
N_cells = 64        # Anzahl der Gitterzellen
L = N_cells         # Länge des Gebietes (Zellabstand = 1 in Normierung)
N_p = 10000         # Anzahl Makropartikel
m_e = 1.0           # Elektronenmasse (normiert)
q_e = - N_cells/N_p # Elektronenladung so gewählt, dass mittlere Dichte 1 ist
epsilon0 = 1.0      # Permittivitätskonstante normiert auf 1

# Initialisierung der Teilchen
np.random.seed(0)                    # Saat für Reproduzierbarkeit
positions = np.random.rand(N_p) * L  # zufällige Anfangspositionen (0 bis L)
velocities = np.zeros(N_p)

# Zwei Strahlen: Hälfte der Teilchen mit +v0, Hälfte mit -v0
v0 = 3.0  # Driftgeschwindigkeit der Strahlen (z. B. 3 * Thermalgeschw.)
half = N_p // 2
velocities[:half] = v0
velocities[half:] = -v0

# Kleiner Perturbation zur Anregung einer Welle (Mode 1, cos-Förmig)
dv = 0.1 * v0  # Amplitude der Geschwindigkeitsstörung (10% von v0)
phase = 2*np.pi * positions / L
velocities[:half] += dv * np.cos(phase[:half])   # Beam 1: +cos-Störung
velocities[half:] -= dv * np.cos(phase[half:])   # Beam 2: -cos-Störung

firhose_case=PIC3D(border=25.0,gridpoints=10,NPpCell=1,dt=0.01)