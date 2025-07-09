from semi_implicit_particle_sim import  PIC_Solver
import numpy as np
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

        # Implicit Parameter
        self.theta = 0.5  # Implicitness parameter
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





        # Implicit Parameter
        self.theta = 0.5  # Implicitness parameter
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

        """Helper Variabeln nciht imme rneu definiert"""
        self.J_hat = np.zeros([3, self.Nx])
        self.rho_hat = np.zeros([self.Nx])
        self.E_theta=np.zeros([3, self.Nx])
        self.E_theta_p = np.zeros([3, self.Np])


        # In Simulation wurde nur in x Ebene Gerechnet
        """
        self.pos_p = np.random.uniform(0, self.Lx, self.Np)
        self.vstart_p = np.random.normal(0, 1, self.Np)
        self.vp[0,:]=self.vstart_p
        self.B[2,...]=1.
        """


        #self.xp = np.zeros([3, self.Np])
        self.vp = np.zeros([3, self.Np])
        self.xp_iter = np.zeros(self.Np)
        self.vp_iter = np.zeros([3, self.Np])

        self.xp, self.vp, self.B = initialize_two_stream1D(self.Lx, self.Np, self.vp, self.B, amplitude=0.01)
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

class PIC_Explicit1D():
    def __init__(self,border=1, gridpoints=128, NPpCell=20,dt=0.1):
        self.Nx = gridpoints
        self.dt = dt
        self.Lx = border
        self.dx = self.Lx / self.Nx
        self.NPpCell = NPpCell
        self.Np = self.Nx * self.NPpCell
        self.t = 0.0

        self.x = self.dx * np.arange(self.Nx)

        self.qDm = -1
        self.omega_p = 1
        self.epsilon_0 = 1
        self.charge = self.omega_p**2 / self.qDm * self.epsilon_0 * self.Lx / self.Np

        self.E = np.zeros([3, self.Nx])
        self.B = np.zeros([3, self.Nx])
        self.rho = np.zeros(self.Nx)

        self.vp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])

        self.xp, self.vp,self.B= initialize_two_stream1D(self.Lx, self.Np,self.vp,self.B,amplitude=0.01)

        self.Ekin0 = np.sum(self.vp ** 2) * 0.5



    def step(self):
        self.weight_rho()
        self.calc_E()

        self.force()
        self.boris(self.dt)
        self.step_x(self.dt)
        self.boundary()
        self.t += self.dt

    def step_x(self, dt_):
        self.xp += dt_ * self.vp[0, :]

    def boris(self, dt_):
        a = 0.5 * dt_ * self.qDm
        t_b = a * self.Bp
        s_b = 2 * t_b / (1 + self.dot(t_b, t_b))
        v_min = self.vp + a * self.Ep
        v_prime = v_min + self.cross(v_min, t_b)
        v_plus = v_min + self.cross(v_prime, s_b)
        self.vp = v_plus + a * self.Ep

    def dot(self, A, B):
        return np.sum(A * B, axis=0)

    def cross(self, A, B):
        C = np.zeros_like(A)
        C[0] = A[1]*B[2] - A[2]*B[1]
        C[1] = A[2]*B[0] - A[0]*B[2]
        C[2] = A[0]*B[1] - A[1]*B[0]
        return C

    def interpolation_rho_to_grid(self):
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx
            i = int(zeta)
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            self.rho[i] += 1 - diff
            self.rho[ip1] += diff

    def interpolation_to_part(self, part_force, grid_force):
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx
            i = int(zeta)
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            part_force[:, p] = (1 - diff) * grid_force[:, i] + diff * grid_force[:, ip1]

    def weight_rho(self):
        self.rho *= 0
        self.interpolation_rho_to_grid()
        self.rho -= self.Np / self.Nx
        self.rho *= 2 * self.NPpCell * self.charge / self.dx

    def force(self):
        self.interpolation_to_part(self.Ep, self.E)
        self.interpolation_to_part(self.Bp, self.B)

    def calc_E(self):
        rhohat = np.fft.rfft(self.rho)
        kx = 2 * np.pi / self.Lx * np.arange(rhohat.size)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.where(kx*kx > 0, rhohat / (1j * kx), 0.0)
        self.E[0, :] = np.fft.irfft(tmp)


    def boundary(self):
        self.xp = np.mod(self.xp, self.Lx)

    def CalcKinEnergery(self):
        return (0.5 * np.sum(self.vp ** 2) / self.Ekin0 * 100)

    def CalcEFieldEnergy(self):
        return (0.5 * np.sum(self.E ** 2)  )


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
    B[2, ...] = 0
    return xp, vp,B