"""
class PIC1D(PIC_Solver):
    def __init__(self, border=25.0, gridpoints=10, NPpCell=1, dt=0.01):
        super().__init__(dimension=1, dt=dt)
        self.Lx = border
        self.Nx = gridpoints
        self.NPpCell = NPpCell

        self.dx = self.Lx / self.Nx

        self.theta = 0.5
        self.Np = self.NPpCell * self.Nx
        self.charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Lx / self.Np

        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)

        self.rho = np.zeros([self.Nx])
        self.E = np.zeros([1, self.Nx])  # Only Ex
        self.B = np.zeros([1, self.Nx])  # Only Bz?

        self.vp = np.zeros([1, self.Np])
        self.Ep = np.zeros([1, self.Np])
        self.Bp = np.zeros([1, self.Np])
        self.xp = np.zeros([self.Np])

        self.J_hat = np.zeros([1, self.Nx])
        self.rho_hat = np.zeros([self.Nx])
        self.E_theta_p = np.zeros([1, self.Np])
        self.xp_iter = np.zeros([self.Np])
        self.vp_iter = np.zeros([1, self.Np])
    def deposit_charge(self,x_p,rho,ShapeFunction):
        # 1D
        rho*=0
        for particle_index in range(self.Np):
            x=x_p[particle_index]

            xn=(x/self.dx)
            ix = int(xn)
            #Arround The World
            #Muss Rho Volumes zuordnen
            for ax in [0,1]:
                grid_point_x = np.mod(ix + ax, self.Nx) #Entweder x0 oder x1

                # Weight based on linear distance (CIC)
                wx = 1 - abs(xn - (ix + ax))

                w = wx



                rho[grid_point_x]+=ShapeFunction(w)
        return rho


def ShaperParticle(self, x_p, prefaktor, ShapeFunction):
    # Validate prefaktor shape and assign helper
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

    # Initialize helper based on prefaktor type
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

                    if is_single_value:
                        helper[grid_x, grid_y, grid_z] += prefaktor * ShapeFunction(weight)
                    elif is_scalar:
                        helper[grid_x, grid_y, grid_z] += prefaktor[particle_index] * ShapeFunction(weight)
                    else:
                        helper[:, grid_x, grid_y, grid_z] += prefaktor[:, particle_index] * ShapeFunction(weight)

    return helper

        for particle_index in range(self.Np):
            x, y, z = x_p[:, particle_index]

            xn, yn, zn = (x / self.dx), (y / self.dy), (z / self.dz)
            ix, iy, iz = int(xn), int(yn), int(zn)
            # Arround The World
            # Muss Rho Volumes zuordnen
            for ax in [0, 1]:
                for by in [0, 1]:
                    for cz in [0, 1]:
                        grid_point_x = np.mod(ix + ax, self.Nx)  # Entweder x0 oder x1
                        grid_point_y = np.mod(iy + by, self.Ny)
                        grid_point_z = np.mod(iz + cz, self.Nz)
                        # Weight based on linear distance (CIC)
                        wx = 1 - abs(xn - (ix + ax))  # Kein Modulo macht alles kaputt
                        wy = 1 - abs(yn - (iy + by))
                        wz = 1 - abs(zn - (iz + cz))
                        w = wx * wy * wz
                        fieldp[:, particle_index] += field[:, grid_point_x, grid_point_y,
                                                     grid_point_z] * ShapeFunction(w)
        return fieldp
"""
