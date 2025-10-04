import numpy as np
from scipy import sparse
from scipy.sparse import linalg

"""
Solver with the same Scheme as the Electro Statitic Code for two Streams.
In Case of no normalisation of Grid Charge the E field is the same as in the implicit Method.
"""
class Explicit_PIC_Solver():

    def __init__(self, L=1, NG=128, PPC=20, DT=0.1,ES=True):
        self.Nx = NG #gridpoints
        self.dt = DT
        self.Lx = L #Border
        self.dx = self.Lx / self.Nx
        self.NPpCell = PPC #NPpCell
        self.Np = self.Nx * self.NPpCell
        self.t = 0.0

        self.xg = np.linspace(0, self.Lx - self.dx, self.Nx) + 0.5 * self.dx# grid at cell centers
        """
        Plasma Params
        N = NG * PPC  # total number of particles
        WP = 1.  # Plasma frequency
        QM = -1.  # Charge/mass ratio; normalized to to electron mass
        V0 = 0.5  # 0.9 # Stream velocity
        VT = 0.0000001  # Thermal speed
        """
        _Poisson = sparse.spdiags(([1, -2, 1] * np.ones((1, self.Nx - 1), dtype=int).T).T, \
                         [-1, 0, 1], self.Nx - 1, self.Nx - 1)
        self.Poisson = _Poisson.tocsc()
        self.qDm = -1
        self.omega_p = 1
        self.charge=self.omega_p ** 2 / (self.qDm * self.Np / self.Lx)
        self.back_charge_density=-self.charge * self.Np / self.Lx
        # olde self.charge = self.qDm * 1/(self.Nx * self.dx) #self.qDm/(1) *self.dx/self.Np*self.Lx

        self.Eg = np.zeros([3, self.Nx])
        self.B = np.zeros([3, self.Nx])
        self.rho = np.zeros(self.Nx)
        self.xp =np.zeros( self.Np)
        self.vp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])

        static = ES
        self.magnetic_field=False
        if static:
            self.calc_E = self.calc_E_static
        else:
            self.calc_E = self.calc_E_Maxwell
        if self.magnetic_field:
            self.updateVelcotiy=self.boris
        else:
            self.updateVelcotiy = self.uvel




    def step(self):
        self.weight_rho()
        #Electro Magnetic
        # ------------------#
        self.weight_J()
        self.weightStress()
        #------------------#
        self.calc_E()

        if self.magnetic_field: self.calc_B()
        self.force()
        self.updateVelcotiy(self.dt)
        self.step_x(self.dt)
        self.boundary()
        self.t += self.dt


    def step_x(self, dt_):

        self.xp += dt_ * self.vp[0, :]
    def uvel(self, dt_):
        self.vp +=  self.qDm * self.Ep * dt_



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
        C[0] = A[1] * B[2] - A[2] * B[1]
        C[1] = A[2] * B[0] - A[0] * B[2]
        C[2] = A[0] * B[1] - A[1] * B[0]
        return C

    def interpolation_rho_to_grid(self):
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx-0.5 #Cell Centers
            i = np.floor(zeta).astype(int) #Int rundet zur 0 bei neg numbers
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            self.rho[i] += 1 - diff
            self.rho[ip1] += diff

    def interpolation_to_part(self, part_force, grid_force):
            for p in range(self.Np):
                zeta = self.xp[p] / self.dx-0.5 #Cell Centers
                i = np.floor(zeta).astype(int) #Int rundet zur 0 bei neg numbers
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            part_force[:, p] = (1 - diff) * grid_force[:, i] + diff * grid_force[:, ip1]

    def weight_rho(self):
        self.rho *= 0
        self.interpolation_rho_to_grid()
        self.rho*=self.charge/self.dx
        self.rho+=self.back_charge_density

    def force(self):
        self.interpolation_to_part(self.Ep, self.Eg)
        self.interpolation_to_part(self.Bp, self.B)

    def weight_J(self):
        self.J = np.zeros([3, self.Nx])
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx - 0.5  # Cell Centers
            i = np.floor(zeta).astype(int)
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            for d in range(3):
                self.J[d, i] += (1 - diff) * self.vp[d, p]
                self.J[d, ip1] += diff * self.vp[d, p]
        self.J *= self.charge/self.dx
    def weightStress(self):
        self.P = np.zeros([3, self.Nx])
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx - 0.5  # Cell Centers
            i = np.floor(zeta).astype(int)
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            for d in range(3):
                self.P[d, i] += (1 - diff) * self.vp[d, p]*self.vp[d, p]
                self.P[d, ip1] += diff * self.vp[d, p]*self.vp[d, p]
        self.P *= self.charge / self.dx
    def calc_E_static(self):
        """
        #---Weicht ab--
        rhohat = np.fft.rfft(self.rho)
        kx = 2 * np.pi / self.Lx * np.arange(rhohat.size)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.where(kx * kx > 0, rhohat / (1j * kx), 0.)
        self.Eg[0, :] = np.fft.irfft(tmp)
        """
        Phi = linalg.spsolve(self.Poisson, -self.dx ** 2 * self.rho[0:self.Nx - 1])
        Phi = np.concatenate((Phi, [0])) #fix BC
        self.Eg[0, :] = (np.roll(Phi, 1) - np.roll(Phi, -1)) / (2 * self.dx)



    def calc_E_Maxwell(self):
        # Faraday's law: dE/dt = curl B - J
        curl_B = np.zeros_like(self.Eg)
        curl_B[1, 1:-1] = (self.B[2, 2:] - self.B[2, :-2]) / (2 * self.dx)
        curl_B[2, 1:-1] = -(self.B[1, 2:] - self.B[1, :-2]) / (2 * self.dx)

        self.Eg[:, 1:-1] += self.dt * ( -  self.J[:, 1:-1])

        #self.Eg[:, 1:-1] += self.dt * (curl_B[:, 1:-1] -  self.J[:, 1:-1])
    def calc_B(self):
        # dB/dt = - curl E
        curl_E = np.zeros_like(self.B)
        curl_E[1, 1:-1] = -(self.Eg[2, 2:] - self.Eg[2, :-2]) / (2 * self.dx)
        curl_E[2, 1:-1] = (self.Eg[1, 2:] - self.Eg[1, :-2]) / (2 * self.dx)
        self.B[:, 1:-1] += self.dt * curl_E[:, 1:-1]

    def boundary(self):
        self.xp = np.mod(self.xp, self.Lx)

    def calcEnergy(self):

        return 0.5 * (self.Eg ** 2).sum() * self.dx


    def calcPotEnergy(self):

        return (0.5 * np.sum(self.Eg ** 2)*self.dx)
    def calcKinEnergy(self):
        return 0.5 * self.charge / self.qDm * np.sum(self.vp ** 2)
    def calcMomentum(self):
        return (self.charge / self.qDm* np.sum(self.vp)  )