import numpy as np


"""
Solver with the same Scheme as the Electro Statitic Code for two Streams.
In Case of no normalisation of Grid Charge the E field is the same as in the implicit Method.
"""
class Explicit_PIC_Solver():

    def __init__(self, border=1, gridpoints=128, NPpCell=20, dt=0.1):
        self.Nx = gridpoints
        self.dt = dt
        self.Lx = border
        self.dx = self.Lx / self.Nx
        self.NPpCell = NPpCell
        self.Np = self.Nx * self.NPpCell
        self.t = 0.0

        self.x = self.dx * np.arange(self.Nx)
        # self.charge =self.omega_p ** 2 / self.qDm * self.epsilon_0 * self.Lx / self.Np *
        self.qDm = -1
        self.omega_p = 1
        self.epsilon_0 = 1
        self.charge = self.qDm * 1/(self.Nx * self.dx)#self.qDm/(1) *self.dx/self.Np*self.Lx

        self.E = np.zeros([3, self.Nx])
        self.B = np.zeros([3, self.Nx])
        self.rho = np.zeros(self.Nx)
        self.xp =np.zeros( self.Np)
        self.vp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])





        """Not Used For Calculation just for Plotting"""
        sp = [{
            "name": "e",
            "q": -1.0,
            "m": 1.0,
            "beta_mag_par": 0,
            "beta_mag_perp": 0,
            "beta": None,
            "NPpCell": NPpCell,
            "Np": self.Np
        },]

        self.species=sp

    def step(self):
        self.weight_rho()
        #Electro Magnetic
        # ------------------#
        self.weight_J()

        #------------------#
        self.calc_E()
        self.calc_B()
        self.force()
        self.boris(self.dt)
        self.step_x(self.dt)
        self.boundary()
        self.t += self.dt
        self.species[0]["xp"] = self.xp
        self.species[0]["vp"] = self.vp
        self.species[0]["rho"] = self.rho

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
        C[0] = A[1] * B[2] - A[2] * B[1]
        C[1] = A[2] * B[0] - A[0] * B[2]
        C[2] = A[0] * B[1] - A[1] * B[0]
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
        self.rho*=self.charge
        #self.rho+=1/(4*np.pi)
        #self.rho -= self.rhostart
        #self.rho -= self.Np / self.Lx
        #self.rho *= 2 * self.NPpCell * self.charge / self.dx

        # rho -= self.Np / self.Nx its not than if all moments are the same
        #Ist kompliziert weil geklaut aber eigentlich nur  2 *omega^2 m/q *epsilon´
        # print(2 * self.NPpCell * self.charge / (self.Volume/self.GridVolume) )

    def force(self):
        self.interpolation_to_part(self.Ep, self.E)
        self.interpolation_to_part(self.Bp, self.B)

    def weight_J(self):
        self.J = np.zeros([3, self.Nx])
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx
            i = int(zeta) % self.Nx
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            for d in range(3):
                self.J[d, i] += (1 - diff) * self.vp[d, p]
                self.J[d, ip1] += diff * self.vp[d, p]
        self.J *= self.charge
    def calc_E(self):
        """
        rhohat = np.fft.rfft(self.rho)
        kx = 2 * np.pi / self.Lx * np.arange(rhohat.size)
        with np.errstate(divide='ignore', invalid='ignore'):
            tmp = np.where(kx * kx > 0, rhohat / (1j * kx), 0.)
        self.E[0, :] = np.fft.irfft(tmp)
        """
        # Faraday's law: dE/dt = curl B - J
        curl_B = np.zeros_like(self.E)
        curl_B[1, 1:-1] = (self.B[2, 2:] - self.B[2, :-2]) / (2 * self.dx)
        curl_B[2, 1:-1] = -(self.B[1, 2:] - self.B[1, :-2]) / (2 * self.dx)

        #self.E[:, 1:-1] += self.dt * ( - 4 * np.pi * self.J[:, 1:-1])

        self.E[:, 1:-1] += self.dt * (curl_B[:, 1:-1] - 4 * np.pi * self.J[:, 1:-1])

    def calc_B(self):
        # dB/dt = - curl E
        curl_E = np.zeros_like(self.B)
        curl_E[1, 1:-1] = -(self.E[2, 2:] - self.E[2, :-2]) / (2 * self.dx)
        curl_E[2, 1:-1] = (self.E[1, 2:] - self.E[1, :-2]) / (2 * self.dx)
        self.B[:, 1:-1] += self.dt * curl_E[:, 1:-1]

    def boundary(self):
        self.xp = np.mod(self.xp, self.Lx)

    def CalcKinEnergery(self):
        return (0.5 * np.sum(self.vp ** 2) / self.Ekin0 )

    def CalcEFieldEnergy(self):
        return (0.5 * np.sum(self.E ** 2))
