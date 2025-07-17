import numpy as np
from helper import MathTools
from sort import initialize_two_stream1D
class Explicit_PIC_Solver(MathTools):

    def __init__(self, border=1, gridpoints=128, NPpCell=20, dt=0.1):
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
        self.charge = self.omega_p ** 2 / self.qDm * self.epsilon_0 * self.Lx / self.Np

        self.E = np.zeros([3, self.Nx])
        self.B = np.zeros([3, self.Nx])
        self.rho = np.zeros(self.Nx)

        self.vp = np.zeros([3, self.Np])
        self.Ep = np.zeros([3, self.Np])
        self.Bp = np.zeros([3, self.Np])

        self.xp, self.vp, self.B = initialize_two_stream1D(self.Lx, self.Np, self.vp, self.B, amplitude=0.01)

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
        # rho -= self.Np / self.Nx its not than if all moments are the same
        # rho *= 2 * self.NPpCell * self.charge / (self.Volume/self.GridVolume) #Ist kompliziert weil geklaut aber eigentlich nur  2 *omega^2 m/q *epsilon´
        # print(2 * self.NPpCell * self.charge / (self.Volume/self.GridVolume) )

    def force(self):
        self.interpolation_to_part(self.Ep, self.E)
        self.interpolation_to_part(self.Bp, self.B)

    def weight_J(self):
        J = np.zeros(self.Nx)
        for p in range(self.Np):
            zeta = self.xp[p] / self.dx
            i = int(zeta)
            ip1 = (i + 1) % self.Nx
            diff = zeta - i
            J[i] += (1 - diff) * self.vp[0, p]
            J[ip1] += diff * self.vp[0, p]
        J *= self.charge / self.dx
        return J

    def calc_E(self):
        J = self.weight_J()
        self.E[0] += - 4 * np.pi * self.dt * J

    def boundary(self):
        self.xp = np.mod(self.xp, self.Lx)

    def CalcKinEnergery(self):
        return (0.5 * np.sum(self.vp ** 2) / self.Ekin0 * 100)

    def CalcEFieldEnergy(self):
        return (0.5 * np.sum(self.E ** 2))
