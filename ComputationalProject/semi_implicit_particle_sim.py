import numpy as np

#from ..ChaosSim.integrator_module import Integrator
#from ..ChaosSim.fourier_module import FourierSolver
#from ..ChaosSim.animationstudio_module import AnimatedScatter
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments

import unittest


class PIC_Solver:
    def __int__(self):
       pass
    def deposit_charge(self,x_p,rho,ShapeFunction):
        """8 Volumes"""
        rho*=0
        for particle_index in range(self.Np):
            x,y,z=x_p[:,particle_index]

            xn,yn,zn=(x/self.dx),(y/self.dy),(z/self.dz)
            ix,iy,iz = int(xn),int(yn),int(zn)
            #Arround The World
            #Muss Rho Volumes zuordnen
            for ax in [0,1]:
                for by in [0,1]:
                    for cz in [0,1]:
                        grid_point_x = np.mod(ix + ax, self.Nx) #Entweder x0 oder x1
                        grid_point_y = np.mod(iy + by, self.Ny)
                        grid_point_z = np.mod(iz + cz, self.Nz)
                        # Weight based on linear distance (CIC)
                        wx = 1 - abs(xn - (ix + ax))
                        wy = 1 - abs(yn - (iy + by))
                        wz = 1 - abs(zn - (iz + cz))
                        w = wx * wy * wz



                        rho[grid_point_x,grid_point_y,grid_point_z]+=ShapeFunction(w)
        return rho

    def interpolate_fields_to_particles(self,field,x_p,fieldp,ShapeFunctionn):
        """Interpolate E and B fields to particle positions"""
        #Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
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
                        wx = 1 - abs(xn - (ix + ax)) #Kein Modulo macht alles kaputt
                        wy = 1 - abs(yn - (iy + by))
                        wz = 1 - abs(zn - (iz + cz))
                        w = wx * wy * wz
                        fieldp[:,particle_index] += field[:,grid_point_x,grid_point_y,grid_point_z]*ShapeFunctionn(w)
        return fieldp

    #Helper FOr Jhat
    def be_behatted(self,x_p,v_p_tilde,ShapeFunctionx,ShapeFunctiony,ShapeFunctionz,term):
        """8 Volumes"""
        helper=np.zeros([3,self.Nx,self.Ny,self.Nz])
        if term==2:
            v_p_h=v_p_tilde**2
        elif term==1:
            v_p_h = v_p_tilde
        else:
            raise SyntaxError("Only Square or Linear")
        for particle_index in range(self.Np):
            x,y,z=x_p[:,particle_index]
            particle_velocity_tilde=v_p_h[:,particle_index]



            xn,yn,zn=(x/self.dx),(y/self.dy),(z/self.dz)
            ix,iy,iz = int(xn),int(yn),int(zn)
            #Arround The World
            #Muss Rho Volumes zuordnen
            for ax in [0,1]:
                for by in [0,1]:
                    for cz in [0,1]:
                        grid_point_x = np.mod(ix + ax, self.Nx) #Entweder x0 oder x1
                        grid_point_y = np.mod(iy + by, self.Ny)
                        grid_point_z = np.mod(iz + cz, self.Nz)
                        # Weight based on linear distance (CIC)
                        wx = ShapeFunctionx( abs(xn - (ix + ax)))
                        wy = ShapeFunctiony( abs(yn - (iy + by)))
                        wz = ShapeFunctionz( abs(zn - (iz + cz)))
                        w = wx * wy * wz
                        helper[:,grid_point_x,grid_point_y,grid_point_z]+=self.q_p *particle_velocity_tilde *w
        return helper

    def R_tilder(self,vec,beta,Bp):
        gg=vec-beta/self.c *self.cross(vec,Bp)+(beta/self.c)**2 *self.dot(vec,Bp)*Bp
        return gg/(1+(beta/self.c)**2*np.sum(np.abs(Bp)**2, axis=0))

    def calcJ_hat(self,Bp,xp,vp,beta):
        shaper=lambda a: 1-a
        vp_tilde=self.R_tilder(vp,beta,Bp)
        zz=self.be_behatted(xp,vp_tilde,shaper,shaper,shaper,1)
        hh=self.be_behatted(xp,vp_tilde,shaper,shaper,shaper,2)
        self.J_hat= zz - self.theta*self.dt *self.divergence(hh)

    def calcRho_hat(self,J_hat):
        self.rho_hat= self.rho-self.dt*self.theta *self.divergence(J_hat)



    def calc_v_hat(self,vp,beta,E_theta_p):
        return vp+beta*E_theta_p

    def E_theta_RHS(self,E,B,J,rho,combi):
        return E+combi *(self.curl(B)-4*self.pi/self.c* J)-combi**2 *4 *self.pi *self.gradient(rho)
    """Some MAtrix Shit"""
    def E_theta_LHS(self,E_theta,mu):
        helper=E_theta*mu
        return E_theta+helper-self.combi**2 *(self.laplace_vector(E_theta)+self.gradient(self.divergence(helper)))


    def GetETheta(self):
        combi = self.c * self.theta * self.dt
        mu=1
        rhs=self.E_theta_RHS(self.E,self.B,self.J_hat,self.rho_hat,combi)
        # Flatten the 3D arrays into 1D for matrix operations
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        E_theta_flat = self.E_theta.reshape(3 * Nx * Ny * Nz)
        rhs_flat = rhs.reshape(3 * Nx * Ny * Nz)

        # Construct the LHS matrix A (sparse)
        from scipy import sparse
        diag = np.ones(3 * Nx * Ny * Nz) * (1 + mu)
        data = []
        offsets = []
        # Diagonal term (I + mu)
        data.append(diag)
        offsets.append(0)

        # Laplacian and divergence terms (simplified sparse structure)
        laplace = self.laplace_vector(np.zeros_like(self.E_theta)).reshape(3 * Nx * Ny * Nz)
        div_helper = self.divergence(self.gradient(np.zeros_like(self.E_theta[0])))
        div_term = self.gradient(div_helper).reshape(3 * Nx * Ny * Nz)
        lap_div = -(combi ** 2) * (laplace + div_term)
        for i in range(len(lap_div)):
            if abs(lap_div[i]) > 1e-10:  # Non-zero entries
                data.append([lap_div[i]])
                offsets.append(i)

        A = sparse.diags(data, offsets, shape=(3 * Nx * Ny * Nz, 3 * Nx * Ny * Nz), format='csr')

        # Solve using GMRES
        from scipy.sparse.linalg import gmres
        E_theta_flat_new, exit_code = gmres(A, rhs_flat,  maxiter=100)
        if exit_code == 0:
            self.E_theta = E_theta_flat_new.reshape(3, Nx, Ny, Nz)
        else:
            raise ValueError("GMRES failed to converge")

        self.E_theta

    def GetB(self):
        return
    def particle_mover(self,vp_mid,dt):
        return self.xp+dt*vp_mid
    def Half_step(self,x_i):
        """Advance one full PIC cycle"""
        af = lambda a: a
        beta = self.q_p * self.dt / (2 * self.m_p * self.c)

        self.rho=self.deposit_charge(x_i,self.rho,af) #Check
        self.calcJ_hat(self.Bp,x_i,self.vp,beta) #With updated pos
        self.calcRho_hat(self.J_hat)

        self.GetETheta()
        self.GetB()


        self.interpolate_fields_to_particles(self.E_theta,x_i,self.E_theta_p,af)#E
        self.interpolate_fields_to_particles(self.B, x_i, self.Bp, af)#B

        #Now obtained habe En_theta, Bn

        v_hat=self.calc_v_hat(self.vp,beta,self.E_theta_p)

        self.vp_iter=self.R_tilder(v_hat,beta,self.Bp)
        x_i=self.particle_mover(self.vp_iter,0.5* self.dt)

        return x_i
    def step(self):
        """Advance one full PIC cycle"""
        self.xp_iter=self.Half_step(self.xp) #Should be Grid

        for i in range(2):
            self.xp_iter=self.Half_step(self.xp_iter)

        self.xp=self.particle_mover(self.vp_iter,self.dt)
        self.E+=2*self.E_theta
        self.B-=self.c*self.c*self.curl(self.E_theta)
        #Könnte auch V_n1 bestimmen aber brauch man nicht. Wird beim nächsten neu Approximiert
        self.t += self.dt

    """Math Operations"""
    def laplace_vector(self,A): #[3,..]
        lap_A = np.zeros_like(A)
        dk=[self.dx,self.dy,self.dz]
        for index,a_k in enumerate(A):
            rolledFor = np.roll(a_k, shift=-1, axis=index)  # Shift along x-axis
            rolledBack = np.roll(a_k, shift=1, axis=index)  # Shift along y-axis
            lap_A[index]=(rolledFor+rolledBack-2*a_k)/(dk[index]**2)
        return lap_A

    def cross(self, A, B):
        U = np.zeros_like(A)
        U[0, ...] = A[1, ...] * B[2, ...] - A[2, ...] * B[1, ...]
        U[1, ...] = A[2, ...] * B[0, ...] - A[0, ...] * B[2, ...]
        U[2, ...] = A[0, ...] * B[1, ...] - A[1, ...] * B[0, ...]
        """
        Or Var:
        
        result = np.cross(a.T, b.T)
        3 packs of vec  mal particles 
        """
        return U

    def dot(self,A, B):
        U = np.zeros_like(A)
        U[0, ...] = A[0, ...] * B[0, ...]
        U[1, ...] = A[1, ...] * B[1, ...]
        U[2, ...] = A[2, ...] * B[2, ...]
        """
        Or Var:

        result = np.sum(a * b, axis=0)  # Shape (self.Np,)
        doesnt work
        """
        return  np.sum(A * B, axis=0)

    # Finite difference functions
    def gradient(self,f):
        # Compute shifted arrays using np.roll for finite differences
        grad = np.zeros([3, *f.shape])
        rolledx = np.roll(f, shift=-2, axis=0)  # Shift along x-axis
        rolledy = np.roll(f, shift=-2, axis=1)  # Shift along y-axis
        rolledz = np.roll(f, shift=-2, axis=2)  # Shift along z-axis

        grad[0]=(rolledx-f)/(2*self.dx)
        grad[1]=(rolledy-f)/(2*self.dy)
        grad[2]=(rolledz-f)/(2*self.dz)
        return grad

    def divergence(self,A):
        div_A = np.zeros_like(A[0])  # At B points (reduced size)
        rolledx = np.roll(A, shift=-1, axis=1)[0]  # Shift along x-axis
        rolledy = np.roll(A, shift=-1, axis=2)[1]  # Shift along y-axis
        rolledz = np.roll(A, shift=-1, axis=3)[2]  # Shift along z-axis

        div_A[:, :, :] = ( rolledx-A[0]) / self.dx + \
                         (rolledy-A[1]) / self.dy + \
                         (rolledz-A[2]) / self.dz
        return div_A
    def curl(self,A):
        curl = np.zeros_like(A)  # Initialize curl array with same shape as A

        # Compute shifted arrays using np.roll for finite differences
        rolledx = np.roll(A, shift=-1, axis=1)  # Shift along x-axis
        rolledy = np.roll(A, shift=-1, axis=2)  # Shift along y-axis
        rolledz = np.roll(A, shift=-1, axis=3)  # Shift along z-axis

        # Compute finite differences
        # Forward difference: (A(x+dx) - A(x)) / dx

        dAy_dx = (rolledx[1] - A[1]) / self.dx  # ∂Ay/∂x
        dAz_dx = (rolledx[2] - A[2]) / self.dx  # ∂Az/∂x

        dAx_dy = (rolledy[0] - A[0]) / self.dy  # ∂Ax/∂y

        dAz_dy = (rolledy[2] - A[2]) / self.dy  # ∂Az/∂y

        dAx_dz = (rolledz[0] - A[0]) / self.dz  # ∂Ax/∂z
        dAy_dz = (rolledz[1] - A[1]) / self.dz  # ∂Ay/∂z


        # Compute curl components
        # curl_x = ∂Az/∂y - ∂Ay/∂z
        curl[0] = dAz_dy - dAy_dz
        # curl_y = ∂Ax/∂z - ∂Az/∂x
        curl[1] = dAx_dz - dAz_dx
        # curl_z = ∂Ay/∂x - ∂Ax/∂y
        curl[2] = dAy_dx - dAx_dy

        return curl


class Pic_2DCase(PIC_Solver):
    # Borders
    Lx = 25.0  # Border length
    Ly = 25.0  # box size (fits fastest growing mode of two-stream); space step (normalized to u_d/omega_p)
    Lz = 25.0
    # Grid Points alongs Axis
    Nx = 10  # Number of grid points
    Ny = 10  # Number of grid points
    Nz = 10
    # Number per Cell
    NPpCell = 1

    # Constants
    c = 1
    pi = np.pi
    q_p = -1.0
    m_p = 1.0
    omega_p = 1.  # plasma frequency
    epsilon_0 = 1.  # convenient normalization

    def __init__(self, dimension="2d", step_method="EulerForward"):
        super().__init__()

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        self.dt = 0.01  # Check Bedinung

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
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,
                                             indexing='ij')  # Meshgrid Discussion about Indexing

        # Grid Fields and Densities
        self.rho = np.zeros([self.Nx, self.Ny, self.Nz])
        self.E = np.zeros(
            [3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
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
        self.E_theta = np.zeros(
            [3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
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

        # Initial state of the simulation
        self.t = 0.0
        self.N_steps = 1

        self.Ekin0 = np.sum(self.vp ** 2) * 0.5
        self.Ekin = []
        self.times = []

class Pic_3DCase(PIC_Solver):
    # Borders
    Lx = 25.0  # Border length
    Ly = 25.0  # box size (fits fastest growing mode of two-stream); space step (normalized to u_d/omega_p)
    Lz = 25.0
    # Grid Points alongs Axis
    Nx = 10  # Number of grid points
    Ny = 10  # Number of grid points
    Nz = 10
    # Number per Cell
    NPpCell = 1

    # Constants
    c = 1
    pi = np.pi
    q_p = -1.0
    m_p = 1.0
    omega_p = 1.  # plasma frequency
    epsilon_0 = 1.  # convenient normalization

    def __init__(self, dimension="2d", step_method="EulerForward"):
        super().__init__()

        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz

        self.dt = 0.01  # Check Bedinung

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
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z,
                                             indexing='ij')  # Meshgrid Discussion about Indexing

        # Grid Fields and Densities
        self.rho = np.zeros([self.Nx, self.Ny, self.Nz])
        self.E = np.zeros(
            [3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
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
        self.E_theta = np.zeros(
            [3, self.Nx, self.Ny, self.Nz])  # Ex, Ey Its E but bc only in forward time its used no one cares
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

        # Initial state of the simulation
        self.t = 0.0
        self.N_steps = 1

        self.Ekin0 = np.sum(self.vp ** 2) * 0.5
        self.Ekin = []
        self.times = []
