import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from helper import MathTools

class PIC_Solver(MathTools):

    def __init__(self, dimension,dt,steps,border,Np,gridNumbers):


        super().__init__(dimension=dimension,steps=steps)
        self.dimension = dimension
        self.dt = dt  # Check Bedinung

        # Physical constants
        self.c = 1
        self.pi = np.pi
        self.q_p = -1.0
        self.m_p = 1.0
        self.omega_p = 1.  # plasma frequency
        self.epsilon_0 = 1.  # convenient normalization

        self.Volume=np.prod(border)
        self.GridVolume=np.prod(gridNumbers)
        self.charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Volume / Np  # particle charge

        # Implicit Parameter
        self.theta = 0.5# Implicitness parameter

        # Initial state of the simulation
        self.t = 0.0
        self.N_steps = 1

        #self.Ekin0 = np.sum(self.vp ** 2) * 0.5

        self.times = []
        if self.dimension ==1:
            self.particle_mover=self.particle_mover1d
        elif self.dimension==3:
            self.particle_mover=self.particle_mover3d
        """TEST CFL Condition"""
        print(self.epsilon_0)



    def deposit_charge(self, x_p, ShapeFunction):
        """8 Volumes 3 Dimensional"""
        rho=self.ShaperParticle(x_p,1, ShapeFunction)
        rho -= self.Np / self.Nx
        rho *= 2 * self.NPpCell * self.charge / self.dx
        return rho

    def interpolate_fields_to_particles(self,  x_p,field, ShapeFunction):
        """Interpolate E and B fields to particle positions"""
        # Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
        return self.ShaperParticle(x_p, field, ShapeFunction, toParticle=True)

    def E_theta_RHS(self, E, B, J, rho, combi):
        #a=self.curl(B) Vor Klammer entfernt
        return E + combi * (self.curl(B) - 4 * self.pi / self.c * J) - combi ** 2 * 4 * self.pi * self.gradient(rho)

    def E_theta_LHS(self, E_theta, beta,combi):


        a_expanded = np.repeat(self.rho[np.newaxis, ...], 3, axis=0)
        mu_E_theta = 4 * np.pi * self.theta * self.dt * beta * a_expanded * self.Evolver_R(E_theta, beta, self.B)

        return E_theta + mu_E_theta - combi ** 2 * (
                self.laplace_vector(E_theta) + self.gradient(self.divergence(mu_E_theta)))

    def A_operator(self, v_flat, beta,combi):
        # Reshape flat vector to [3, Nx, Ny, Nz]

        if self.dimension == 3:
            v = v_flat.reshape(3, self.Nx, self.Ny, self.Nz)
        elif self.dimension == 1:
            v = v_flat.reshape(3, self.Nx)
        else:
            raise SyntaxError("Wrong Dim"+str(self.dimension))

        # Compute A * v using E_theta_LHS
        Av = self.E_theta_LHS(v, beta,combi)
        # Flatten result back to 1D
        return Av.ravel()

    def CalcE_Theta(self,beta,combi):


        rhs = -self.E_theta_RHS(self.E, self.B, self.J_hat, self.rho_hat, combi) #TO Vector


        rhs_flat = rhs.ravel()

        A = LinearOperator((self.totalN, self.totalN), matvec=lambda v: self.A_operator(v, beta,combi))
        E_theta_flat, info = gmres(A, rhs_flat, x0=self.E.ravel(), rtol=1e-6, restart=30)
        if info == 0:
            if self.dimension==3:
                self.E_theta = E_theta_flat.reshape(3, self.Nx, self.Ny, self.Nz)
            elif self.dimension==1:
                self.E_theta = E_theta_flat.reshape(3, self.Nx)
            else:
                raise SyntaxError("Wrong Dim" + str(self.dimension))
        else:
            raise ValueError("GMRES failed to converge")


    def CalcKinEnergery(self):
        return np.sum(self.vp**2)*0.5
    def CalcEFieldEnergy(self):
        return np.sum(self.E**2)*0.5

    #Denoted as R in LEcture
    def Evolver_R(self,vec,beta,Field):
        #return vec #Electro static
        gg=vec+beta/self.c *self.cross(vec,Field)+(beta/self.c)**2 *self.dot(vec,Field)*Field
        return gg/(1+(beta/self.c)**2*np.sum(np.abs(Field)**2, axis=0))

    def binomial_filter(self,array):

        return 0.25 * np.roll(array, 1) + 0.5 * array + 0.25 * np.roll(array, -1)

    def calcJ_hat(self,xp,R_vp,ShapeFunction):

        first_sum_vec=self.ShaperParticle(xp, R_vp, ShapeFunction)                  #[3,nx]
        second_sum=_scalar=self.ShaperParticle(xp, np.sum(R_vp**2, axis=0), ShapeFunction) # [1,Nx]



        self.J_hat= first_sum_vec - self.theta*self.dt *self.gradient(second_sum)

    def calcRho_hat(self,J_hat):
        self.rho_hat= self.rho-self.dt*self.theta *self.divergence(J_hat)



    def calc_v_hat(self,vp,beta,E_theta_p):
        return vp+beta*E_theta_p

    def particle_mover3d(self,vp_mid,dt):
        return self.xp+dt*vp_mid
    def particle_mover1d(self,vp_mid,dt):
        return self.xp+dt*vp_mid[0,...]

    def boundary(self,x):
        if self.dimension==3:
            raise NotImplementedError()
        return  np.mod(x, self.Lx)

    def Half_step(self,x_i,af):
        """Advance one full PIC cycle"""

        beta = self.q_p * self.dt / (2 * self.m_p * self.c)
        combi = self.c * self.theta * self.dt
        R_vp = self.Evolver_R(self.vp, beta, self.Bp)
        #Gathering Moments of X_I


        self.calcJ_hat(x_i,R_vp,af)
        self.rho = self.deposit_charge(x_i, af)  # Check
        self.rho = self.binomial_filter(self.rho)

        self.calcRho_hat(self.J_hat) #Get Rho Hat

        #MAtrix Implicit Equation Solver
        self.CalcE_Theta(beta,combi)
        self.E_theta[0] = self.binomial_filter(self.E_theta[0])
        # Grid to Particle
        self.E_theta_p=self.interpolate_fields_to_particles(x_i,self.E_theta, af)  # E
        self.Bp=self.interpolate_fields_to_particles( x_i ,self.B, af)  # B
        #Evolve V
        v_hat=self.calc_v_hat(self.vp,beta,self.E_theta_p)

        self.vp_iter=self.Evolver_R(v_hat,beta,self.Bp)
        x_i=self.particle_mover(self.vp_iter,0.5* self.dt)

        return self.boundary(x_i)
    def step(self):
        """Advance one full PIC cycle"""
        af = lambda a: a
        self.xp_iter=self.Half_step(self.xp,af) #Should be Grid

        for i in range(2):
            self.xp_iter=self.Half_step(self.xp_iter,af)
        self.vp=2 *self.vp_iter -self.vp
        self.xp=self.particle_mover(self.vp_iter,self.dt)
        self.xp=self.boundary(self.xp)
        #Update Fields
        self.E= (self.E_theta+(1-self.theta)*self.E)/self.theta
        self.B-=self.c*self.c*self.curl(self.E_theta)
        #Könnte auch V_n1 bestimmen aber brauch man nicht. Wird beim nächsten neu Approximiert
        self.t += self.dt




