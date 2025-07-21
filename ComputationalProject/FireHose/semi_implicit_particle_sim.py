import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from helper import MathTools
from scipy.ndimage import gaussian_filter
class PIC_Solver(MathTools):

    def __init__(self, dimension,dt,steps,border,Np,gridNumbers):
        # Implicit Parameter
        self.theta = 0.8  # Implicitness parameter

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

        self.beta = self.q_p * self.dt / (2 * self.m_p * self.c)
        self.combi = self.c * self.theta * self.dt
        self.Volume=np.prod(border)
        self.GridVolume=np.prod(gridNumbers)
        self.charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Volume / Np  # particle charge


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





    def deposit_charge(self, x_p, ShapeFunction):
        """8 Volumes 3 Dimensional"""
        rho=self.ShaperParticle(x_p,self.q_p, ShapeFunction)

        return rho

    def interpolate_fields_to_particles(self,  x_p,field, ShapeFunction):
        """Interpolate E and B fields to particle positions"""
        # Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
        return self.ShaperParticle(x_p, field, ShapeFunction, toParticle=True)

    def analyze_E_theta_RHS(self):
        curlB = self.combi*self.curl(self.B)
        grad_rho = self.gradient(self.rho_hat)
        term1 = self.E.copy()
        term2 =  (curlB -self.combi * 4 * self.pi / self.c * self.J_hat)
        term3 = - self.combi ** 2 * 4 * self.pi * grad_rho

        E_total = term1 + term2 + term3

        # Normen berechnen
        norm1 = np.linalg.norm(term1)
        norm2 = np.linalg.norm(term2)
        norm3 = np.linalg.norm(term3)
        normE = np.linalg.norm(E_total)
        curlBnorm=np.linalg.norm(curlB)


        print("\n--- E_theta RHS analysis ---")
        print(f"||E||        = {norm1:.4e}")
        print(f"||curlB||        = {curlBnorm:.4e}")
        print(f"||curlB - J||= {norm2:.4e}")
        print(f"||grad(rho)||= {norm3:.4e}")
        print(f"||E_total||  = {normE:.4e}")



    def matrix_rhs_equation(self, E, B, J_hat, rho_hat):

        return E + self.combi * (self.curl(B) - 4 * self.pi / self.c * J_hat) - self.combi ** 2 * 4 * self.pi * self.gradient(rho_hat)


    def Evolver_R(self,vec,Field):

        #return vec #Electro static
        gg=vec+self.beta/self.c *self.cross(vec,Field)+(self.beta/self.c)**2 *self.dot(vec,Field)*Field
        return gg/(1+(self.beta/self.c)**2*np.sum(np.abs(Field)**2, axis=0))
    def interpolate_particlefield_to_grid(self,x_p,field,ShapeFunction):
        return self.ShaperParticle(x_p, field, ShapeFunction, toParticle=False)

    def matrix_lhs_equation(self, E_theta):

        #E_theta_p=self.interpolate_fields_to_particles(x_p,E_theta,ShapeFunction)
        rho_clipped = np.clip(self.rho, 0.0, None)
        alpha_E = self.Evolver_R(self.E_theta, self.B)
        mu_E_theta = - 4 *self.pi*self.theta*self.dt* self.beta*self.q_p * rho_clipped[np.newaxis, ...] * alpha_E
        return E_theta + mu_E_theta - self.combi ** 2 * (self.laplace_vector(E_theta) + self.gradient(self.divergence(mu_E_theta)))



    def A_operator(self, v_flat):
        # Reshape flat vector to [3, Nx, Ny, Nz]

        if self.dimension == 3:
            v = v_flat.reshape(3, self.Nx, self.Ny, self.Nz)
        elif self.dimension == 1:
            v = v_flat.reshape(3, self.Nx)
        else:
            raise SyntaxError("Wrong Dim"+str(self.dimension))

        # Compute A * v using E_theta_LHS
        Av = self.matrix_lhs_equation(v)
        # Flatten result back to 1D
        return Av.ravel()

    def solveMatrixEquation(self,rhs,prevEtheta):
        rhs_flat = rhs.ravel()

        A = LinearOperator((self.totalN, self.totalN), matvec=lambda v: self.A_operator(v))
        E_theta_flat, info = gmres(A, rhs_flat, x0=prevEtheta.ravel(), rtol=1e-6, restart=30)
        if info == 0:
            if self.dimension==3:
                return E_theta_flat.reshape(3, self.Nx, self.Ny, self.Nz)
            elif self.dimension==1:
                return  E_theta_flat.reshape(3, self.Nx)
            else:
                raise SyntaxError("Wrong Dim" + str(self.dimension))
        else:
            raise ValueError("GMRES failed to converge")




    #Denoted as R in LEcture


    def binomial_filter(self,array):

        return 0.25 * np.roll(array, 1) + 0.5 * array + 0.25 * np.roll(array, -1)

    def calcJ_hat(self,xp,R_vp,ShapeFunction):

        first_sum_vec=self.q_p*self.ShaperParticle(xp, R_vp, ShapeFunction)                  #[3,nx]

        second_sum=_scalar=self.q_p*self.ShaperParticle(xp, np.sum(R_vp**2, axis=0), ShapeFunction) # [1,Nx]



        return first_sum_vec - self.theta*self.dt *self.gradient(second_sum)

    def calcRho_hat(self,J_hat):
        return self.rho-self.dt*self.theta *self.divergence(J_hat)



    def calc_v_hat(self,vp,E_theta_p):
        return vp+self.beta*E_theta_p

    def particle_mover3d(self,vp_mid,dt):
        return self.xp+dt*vp_mid
    def particle_mover1d(self,vp_mid,dt):
        return self.xp+dt*vp_mid[0,...]

    def boundary(self,x):
        if self.dimension==3:
            raise NotImplementedError()
        return  np.mod(x, self.Lx)

    def MomentsGathering(self, xp, af):
        """Advance one full PIC cycle"""

        self.rho = self.deposit_charge(xp, af)  # That works
        #self.rho = self.binomial_filter(self.rho)


        #self.rho = gaussian_filter(self.rho, sigma=1.0)
        # Could Add SMoothing function for Oszilations

        R_vp = self.Evolver_R(self.vp, self.Bp)

        self.J_hat = self.calcJ_hat(xp, R_vp, af)

        self.rho_hat = self.calcRho_hat(self.J_hat)
        self.rho_hat = self.binomial_filter(self.rho_hat)

        # MAtrix
        rhs = self.matrix_rhs_equation(self.E, self.B, self.J_hat, self.rho_hat)  # TO Vector
        self.E_theta = self.solveMatrixEquation(rhs, self.E_theta)
        self.E_theta[0] = self.binomial_filter(self.E_theta[0])



    def Looper(self, x_i, af):

        # Grid to Particle
        self.E_theta_p = self.interpolate_fields_to_particles(x_i, self.E_theta, af)  # E
        self.Bp = self.interpolate_fields_to_particles(x_i, self.B, af)  # B

        # Calc Velocity


        #v_hat = self.calc_v_hat(self.vp, self.E_theta_p)  # Here its Important that it is vp
        v_hat= self.vp
        #v_hat = self.Evolver_R(v_hat, self.Bp)

        x_i = self.particle_mover(v_hat, 0.5 * self.dt)

        return self.boundary(x_i),v_hat

    def step(self):
        """Advance one full PIC cycle"""
        af = lambda a: a

        self.MomentsGathering(self.xp,  af)

        # Hand Loop
        self.xp_iter, self.vp_iter = self.Looper(self.xp, af)
        count = 0
        for k in range(5):

            xp_iter = self.xp_iter.copy()
            self.xp_iter, self.vp_iter = self.Looper(self.xp_iter, af)
            if np.linalg.norm((xp_iter - self.xp_iter)) < 1e-6:
                #print("Iteration stopped after:" + str(count))
                break

            count += 1

        ###
        # End
        ###

        self.vp = 2 * self.vp_iter - self.vp
        # self.vp=(self.vp_iter-(1-self.theta)*self.vp)/self.theta #For all Thetas

        ## For Debugging not needed Elsewhere
        self.rho = self.deposit_charge(self.xp, af)
        self.E_prev = self.E
        ##

        self.xp = self.particle_mover(self.vp_iter, self.dt)
        self.xp = self.boundary(self.xp)

        # Update Fields

        self.E = (self.E_theta - (1 - self.theta) * self.E) / self.theta  # For all Thetas

        self.B -= self.c * self.c * self.curl(self.E_theta)

        self.t += self.dt

    def CalcKinEnergery(self):
        return np.sum(self.vp**2)*0.5
    def CalcEFieldEnergy(self):
        return np.sum(self.E**2)*0.5


