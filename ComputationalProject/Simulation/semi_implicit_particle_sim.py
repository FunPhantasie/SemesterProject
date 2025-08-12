import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from .helper import MathTools
from scipy.ndimage import gaussian_filter

class PIC_Solver(MathTools):

    def __init__(self, dimension,dt,stepssize,border,gridNumbers,species):

        # Physical constants
        self.c = 1
        self.pi = np.pi
        self.omega_p = 1.  # Plasma Freq.
        self.epsilon_0 = 1.  # Copied Convenient normalization

        # Setup Math Tool Stepsize include dx,dy,dz if exist
        super().__init__(dimension=dimension, stepssize=stepssize)
        self.dimension = dimension

        #Stabilitay Evolution Params
        self.theta = 0.8  # Implicit Parameter
        self.dt = dt



        # Resulting Connected Conditions
        self.combi = self.c * self.theta * self.dt
        self.Volume=np.prod(border)
        self.GridVolume=np.prod(gridNumbers)
        self.weight=1/(self.Nx*self.dx)


        # Handling Multiple Species
        for sp in species:
            sp["beta"] = sp["q"] * self.dt / (2 * sp["m"] * self.c)*self.weight
            sp["q"]*=self.weight

            sp["rho"] = np.zeros([*gridNumbers])


            # Initialize the Particles Global Positions and Velocities

            sp["Fp"] = np.zeros([3, sp["Np"]])
            sp["Ep"] = np.zeros([3, sp["Np"]])
            sp["Bp"] = np.zeros([3, sp["Np"]])
            sp["E_theta_p"] = np.zeros([3, sp["Np"]])
            sp["vp"] = np.zeros([3, sp["Np"]])

            """Helpers Iter"""
            sp["J_hat"] = np.zeros([3, *gridNumbers])
            sp["rho_hat"] = np.zeros([*gridNumbers])
            sp["xp_iter"] = np.zeros([sp["Np"]])
            sp["vp_iter"] = np.zeros([3, sp["Np"]])

            if self.dimension==3:
                sp["xp"] = np.zeros([3, sp["Np"]])
            elif self.dimension==1:
                sp["xp"] = np.zeros([sp["Np"]])
            else:
                raise SyntaxError("Dimension must be 3 or 1")



        self.species = species

        # Start Initial State
        self.t = 0.0
        self.N_steps = 1



        self.times = []
        if self.dimension ==1:
            self.particle_mover=self.particle_mover1d
        elif self.dimension==3:
            self.particle_mover=self.particle_mover3d


        """Analytics Initialization"""
        self.Ekin0 = np.sum(self.species[0]["vp"] ** 2) * 0.5
        # self.charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Volume / Np  # particle charge
        #charge = self.omega_p ** 2 / (self.q_p / self.m_p) * self.epsilon_0 * self.Lx / self.Np  # particle charge




        self.checkStability()  # Cfl Stability

    def deposit_charge(self,x_p,Np,q_p, ShapeFunction):
        """8 Volumes 3 Dimensional"""
        rho=self.ShaperParticle(x_p,Np,q_p, ShapeFunction)
        rho *=self.weight
        return rho

    def interpolate_fields_to_particles(self,  x_p,field,Np, ShapeFunction):
        """Interpolate E and B fields to particle positions"""
        # Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
        return self.ShaperParticle(x_p,Np ,field, ShapeFunction, toParticle=True)

    def matrix_rhs_equation(self, E, B, J_hat, rho_hat,combi,c):

        return E + combi * (self.curl(B) - 4 * self.pi / c * J_hat) - combi ** 2 * 4 * self.pi * self.gradient(rho_hat)

    def Evolver_R(self,vec,Field,beta,c):

        #return vec #Electro static
        gg=vec+beta/c *self.cross(vec,Field)+(beta/c)**2 *self.dot(vec,Field)*Field
        return gg/(1+(beta/c)**2*np.sum(np.abs(Field)**2, axis=0))

    def interpolate_particlefield_to_grid(self,x_p,Np,field,ShapeFunction):
        return self.ShaperParticle(x_p,Np, field, ShapeFunction, toParticle=False)

    def matrix_lhs_equation(self, E_theta,species,combi,c):


        mu_E_theta=np.zeros_like(E_theta,dtype=float)
        for ssp in species:
            beta_ssp=ssp["beta"]
            q_ssp=ssp["q"]
            rho_clipped = np.clip(ssp["rho"], 0.0, None)
            alpha_E = self.Evolver_R(self.E_theta, self.B, beta_ssp, c)
            mu_E_theta += - 4 *self.pi*self.theta*self.dt* beta_ssp*q_ssp * rho_clipped[np.newaxis, ...] * alpha_E
        return E_theta + mu_E_theta - combi ** 2 * (self.laplace_vector(E_theta) + self.gradient(self.divergence(mu_E_theta)))

    def A_operator(self, v_flat,species,combi,c):
        # Reshape flat vector to [3, Nx, Ny, Nz]

        if self.dimension == 3:
            v = v_flat.reshape(3, self.Nx, self.Ny, self.Nz)
        elif self.dimension == 1:
            v = v_flat.reshape(3, self.Nx)
        else:
            raise SyntaxError("Wrong Dim"+str(self.dimension))

        # Compute A * v using E_theta_LHS
        Av = self.matrix_lhs_equation(v,species=species,combi=combi,c=c)
        # Flatten result back to 1D
        return Av.ravel()

    def solveMatrixEquation(self,rhs,prevEtheta,species,combi,c):
        rhs_flat = rhs.ravel()

        A = LinearOperator((self.totalN, self.totalN), matvec=lambda v: self.A_operator(v,species=species,combi=combi,c=c))
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

    def binomial_filter(self,array):

        return 0.25 * np.roll(array, 1) + 0.5 * array + 0.25 * np.roll(array, -1)

    def calcJ_hat(self,xp,R_vp,Np,q_p,ShapeFunction):

        first_sum_vec=q_p*self.ShaperParticle(xp,Np, R_vp, ShapeFunction)                  #[3,nx]

        #return first_sum_vec
        second_sum=_scalar=q_p*self.ShaperParticle(xp,Np, np.sum(R_vp**2, axis=0), ShapeFunction) # [1,Nx]



        return first_sum_vec - self.theta*self.dt *self.gradient(second_sum)

    def calcRho_hat(self,rho,J_hat):
        return rho
        return rho-self.dt*self.theta *self.divergence(J_hat)

    def calc_v_hat(self,vp,E_theta_p,beta):
        return vp+beta*E_theta_p

    def particle_mover3d(self,vp_mid,xp,dt):
        return xp+dt*vp_mid

    def particle_mover1d(self,vp_mid,xp,dt):
        return xp+dt*vp_mid[0,...]

    def boundary(self,x):
        if self.dimension==3:
            x[0]=np.mod(x[0], self.Lx)
            x[1]=np.mod(x[1], self.Ly)
            x[2]=np.mod(x[2], self.Lz)
            return x
        elif self.dimension==1:
            return  np.mod(x, self.Lx)
        else:
            raise NotImplementedError("Wrong Dim"+str(self.dimension))

    def MomentsGathering(self, xp,vp,Bp,Np,qp,beta,c, af):
        """Advance one full PIC cycle"""

        rho = self.deposit_charge(xp,Np,qp, af)
        #rho = self.binomial_filter(rho)
        #rho = gaussian_filter(rho, sigma=1.0)
        #vp = self.binomial_filter(vp)
        #vp=gaussian_filter(vp, sigma=1.0)
        R_vp = self.Evolver_R(vp, Bp,beta=beta,c=c)
        J_hat = self.calcJ_hat(xp, R_vp,Np,qp,af)
        #J_hat =self.binomial_filter(J_hat)
        return rho,J_hat


    def Looper(self, x_i,vp,Np,beta,c, af):
        # Grid to Particle
        #self.E_theta= np.zeros_like(self.E_theta)
        #self.B= np.zeros_like(self.B)

        E_theta_p = self.interpolate_fields_to_particles(x_i,self.E_theta,Np, af)  # E
        Bp = self.interpolate_fields_to_particles(x_i, self.B, Np,af)  # B

        # Calc Velocity

        v_hat = self.calc_v_hat(vp, E_theta_p,beta)  # Here its Important that it is vp

        v_hat = self.Evolver_R(v_hat, Bp,beta = beta, c = c)

        x_i = self.particle_mover(v_hat,x_i, 0.5 * self.dt)

        return self.boundary(x_i), v_hat

    """Advance one full PIC cycle for all species"""
    def step(self):
        """Take Params to Use"""
        af = lambda a: a #Spline Function (Shaperfunction) Identity yet
        c=self.c
        combi=self.combi

        """Moments Gathering for all species"""
        for spp in self.species:
            q_spp = spp["q"]
            m_spp = spp["m"]
            beta_spp = spp["beta"]
            x_spp = spp["xp"] #Note this doesnt Copy just references
            v_spp = spp["vp"]
            Np_ssp= spp["Np"]
            Bp_ssp=spp["Bp"]
            spp["rho"],spp["J_hat"]=self.MomentsGathering(x_spp,v_spp,Bp=Bp_ssp,Np=Np_ssp,qp=q_spp,beta=beta_spp,c=c,af=af)
            spp["rho_hat"] = self.calcRho_hat(spp["rho"], spp["J_hat"])
            #spp["rho_hat"] = self.binomial_filter(spp["rho_hat"])

        rho_total = np.zeros_like(spp["rho"])
        J_hat_total = np.zeros_like(spp["J_hat"])
        rho_hat_total = np.zeros_like(spp["rho_hat"])
        for spp in self.species:
            rho_total += spp["rho"]
            J_hat_total += spp["J_hat"]
            rho_hat_total += spp["rho_hat"]
        """Moments Gathering Finshed"""


        # Matrix
        rhs = self.matrix_rhs_equation(self.E, self.B, J_hat_total, rho_hat_total,combi=combi,c=c)  # TO Vector
        self.E_theta = self.solveMatrixEquation(rhs, self.E_theta, self.species, combi, c)



        #self.E_theta[0] = self.binomial_filter(self.E_theta[0])



        """
        Current Looping for Updated Positions
        """
        for spp in self.species:
            spp["xp_iter"], spp["vp_iter"] = self.Looper(spp["xp"], spp["vp"], spp["Np"], beta=spp["beta"], c=c, af=af)
        count = 0
        for _ in range(5):
            total_error = 0.0
            for spp in self.species:
                xp_old = spp["xp_iter"].copy()
                spp["xp_iter"], spp["vp_iter"] = self.Looper(spp["xp_iter"], spp["vp_iter"],spp["Np"],beta=spp["beta"], c=c, af=af)
                total_error += np.linalg.norm(spp["xp_iter"] - xp_old)
            if total_error < 1e-6:
                #print("Iteration stopped after:" + str(count))
                break

            count += 1



        for spp in self.species:
            q_spp = spp["q"]
            m_spp = spp["m"]
            beta_spp = spp["beta"]
            spp["vp"] = 2 * spp["vp_iter"] - spp["vp"]
            # spp["vp"]=(spp["vp_iter"]-(1-self.theta)*spp["vp"])/self.theta #For all Thetas
            spp["xp"] = self.particle_mover(spp["vp_iter"], spp["xp"], self.dt)
            spp["xp"] = self.boundary(spp["xp"])

            ## For Debugging not needed Elsewhere
            spp["rho"] = self.deposit_charge(spp["xp"],spp["Np"],q_spp, af)

        # Update Fields
        self.E = (self.E_theta - (1 - self.theta) * self.E) / self.theta  # For all Theta
        self.B -= c * c * self.curl(self.E_theta)


        self.t += self.dt




    """
    Analytics for Debugging
    """
    def CalcKinEnergery(self):
        el = self.species[0]
        return np.sum(el["vp"]**2)*0.5
    def CalcEFieldEnergy(self):
        return np.sum(self.E**2)*0.5
    def analyze_E_theta_RHS(self):
        el=self.species[0]
        curlB = self.combi*self.curl(self.B)
        grad_rho = self.gradient(el["rho_hat"])
        term1 = self.E.copy()
        term2 =  (curlB -self.combi * 4 * self.pi / self.c * el["J_hat"])
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
    def checkStability(self):
        """Idk what Im doing"""
        return
