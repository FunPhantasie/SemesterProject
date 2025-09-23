import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from .helper import MathTools
from scipy.ndimage import gaussian_filter







class IPIC_Solver():

    def __init__(self, dimension,stepssize,border,gridNumbers,species):
        #super().__init__(dimension=dimension, stepssize=stepssize) # Setup Math Tool Stepsize include dx,dy,dz if exist

        self.dimension = dimension

        #Stabilitay Evolution Params
        self.theta = 0.8  # Implicit Parameter
        self.combi = self.c * self.theta * self.dt #Used For Calc
        # Handling Multiple Species
        # Initialize the Particles Global Positions and Velocities
        for sp in species:
            sp["rho"] = np.zeros([*gridNumbers])
            sp["Fp"] = np.zeros([3, sp["Np"]])
            sp["Ep"] = np.zeros([3, sp["Np"]])
            sp["Bp"] = np.zeros([3, sp["Np"]])
            sp["E_theta_p"] = np.zeros([3, sp["Np"]])
            sp["xp_iter"] = np.zeros([sp["Np"]])
            sp["vp_iter"] = np.zeros([sp["Np"]])

        self.species = species
        self.particle_mover=self.particle_mover1d

    def MomentsGathering(self, xp, vp_x, Bp, Np, qDm, charge, c):

        mat_weights = self.ShapeFunction(xp, Np)
        rho = (charge / self.dx) * mat_weights.toarray().sum(axis=0)
        rho += self.back_charge_density

        mat_vel=mat_weights.multiply(vp_x.reshape(Np, 1))
        J = (charge / self.dx) * mat_vel.toarray().sum(axis=0)

        mat_vel_2=mat_vel.multiply(vp_x.reshape(Np, 1))

        P = (charge / self.dx) * mat_vel_2.toarray().sum(axis=0)
        beta=qDm*self.dt/2
        R_vp = self.Evolver_R(vp_x, Bp, beta=beta, c=c)

        mat_Rvel=mat_weights.multiply(R_vp.reshape(Np, 1))
        mat_Rvel_2=mat_Rvel.multiply(R_vp.reshape(Np, 1))

        #J_hat = (charge / self.dx)*mat_Rvel.toarray().sum(axis=0)- (charge / self.dx)*self.theta * self.dt * self.gradient(mat_Rvel_2.toarray().sum(axis=0))
        J_hat = J - self.theta * self.dt*self.gradient(P) # v *v => scalrar


        rho_hat= rho- self.dt * self.theta *self.divergence(J_hat)
        return rho,rho_hat,P, J,J_hat






    def interpolate_fields_to_particles(self,  x_p,field,Np):
        """Interpolate E and B fields to particle positions"""
        # Field Dimension noch nicht bestimmt Ersten sollten Drei sein .
        field_p = np.zeros([3, Np])
        # Process each particle
        for particle_index in range(Np):
            # Particle position in grid coordinates
            x = x_p[particle_index]
            xn = (x / self.dx)
            ix = np.floor(xn).astype(int)  # int Verhalten bei Negativen Zahlen Falsch
            # Arround The World
            # Muss Rho Volumes zuordnen

            # Compute weights for all 8 grid points at once
            for ax in [0, 1]:
                # Periodic boundary conditions
                grid_x = np.mod(ix + ax, self.Nx)
                # Weight based on linear distance (CIC)
                weight = 1 - abs(xn - (ix + ax))
                # Apply shape function and update grid

                field_p[:, particle_index] += field[grid_x] * weight


        return field_p






    def matrix_rhs_equation(self, E, B, J_hat, rho_hat,combi,c):
        return E + combi * ( - 4 * self.pi / c * J_hat) - combi ** 2 * 4 * self.pi * self.gradient(rho_hat)
        return E + combi * (self.curl(B) - 4 * self.pi / c * J_hat) - combi ** 2 * 4 * self.pi * self.gradient(rho_hat)

    def gradient(self,f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * self.dx)
    def divergence(self, j):
        """In 1D ES this is just dj/dx; provided for possible uses."""
        return (np.roll(j, -1) - np.roll(j, 1)) / (2.0 * self.dx)
    def curl(self,v):
        return v*0
    def laplacian(self,f):
        return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (self.dx ** 2)
    def Evolver_R(self,vec,Field,beta,c):

        return vec #Electro static
        gg=vec+beta/c *self.cross(vec,Field)+(beta/c)**2 *self.dot(vec,Field)*Field
        return gg/(1+(beta/c)**2*np.sum(np.abs(Field)**2, axis=0))








    def A_operator(self, E_flat,rho,combi,charge,qDm):
        # Reshape flat vector to [3, Nx, Ny, Nz]
        E_theta = E_flat.reshape( self.Nx)
        beta_ssp = qDm * self.dt / 2
        """Var 1 -- E to Ep"""
        mu_E_theta = np.zeros_like(E, dtype=float)

        """Var 2 -- E in Nx"""
        #alpha_E = self.Evolver_R(self.E_theta, self.B, beta_ssp, c)
        alpha_E=E_theta
        # matrix_lhs_equation(self, E_theta, species, combi, c):

        mu_E_theta += - 4 * self.pi * self.theta * self.dt * beta_ssp * charge * rho * alpha_E
        Av = E_theta + mu_E_theta - combi ** 2 * (self.laplacian(E_theta) + self.laplacian(mu_E_theta))



        return Av.ravel()

    def solveMatrixEquation(self,rhs,prevEtheta,rho,combi,charge,qDm):
        rhs_flat = rhs.ravel()

        A = LinearOperator((self.totalN, self.totalN), matvec=lambda v: self.A_operator(v,rho=rho,combi=combi,charge=charge,qDm=qDm))
        E_theta_flat, info = gmres(A, rhs_flat, x0=prevEtheta.ravel(), rtol=1e-6, restart=30)
        if info == 0:
            if self.dimension==3:
                return E_theta_flat.reshape(3, self.Nx, self.Ny, self.Nz)
            elif self.dimension==1:
                return  E_theta_flat.reshape(self.Nx)
            else:
                raise SyntaxError("Wrong Dim" + str(self.dimension))
        else:
            raise ValueError("GMRES failed to converge")







    def calc_v_hat(self,vp,E_theta_p,beta):

        return vp+beta*E_theta_p[0,:]

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




    def Looper(self, x_i,vp,Np,beta,c):
        # Grid to Particle
        #self.E_theta= np.zeros_like(self.E_theta)
        #self.B= np.zeros_like(self.B)

        E_theta_p = self.interpolate_fields_to_particles(x_i,self.E_theta,Np)  # E
        Bp = self.interpolate_fields_to_particles(x_i, self.B, Np)  # B

        # Calc Velocity

        v_hat = self.calc_v_hat(vp, E_theta_p,beta)  # Here its Important that it is vp

        v_hat = self.Evolver_R(v_hat, Bp,beta = beta, c = c)

        x_i = self.particle_mover(v_hat,x_i, 0.5 * self.dt)

        return self.boundary(x_i), v_hat

    """Advance one full PIC cycle for all species"""
    def step(self):

        c = self.c
        combi = self.combi
        self.Jg*=0
        self.Pg*=0
        self.rhog*=0
        self.rhog_hat *=0
        self.Jg_hat *= 0
        """Moments Gathering for all species"""
        for spp in self.species:
            q_spp = spp["q"]
            charge_spp = spp["charge"]
            x_spp = spp["xp"]
            v_spp = spp["vp"]
            Np_ssp = spp["Np"]
            Bp_ssp = spp["Bp"]
            qDm_ssp= spp["qDm"]
            spp["rho"],spp["rho_hat"],spp["P"],spp["J"], spp["J_hat"] = self.MomentsGathering(x_spp, v_spp, Bp=Bp_ssp, Np=Np_ssp, qDm=qDm_ssp,charge=charge_spp, c=c,)

            self.rhog_hat+=spp["rho_hat"]
            self.Jg_hat+=spp["J_hat"]

            self.rhog += spp["rho"]
            self.Jg += spp["J"]
            self.Pg+=spp["P"]

        """------------------------Moments Finished------------------------------"""
        # Matrix
        rhs = self.matrix_rhs_equation(self.Eg, self.B, self.Jg_hat, self.rhog_hat, combi=combi, c=c)  # TO Vector
        charge_spp=self.species[0]["charge"]
        q_spp=self.species[0]["q_spp"]
        self.E_theta = self.solveMatrixEquation(rhs, self.E_theta, self.rhog, combi=combi, charge_spp=charge_spp,qDm=q_spp)



        # Current Looping for Updated Positions

        for spp in self.species:
            beta_ssp = spp["qDm"] * self.dt / 2
            spp["xp_iter"], spp["vp_iter"] = self.Looper(spp["xp"], spp["vp"], spp["Np"], beta=beta_ssp, c=c)
        count = 0
        for _ in range(5):
            total_error = 0.0
            for spp in self.species:
                xp_old = spp["xp_iter"].copy()
                beta_ssp = spp["qDm"] * self.dt / 2
                spp["xp_iter"], spp["vp_iter"] = self.Looper(spp["xp_iter"], spp["vp_iter"], spp["Np"],
                                                             beta=beta_ssp, c=c)
                total_error += np.linalg.norm(spp["xp_iter"] - xp_old)
            if total_error < 1e-6:
                # print("Iteration stopped after:" + str(count))
                break

            count += 1

        for spp in self.species:


            spp["vp"] = 2 * spp["vp_iter"] - spp["vp"]
            # spp["vp"]=(spp["vp_iter"]-(1-self.theta)*spp["vp"])/self.theta #For all Thetas
            spp["xp"] = self.particle_mover(spp["vp_iter"], spp["xp"], self.dt)
            spp["xp"] = self.boundary(spp["xp"])

            ## For Debugging not needed Elsewhere
            #spp["rho"] = self.deposit_charge(spp["xp"], spp["Np"], q_spp, af)

        # Update Fields
        self.Eg = (self.E_theta - (1 - self.theta) * self.Eg) / self.theta  # For all Theta
        #self.B -= c * c * self.curl(self.E_theta)

        self.t += self.dt


    """
    Analytics for Debugging
    """
    def calcEnergy(self):
        #return np.sum(self.Eg ** 2) * 0.5
        return 0.5 * (self.Eg ** 2).sum() * self.dx


    def calcPotEnergy(self):

        return (0.5 * np.sum(self.Eg ** 2)*self.dx)
    def calcKinEnergy(self):
        el = self.species[0]
        #return np.sum(el["vp"] ** 2) * 0.5
        return 0.5 * el["charge"] / el["qDm"] * np.sum(el["vp"] ** 2)
    def calcMomentum(self):
        el = self.species[0]
        return (el["charge"] / el["qDm"]* np.sum(el["vp"])  )

