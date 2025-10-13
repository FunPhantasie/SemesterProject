import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from scipy.ndimage import gaussian_filter

from scipy import sparse
from scipy.sparse import linalg





class IPIC_Solver():

    def __init__(self,L=1,NG=1,PPC=20,DT=0.1,ES=True):
        """Only Params"""
        self.theta = 0.8  # Implicit Parameter
        Np= NG * PPC
        # Parameter Conditions
        self.Nx = NG  # gridpoints
        self.dt = DT
        self.Lx = L  # Border
        self.dx = self.Lx / self.Nx
        self.t = 0.0
        self.xg = np.linspace(0, self.Lx - self.dx, self.Nx) + 0.5 * self.dx
        # grid at cell centers
        # Npp per Cell is Electons in Species
        """--------Fields--------"""

        self.totalN = self.Nx  # Total Number of Gridppoints (3 Could be Wrong)
        self.NPpCell = PPC  # NPpCell

        # Global Resulting  Conditions
        self.Eg = np.zeros([self.Nx])  # E[0]
        self.rhog = np.zeros([self.Nx])
        self.Jg = np.zeros([self.Nx])  # Global
        self.Pg = np.zeros([self.Nx])
        self.rhog_hat = np.zeros([self.Nx])
        self.Jg_hat = np.zeros([self.Nx])
        #Fields
        self.B = np.zeros([self.Nx])  # B[2]
        self.E_theta = np.zeros([self.Nx])

        """
        All The Fields and Moments
        """
        # Physical constants
        self.c = 1
        self.pi = np.pi
        self.epsilon_0 = 1.  # Copied Convenient normalization
        self.omega_p = 1.  # Plasma Freq.
        # Handling Multiple Species
        # Initialize the Particles Global Positions and Velocities
        # ---------------Species-------------------------------#

        el = dict(name="e", q=-1, qDm=-1, Np=Np)
        pr = dict(name= "p", q= 1, qDm= 1. / 1836., Np= Np, vp= np.zeros(Np))
        el["charge"] = self.omega_p ** 2 / (el["qDm"] * el["Np"] / self.Lx)
        sc_faktor = - (pr["qDm"] * pr["Np"]) / (el["qDm"] * el["Np"])
        pr["charge"] =self.omega_p ** 2 / (pr["qDm"] * pr["Np"] / self.Lx) * sc_faktor

        pr["xp"] = np.linspace(0, L, pr["Np"], endpoint=False)
        pr["xp"] += self.dx
        pr["xp"] = np.mod(pr["xp"], self.Lx)
        species = [el, pr]

        for sp in species:
            sp["rho"] = np.zeros([self.Nx])
            sp["Fp"] = np.zeros([3, sp["Np"]])
            sp["Ep"] = np.zeros([3, sp["Np"]])
            sp["Bp"] = np.zeros([3, sp["Np"]])
            sp["E_theta_p"] = np.zeros([3, sp["Np"]])
            sp["xp_iter"] = np.zeros([sp["Np"]])
            sp["vp_iter"] = np.zeros([sp["Np"]])

        self.species = species
        self.particle_mover=self.particle_mover1d
        self.combi = self.c * self.theta * self.dt  # Used For Calc
    def MomentsGathering(self, xp, vp, Np, qDm, charge):
        """

        :param xp:
        :param vp:
        :param Bp: Zero Not Implemented
        :param Np:
        :param qDm:
        :param charge:
        :return: rho,rho_hat,P, J,J_hat
        Same Design as in Solution PIC
        """
        mat_weights = self.ShapeFunction(xp, Np)

        rho = (charge / self.dx) * mat_weights.toarray().sum(axis=0)

        mat_vel=mat_weights.multiply(vp.reshape(Np, 1))
        J=np.zeros(self.Nx)
        J += (charge / self.dx) * mat_vel.toarray().sum(axis=0)

        mat_vel_2=mat_vel.multiply(vp.reshape(Np, 1))

        P = (charge / self.dx) * mat_vel_2.toarray().sum(axis=0)
        beta=qDm*self.dt/2
        """Only Relevant if B != 0"""
        #R_vp = self.Evolver_R(vp_x, Bp, beta=beta, c=c)
        #mat_Rvel=mat_weights.multiply(R_vp.reshape(Np, 1))
        #mat_Rvel_2=mat_Rvel.multiply(R_vp.reshape(Np, 1))

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






    def matrix_rhs_equation(self, E, B, species,combi,c):
        rho_hat=0
        J_hat=0
        for i in species:
            rho_hat+=i["rho_hat"]
            J_hat+=i["J_hat"]
        return E + combi * ( - 4 * self.pi / c * J_hat) - combi ** 2 * 4 * self.pi * self.gradient(rho_hat)
        #return E + combi * (self.curl(B) - 4 * self.pi / c * J_hat) - combi ** 2 * 4 * self.pi * self.gradient(rho_hat)

    def gradient(self,f):
        return (np.roll(f, -1) - np.roll(f, 1)) / (2.0 * self.dx)
    def divergence(self, j):
        """In 1D ES this is just dj/dx; provided for possible uses."""
        return (np.roll(j, -1) - np.roll(j, 1)) / (2.0 * self.dx)
    def curl(self,v):
        """
        :param v:
        :return: 0 Doesnt exist in Scalar World
        """
        return v*0
    def laplacian(self,f):
        return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (self.dx ** 2)
    def Evolver_R(self,vec,Field,beta,c):

        return vec #Electro static
        #gg=vec+beta/c *self.cross(vec,Field)+(beta/c)**2 *self.dot(vec,Field)*Field
        #return gg/(1+(beta/c)**2*np.sum(np.abs(Field)**2, axis=0))








    def A_operator(self, E_calc,species,combi):
        """
        :param E_calc: The E_theta to be solved for
        :param rho:
        :param combi:
        :param beta:
        :return:
        """
        R_E = E_calc
        mu_E_calc=0
        for sp in species:
            qDm_s=sp["qDm"]
            beta_s = qDm_s * self.dt / 2
            prefaktor =  4 * self.pi * self.theta * self.dt * beta_s
            mu_E_calc += sp["rho"] * E_calc * prefaktor
        #-----------Variante 1 -- E Ignoriert elementweise mult--------------#
        #mu_E_calc =  np.multiply(E_calc,rhos)*prefaktor
        #E

        #P


        # -----------Variante 2 -- E to Particle--------------#
        #Ep=self.interpolate_fields_to_particles(xp,E_calc,Np)
        #mat_E = mat_weights.multiply(Ep.reshape(Np, 1))
        #mu_E_calc = (charge / self.dx)*prefaktor * mat_E.toarray().sum(axis=0)
        # -----------Variante 3 -- E on Grid--------------#
        Av = E_calc + mu_E_calc - combi ** 2 * (self.laplacian(E_calc) + self.laplacian(mu_E_calc))
        return Av

    def solveMatrixEquation(self,rhs,prevEtheta,combi,species):


        """
            Grid-based scalar field solve:
            Solves A(E) = rhs using GMRES, where A is defined by A_operator.

            Parameters
            ----------
            rhs : ndarray, shape (Nx,)
                Right-hand side vector.
            prevEtheta : ndarray, shape (Nx,)
                Initial guess for GMRES.
            rho : ndarray, shape (Nx,)
                Charge density on the grid.
            combi : float
                c * dt * theta
            charge : float
                Normalized charge (may be unused inside A_operator).
            qDm : float
                Charge-to-mass ratio sign (-1 for electrons, +1 for ions).
        """


        n = self.totalN

        def matvec(x):
            return self.A_operator(x, species=species, combi=combi,  )

        A = LinearOperator((n, n), matvec=matvec, dtype=rhs.dtype)



        E_theta, info = gmres(A, rhs, x0=prevEtheta, rtol=1e-7, restart=30)
        if info != 0:
            raise ValueError(f"GMRES failed to converge (info={info})")

        return E_theta









    def particle_mover1d(self,vp_mid,xp,dt):

        return xp+dt*vp_mid[0,...]

    def boundary(self,x):
        return  np.mod(x, self.Lx)





    def Looper(self, x_i,vp,Np,beta,c):
        # Grid to Particle
        Bp = np.zeros([Np])

        E_theta_p = self.interpolate_fields_to_particles(x_i,self.E_theta,Np)  # E
        #Bp = self.interpolate_fields_to_particles(x_i, self.B, Np)  # B

        # Calc Velocity

        v_hat = vp+beta*E_theta_p[0]  # Here its Important that it is vp

        R_v = self.Evolver_R(v_hat, Bp,beta = beta, c = c)

        x_i = self.particle_mover(R_v,x_i, 0.5 * self.dt)

        return self.boundary(x_i), R_v

    """Advance one full PIC cycle for all species"""
    def Moments(self):
        c = self.c
        combi = self.combi
        #Comparision
        self.Jg *= 0
        self.Pg *= 0
        self.rhog *= 0
        #For Implicit
        self.rhog_hat *= 0
        self.Jg_hat *= 0
        """Moments Gathering for all species"""


        for spp in self.species:
            x_spp = spp["xp"]
            v_spp = spp["vp"]
            q_spp = spp["q"]
            Bp_ssp = spp["Bp"]
            Np_ssp = spp["Np"]
            qDm_ssp = spp["qDm"]
            charge_spp = spp["charge"]
            name_spp = spp["name"]

            params_M = dict(xp=x_spp,vp=v_spp, Np=Np_ssp,qDm=qDm_ssp,charge=charge_spp)
            spp["rho"], spp["rho_hat"], spp["P"], spp["J"], spp["J_hat"] = self.MomentsGathering(**params_M )
            self.rhog += spp["rho"]
            self.rhog_hat += spp["rho_hat"]
            self.Pg += spp["P"]
            self.Jg += spp["J"]
            self.Jg_hat += spp["J_hat"]





        """------------------------Moments Finished------------------------------"""
    def step(self):

        c = self.c
        combi = self.combi
        self.Moments()
        print("Moments Gathering")
        # Matrix
        rhs = self.matrix_rhs_equation(self.Eg, self.B,species=self.species, combi=combi, c=c)  # TO Vector

        self.E_theta = self.solveMatrixEquation(rhs, self.Eg, combi=combi, species=self.species)

        # Update Fields
        self.Eg = (self.E_theta - (1 - self.theta) * self.Eg) / self.theta  # For all Theta
        # self.B -= c * c * self.curl(self.E_theta)

        # Current Looping for Updated Positions

        electrons=self.species[0]
        beta_ssp = electrons["qDm"] * self.dt / 2
        electrons["xp_iter"], electrons["vp_iter"] = self.Looper(electrons["xp"], electrons["vp"], electrons["Np"], beta=beta_ssp, c=c)
        count = 0
        for _ in range(5):
            total_error = 0.0
            xp_old = electrons["xp_iter"].copy()
            beta_ssp = electrons["qDm"] * self.dt / 2
            electrons["xp_iter"], electrons["vp_iter"] = self.Looper(electrons["xp_iter"], electrons["vp_iter"], electrons["Np"],
                                                             beta=beta_ssp, c=c)
            total_error += np.linalg.norm(electrons["xp_iter"] - xp_old)
            if total_error < 1e-6:
                # print("Iteration stopped after:" + str(count))
                break

            count += 1


        electrons["vp"] = 2 * electrons["vp_iter"] - electrons["vp"]
        # spp["vp"]=(spp["vp_iter"]-(1-self.theta)*spp["vp"])/self.theta #For all Thetas
        electrons["xp"] = self.particle_mover(electrons["vp_iter"], electrons["xp"], self.dt)
        electrons["xp"] = self.boundary(electrons["xp"])

        ## For Debugging not needed Elsewhere
        # spp["rho"] = self.deposit_charge(spp["xp"], spp["Np"], q_spp, af)

        self.t += self.dt
    def ShapeFunction(self,x_p,Np):
        xn = (x_p / self.dx)
        ix = np.floor(xn-0.5).astype(int)

        wx = 1 - abs(xn - (ix + 0.5))

        indexes = np.concatenate((ix, ix+1))
        #ix = np.mod(ix, self.Nx)
        NG=self.Nx
        indexes[indexes < 0] += NG
        indexes[indexes > NG - 1] -= NG
        weights= np.concatenate((wx,1-wx))

        p = np.arange(Np)
        prow = np.concatenate((p, p))

        mat = sparse.csc_matrix((weights, (prow, indexes)), shape=(Np, self.Nx))


        return mat



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

