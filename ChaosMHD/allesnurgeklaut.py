import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import os
import shutil
import csv
import matplotlib.pyplot as plt
jax.config.update("jax_disable_jit", True)




class TimeGrid:
    def __init__(self, Nt, t_begin=-10.0,t_end=0, talpha=1.0):
        self.Nt = Nt
        tb, te = t_begin, t_end

        # uniform grid in NumPy first
        t = np.linspace(tb, te, Nt)

        # exponential clustering toward te=0
        t_A = (te - tb) / (np.exp(-talpha * te) - np.exp(-talpha * tb))
        t_B = tb - t_A * np.exp(-talpha * tb)
        t = t_A * np.exp(-talpha * t) + t_B

        # force exact endpoints
        t[0] = tb
        t[-1] = te

        # compute dt and drop last element
        dt = np.roll(t, -1) - t
        dt = dt[:-1]

        # store as JAX arrays
        self.t = jnp.array(t)
        self.dt = jnp.array(dt)


class RealSpaceGrid2D:
    def __init__(self, Nx,Ny, L=2*jnp.pi):
        self.Nx = Nx
        self.Ny =Ny
        self.L = L
        self.dx = L / Nx
        self.dy =L/Ny
        x1d = jnp.linspace(0.0, L, num=Nx, endpoint=False)

        y1d = jnp.linspace(0.0, L, num=Ny, endpoint=False)
        self.x, self.y = jnp.meshgrid(x1d, y1d, indexing='ij')



class FourierSpaceGrid2D:
    def __init__(self, Nx,Ny):
        #Real und Img Fourier
        kk_x = jnp.fft.fftfreq(Nx, d=1.0/Nx)
        kk_y = jnp.fft.fftfreq(Ny, d=1.0/Ny)
        kx, ky = jnp.meshgrid(kk_x, kk_y, indexing='ij')
        # real wavenumbers (no i); we multiply by 1j during ops
        self.kx = (2*jnp.pi/Nx) * kx
        self.ky = (2*jnp.pi/Ny) * ky
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2inv = jnp.where(self.k2 > 0, 1.0/self.k2, 0.0)
        # 2/3 rule dealias mask
        #self.mask = (jnp.sqrt(jnp.abs(self.k2)) <= float(self.Nx) / 3.)  # square
        kabs = jnp.sqrt(self.k2) * (Nx / (2 * jnp.pi))
        kcut = (2.0 / 3.0) * (Nx / 2.0)
        self.mask = (kabs <= kcut).astype(jnp.float32)

class MHDInstanton2D(TimeGrid, RealSpaceGrid2D, FourierSpaceGrid2D):
    def __init__(self,F, mu, a, out_dir,Nx=128,Ny=128,t_begin=-10.0,t_end=0,Nt=7, nu=1e-3, talpha=1.0 ):
        TimeGrid.__init__(self, Nt=Nt, t_begin=t_begin,t_end=t_end ,talpha=talpha)
        RealSpaceGrid2D.__init__(self, Nx=Nx,Ny=Ny)
        FourierSpaceGrid2D.__init__(self, Nx=Nx,Ny=Ny)


        self.nu = nu  # viscosity
        self.F = F  # linear penalty
        self.mu = mu  # quadratic penalty
        self.a = a  # terminal constraint target value
        self.out_dir = out_dir
        self.init_output()

    def init_output(self):
        self.out_num = 0
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)


    def solve_forward_F(self, p_base_F, omega_base_F, dt):
        Nt = self.Nt
        omega_base_F_prev = omega_base_F[:,0] #plus/minus,time,x,y
        for i in range(1, Nt): #Von 1 bis N-1 , O wurde schon ausgeführt, Omega hat N Einträge alle updates
            delta_t = dt[i - 1]
            omega_base_F_prev = self.force_symmetry(self.step_forward_F(p_base_F[:,i - 1], omega_base_F_prev, delta_t))
            omega_base_F = omega_base_F.at[:,i].set(omega_base_F_prev)
        return omega_base_F

    def step_forward_F(self, p_base_F, omega_base_F, dt):
        RHS = self.RHS_omega_F(p_base_F, omega_base_F)
        ret = self.prop_F(dt) * (omega_base_F + RHS * dt) # Minus sigh changed
        return ret

    def RHS_omega_F(self, p_base_F, omega_base_F):

        kx, ky = self.kx, self.ky
        k = [kx, ky]
        k2inv=self.k2inv

        T_p_m_F,T_m_p_F = self.getTensors(k2inv=k2inv, kx=kx, ky=ky,omega_base_F= omega_base_F)
        diffusion_p_F = self.Diffusion(T_m_p_F,k)
        diffusion_m_F = self.Diffusion(T_p_m_F,k)
        diffusion_base_f=jnp.stack([diffusion_p_F,diffusion_m_F],axis=0)

        return -diffusion_base_f + self.convolution_F(self.chi_F(), p_base_F)

    def chi_F(self):
        if not hasattr(self, "_chi_F_first_call_done"):
            print("chi_F needs to be tested!")
            self._chi_F_first_call_done = True
        #Chat GPT
        kx = self.kx  # assume 2D arrays matching chi_F shape (or broadcastable)
        ky = self.ky
        k2 = self.k2
        Nx, Ny = self.Nx, self.Ny

        lam = .5 ** 2. #BEfore 8
        chi_F = lam * (2.0 * jnp.pi) * k2 * jnp.exp(-0.5 * lam * k2)
        # circular 2/3 de-alias mask
        k = jnp.sqrt(k2)
        k_max = jnp.max(k)
        mask = k < (2.0 / 3.0) * k_max

        # normalize OVER THE SAME MASK (exclude DC implicitly via k2 factor)
        eps = 1.0  # total energy target
        sigma = jnp.sum(jnp.where(mask, chi_F, 0.0))
        chi_F = jnp.where(sigma != 0.0, chi_F * (eps * (2.0 / 3.0) * Nx * Ny / sigma), chi_F)

        # apply mask after normalization (keeps total energy consistent)
        chi_F = jnp.where(mask, chi_F, 0.0)

        # if you truly need a 2-component forcing, keep the stack; otherwise return chi_F
        return jnp.stack([chi_F, chi_F], axis=0)

    def dealias(self, IN_F):
        # mask = (jnp.abs(self.kx) <= float(self.N) / 3.) & (jnp.abs(self.ky) <= float(self.N) / 3.)

        return IN_F * self.mask
    def getTensors(self,k2inv,kx,ky,omega_base_F):
        w_plus, w_minus = self.dealias(-omega_base_F[0]), self.dealias(-omega_base_F[1])
        z_plus_x_=jnp.fft.ifft2(-1.j* w_plus * ky*k2inv)
        z_plus_y_=jnp.fft.ifft2(1.j*w_plus * kx*k2inv)
        z_minus_x_=jnp.fft.ifft2(-1.j*w_minus * ky*k2inv)
        z_minus_y_=jnp.fft.ifft2(1.j*w_minus * kx*k2inv)
        z_plus = jnp.stack(
            [z_plus_x_,  # dy w+
             z_plus_y_])
        z_minus = jnp.stack(
            [z_minus_x_,  # dy w+
            z_minus_y_  ])
        T_p_m = jnp.einsum("i...,j...->ij...", z_plus, z_minus)
        T_m_p = jnp.einsum("i...,j...->ij...", z_minus, z_plus)
        # back to Fourier space over the last two axes (no Python loops)
        T_p_m_F = self.dealias(jnp.fft.fft2(T_p_m, axes=(-2, -1))) #Along axis -2 and -1
        T_m_p_F = self.dealias(jnp.fft.fft2(T_m_p, axes=(-2, -1)))
        return T_p_m_F,T_m_p_F


    def convolution_F(self, f_F, g_F):
        print("Copied without Brain from Origin")#DOnt get what happens
        kx = self.kx
        ky = self.ky
        Ny = self.Ny
        Nx = self.Nx
        xb = 0.0
        yb=0.0
        phasi=kx*xb+ky*yb
        return 2. * jnp.pi * f_F * g_F / (Nx*Ny) * jnp.exp(1.j *phasi)
    def Diffusion(self,T_ij,k):

        result = 0.0
        for i in range(2):  # maybe update
            for j in range(2):
                jp1 = (j + 1) % 2  # j+1 modulo n
                overflow = (j + 1) >= 2  # True falls Überschlag
                sign = -1 if overflow else 1  # Faktor bei Überschlag

                term = -1 * T_ij[i, j] * k[i] * k[jp1] * sign
                result += term
        return result

    def omega_Diffusion(self, omega_base_F, back_prop_F, kx, ky, k2inv):
        # z^± from ω^±  (Fourier -> real)
        w_plus = self.dealias(omega_base_F[0])
        w_minus = self.dealias(omega_base_F[1])
        zpx = jnp.fft.ifft2(1j * ky * k2inv * w_plus)  # z_x^+
        zpy = jnp.fft.ifft2(-1j * kx * k2inv * w_plus)  # z_y^+
        zmx = jnp.fft.ifft2(1j * ky * k2inv * w_minus)  # z_x^-
        zmy = jnp.fft.ifft2(-1j * kx * k2inv * w_minus)  # z_y^-

        # second-derivative multipliers
        kxx = -(kx * kx);
        kxy = -(kx * ky);
        kyy = -(ky * ky)

        # p^+ second derivatives (to real)
        p_plus = self.dealias(back_prop_F[0])
        pxx_p = jnp.fft.ifft2(kxx * p_plus)
        pxy_p = jnp.fft.ifft2(kxy * p_plus)
        pyy_p = jnp.fft.ifft2(kyy * p_plus)

        # p^- second derivatives (to real)
        p_minus = self.dealias(back_prop_F[1])
        pxx_m = jnp.fft.ifft2(kxx * p_minus)
        pxy_m = jnp.fft.ifft2(kxy * p_minus)
        pyy_m = jnp.fft.ifft2(kyy * p_minus)

        # ----- rhs_plus = self_plus + cross_plus -----
        nplus_terms_omp = -(1j * kx * k2inv * jnp.fft.fft2(zmx * pxx_p + zmy * pxy_p)
                      + 1j * ky * k2inv * jnp.fft.fft2(zmx * pxy_p + zmy * pyy_p))
        nminus_terms_omp = (+1j * ky * k2inv * jnp.fft.fft2(zmy * pxx_m)
                        - 1j * ky * k2inv * jnp.fft.fft2(zmx * pxy_m)
                      - 1j * kx * k2inv * jnp.fft.fft2(zmy * pxy_m)
                      + 1j * kx * k2inv * jnp.fft.fft2(zmx * pyy_m))
        rhs_plus = nplus_terms_omp + nminus_terms_omp

        # ----- rhs_minus = self_minus + cross_minus -----
        nplus_terms_omm = -(1j * kx * k2inv * jnp.fft.fft2(zpx * pxx_m + zpy * pxy_m)
                       + 1j * ky * k2inv * jnp.fft.fft2(zpx * pxy_m + zpy * pyy_m))
        nminus_terms_omm = (+1j * ky * k2inv * jnp.fft.fft2(zpy * pxx_p)
                       - 1j * kx * k2inv * jnp.fft.fft2(zpy * pxy_p)
                       - 1j * ky * k2inv * jnp.fft.fft2(zpx * pxy_p)
                       + 1j * kx * k2inv * jnp.fft.fft2(zpx * pyy_p))
        rhs_minus = nplus_terms_omm + nminus_terms_omm

        return jnp.stack([rhs_plus, rhs_minus], axis=0)

    def prop_F(self, dt):
        return jnp.exp(-self.nu * self.k2 * dt)

    def force_symmetry(self, f_F):
        #raise NotImplementedError("Yes")
        return  f_F
        #return 1.j * f_F.imag
    def solve_adjoint_F(self,  back_prop_F,omega_base_F, dt):
        Nt = self.Nt

        back_prop_F_prev = self.gradObsT_F(omega_base_F) #Kopiert
        back_prop_F = back_prop_F.at[:,-1].set(back_prop_F_prev) #Backwards
        for i in range(1, Nt):
            delta_t = dt[Nt - i - 1]
            back_prop_F_prev = self.step_adjoint_F(back_prop_F_prev, omega_base_F[:,Nt - i], delta_t)
            back_prop_F = back_prop_F.at[:,Nt - i - 1].set(back_prop_F_prev)
        return back_prop_F

    def step_adjoint_F(self, p_base_F, omega_base_F, dt):
        RHS = self.RHS_adjoint_F(p_base_F, omega_base_F)
        ret = self.prop_F(dt) * (p_base_F - RHS * dt) # Check Minus Sign
        return ret
    def RHS_adjoint_F(self, p_base_F, omega_base_F):
        kx, ky = self.kx, self.ky
        k2=self.k2
        k2inv = self.k2inv


        RHS_p_F = - self.omega_Diffusion(omega_base_F=omega_base_F, back_prop_F=p_base_F, kx=kx, ky=ky, k2inv=k2inv)
        return RHS_p_F
    def gradObsT_F(self,omega_base_F):

        gradObsT_F = -(self.F - self.mu * self.observable(omega_base_F)) * self.delta_F() #vz
        # for Obs=j_z

        gradObsT_F = jnp.stack([gradObsT_F, gradObsT_F], axis=0)
        return gradObsT_F
    def delta_F(self): # periodic bd, 2D
        dx, dy = self.dx, self.dy
        xb = 0; yb = 0
        phase = jnp.exp(1j * (self.kx * xb + self.ky * yb))
        normalization = 1.0 / (dx * dy)
        return normalization * phase  # shape (Nx, Ny)
    def delta_x_F(self):
        #Propably Wrong
        dx,dy = self.dx,self.dy
        xb,yb = 0,0
        kx,ky= self.kx,self.ky
        phasi = kx * xb + ky * yb
        return (1.j * self.k2) / (dx*dy) * jnp.exp(1.j * phasi)

    def observable(self, omega_base_F):  # NOt TODO: current density? vorticity?
        j_z_F = 0.5 * (omega_base_F[0] - omega_base_F[1])
        j_z = jnp.fft.ifft2(j_z_F, axes=(-2, -1)).real #Along Axis -2 and -1
        return j_z[-1][self.Nx // 2, self.Ny // 2]  # return value at center of grid, last time step


def run_CS(sim: MHDInstanton2D, p_base_F,omega_base_F,  dt):
    A = 0.
    epsilon = 1.e-8
    sigma = 1.
    delta_A = epsilon + 1
    step=0
    while (jnp.abs(delta_A) > epsilon):
        back_prop_F = sim.solve_adjoint_F(p_base_F,omega_base_F, dt)#Mostly Implemented
        S=1 #Not Implemented

        diff=(p_base_F- back_prop_F)

        p_base_F -=  sigma * diff # I - theta (I-BackProp)
        omega_base_F = sim.solve_forward_F(p_base_F, omega_base_F, dt) #Symm Not Implemented
        #Zum TEsten Forward Fertig
        A_bckp = A
        A = sim.observable(omega_base_F)
        print("F = ", sim.F, "A =", A, "S = ", S, "max grad",  jnp.max(jnp.abs(diff)), "sigma = ", sigma)
        delta_A_bckp = delta_A
        delta_A = (A_bckp - A) / sigma / A_bckp
        if (delta_A_bckp * delta_A < 0.):
            sigma *= 0.8  # 0.95
        step += 1
        print("# iters = ", step)

    with open(sim.out_dir + '/FAS.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sim.F, A, S])


    return p_base_F, omega_base_F


"""
#####***#####
Params
####***######
"""
N = 30
nu = 1.# normal viscosity
t_end = 0
t_begin=-1.0
Nt=100
talpha=1.0
Nx , Ny=N,N
F=-.00001
dF = -.0005
mu=0
a=30
out_dir="output_burgers"
"""
#####***#####
Ending
####***######
"""
mhd2d = MHDInstanton2D(Nx=Nx,Ny=Ny,t_begin=t_begin,t_end=t_end ,Nt=Nt, nu=nu, talpha=talpha, F=F, mu=mu, a=a, out_dir=out_dir)
"""
Variabeln Omega
VAriabeln P
"""
omega_prefab_F = jnp.zeros((mhd2d.Nt, mhd2d.Nx,mhd2d.Ny), dtype=jnp.complex64)
omega_base_F=jnp.stack([omega_prefab_F, omega_prefab_F], axis=0)
p_base_F = jnp.zeros_like(omega_base_F)





while mhd2d.F < 100:
    p_base_F, omega_base_F = run_CS(mhd2d, p_base_F= p_base_F,omega_base_F=omega_base_F, dt=mhd2d.dt)
    mhd2d.F += dF

# plt.plot(np.fft.irfft(u_F[-1])); plt.show()
