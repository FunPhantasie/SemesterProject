import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
from pyevtk.hl import imageToVTK
import os
import shutil

class RealSpaceGrid:
    def __init__(self, N):
        self.L = 2 * jnp.pi
        self.dx = self.L / N

        xx = jnp.linspace(0.0, self.L, num=N, endpoint=False)
        self.x_val, self.y_val = jnp.meshgrid(xx, xx)

class FourierSpaceGrid:
    def __init__(self, N):
        dk = 1
        kk = dk * jnp.concatenate((jnp.arange(0, N // 2), jnp.arange(-N // 2, 0)))
        self.kx, self.ky = jnp.meshgrid(kk, kk)
        self.k_max = float(N) / 3.0

        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2 = self.k2.at[0, 0].set(1.0)
        self.k2_inv = 1.0 / self.k2
        self.k2 = self.k2.at[0, 0].set(0.0)
        self.k4 = self.k2 * self.k2

class MHD2D(RealSpaceGrid, FourierSpaceGrid):
    def __init__(self, N, nu, out_dir):
        RealSpaceGrid.__init__(self, N)
        FourierSpaceGrid.__init__(self, N)

        self.N = N
        self.nu = nu
        self.out_dir = out_dir
        self.init_output()
        self.setup_turbulence()
    
    def init_output(self):
        self.out_num = 0
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir)

    def setup_turbulence(self):
        x, y = self.x_val, self.y_val

        # Orszag-Tang
        # phi = 2 (cos(x) + cos(y)) --> om = -2 (cos(x) + cos(y))
        # psi = 2 cos(2x) + cos(2y) --> j  = -4 (2 cos(2x) + cos(2y))
        # --> omp = -2 (cos(x) + cos(y)) - 4 (2 cos(2x) + cos(2y))
        # --> omm = -2 (cos(x) + cos(y)) + 4 (2 cos(2x) + cos(2y))
        # omp = -(jnp.cos(x) + jnp.cos(y)) - (2*jnp.cos(2*x) + jnp.cos(y))
        # omm = -(jnp.cos(x) + jnp.cos(y)) + (2*jnp.cos(2*x) + jnp.cos(y))

        # Biskamp-Welter
        # phi = cos(x+1.4) + cos(y+2.0)        --> om = -cos(x+1.4) - cos(y+2.0)
        # psi = 1/3 (cos(2x+2.3) + cos(y+6.2)) --> j  = -1/3 (4 cos(2x+2.3) + cos(y+6.2))
        # --> omp = -cos(x+1.4) - cos(y+2.0) - 1/3 (4 cos(2x+2.3) + cos(y+6.2))
        # --> omm = -cos(x+1.4) - cos(y+2.0) + 1/3 (4 cos(2x+2.3) + cos(y+6.2))
        omp = -jnp.cos(x + 1.4) - jnp.cos(y + 2.0) - 1./3. * (4 * jnp.cos(2 * x + 2.3) + jnp.cos(y + 6.2))
        omm = -jnp.cos(x + 1.4) - jnp.cos(y + 2.0) + 1./3. * (4 * jnp.cos(2 * x + 2.3) + jnp.cos(y + 6.2))

        self.om_F = jnp.fft.fft2(jnp.stack([omp,omm],axis=0),axes=(-2,-1))

    def delta_F(self):
        dy = dx = self.dx
        xb = yb = 0
        return 1. / (dx * jnp.exp(1.j * self.kx * xb) * dy * jnp.exp(1.j * self.ky * yb))

    def dealias(self, IN_F):
        # mask = (jnp.abs(self.kx) <= float(self.N) / 3.) & (jnp.abs(self.ky) <= float(self.N) / 3.)
        mask = (jnp.sqrt(jnp.abs(self.k2)) <= float(self.N) / 3.)
        return IN_F * mask

    def get_velocity_from__vorticity_F(self, om_F):
            u_Fx =  1.j * self.ky * self.k2_inv * om_F
            u_Fy = -1.j * self.kx * self.k2_inv * om_F
            u_Fx = u_Fx.at[0, 0].set(0.)
            u_Fy = u_Fy.at[0, 0].set(0.)
            return u_Fx, u_Fy
    
    def calc_dt(self, cfl):
        om_F = 0.5 * (self.om_F[0] + self.om_F[1])
        ux_F, uy_F = self.get_velocity_from__vorticity_F(om_F)

        ux = jnp.fft.ifft2(ux_F).real
        uy = jnp.fft.ifft2(uy_F).real

        umax = jnp.maximum(jnp.abs(ux).max(), jnp.abs(uy).max())
        print("umax = ", umax)

        # cfl = umax*dt/dx --> dt = cfl*self.dx/umax
        print("should be: dt =",cfl*self.dx/umax)
        dt = 0.0002  # hard-coded as per original
        return dt

    def calc_nonlinearity_F(self, om_F):
        omp_F = om_F[0]; omm_F = om_F[1]
        kx, ky = self.kx, self.ky

        zp_x_F, zp_y_F = self.get_velocity_from__vorticity_F(self.dealias(omp_F));
        zp_x = jnp.fft.ifft2(zp_x_F); zp_y = jnp.fft.ifft2(zp_y_F)
        zm_x_F, zm_y_F = self.get_velocity_from__vorticity_F(self.dealias(omm_F))
        zm_x = jnp.fft.ifft2(zm_x_F); zm_y = jnp.fft.ifft2(zm_y_F)
        # calculate nonlinearities
        zp_xzm_x_F = jnp.fft.fft2(zp_x * zm_x)
        zp_xzm_y_F = jnp.fft.fft2(zp_x * zm_y)
        zp_yzm_x_F = jnp.fft.fft2(zp_y * zm_x)
        zp_yzm_y_F = jnp.fft.fft2(zp_y * zm_y)

        RHS_omp_F = kx*ky*zp_xzm_x_F + ky*ky*zp_xzm_y_F - kx*kx*zp_yzm_x_F - kx*ky*zp_yzm_y_F
        RHS_omm_F = kx*ky*zp_xzm_x_F + ky*ky*zp_yzm_x_F - kx*kx*zp_xzm_y_F - kx*ky*zp_yzm_y_F
  
        return jnp.stack([RHS_omp_F,RHS_omm_F],axis=0)
    
    def prop(self, dt):
        return jnp.exp(-self.nu*self.k2*dt)
    
    def stepEuler(self):
        cfl = 0.3 # good choice for Shu-Osher TVD RK
        dt = self.calc_dt(cfl)

        RHS1_om_F = self.calc_nonlinearity_F(self.om_F)
        self.om_F = (self.om_F + dt*RHS1_om_F)*self.prop(dt)

        return dt
    
    def stepHeun(self):
        cfl = 0.3 # good choice for Shu-Osher TVD RK
        dt = self.calc_dt(cfl)

        RHS1_om_F = self.calc_nonlinearity_F(self.om_F)
        om_F_1 = (self.om_F + dt*RHS1_om_F)*self.prop(dt)

        RHS2_om_F = self.calc_nonlinearity_F(om_F_1)
        self.om_F = self.om_F*self.prop(dt) + 0.5*dt*(RHS1_om_F*self.prop(dt) + RHS2_om_F)

        return dt
     
    def stepShuOsher(self):
        cfl = 0.3 # good choice for Shu-Osher TVD RK
        dt = self.calc_dt(cfl)

        k_viscosity = self.k2 # normal viscosity

        RHS1_om_F = self.calc_nonlinearity_F(self.om_F)
        om_F_1 = (self.om_F + dt * RHS1_om_F) * self.prop(dt)

        RHS2_om_F = self.calc_nonlinearity_F(om_F_1)
        omp_F_2 = (self.om_F + 0.25*dt*RHS1_om_F)*self.prop(0.5*dt) + 0.25*dt*RHS2_om_F*self.prop(-0.5*dt)
  
        RHS3_om_F = self.calc_nonlinearity_F(om_F_2)
        self.om_F = (self.om_F + 1./6.*dt*RHS1_om_F)*self.prop(dt) + 1./6.*dt*RHS2_om_F + 2./3.*dt*RHS3_om_F*self.prop(-0.5*dt)
 
        return dt
    
    def print_vtk(self):
        file_name = self.out_dir + '/step_' + str(self.out_num)
        self.out_num += 1

        om_F = 0.5 * (self.om_F[0]+self.om_F[1])
        j_F  = 0.5 * (self.om_F[0]-self.om_F[1])
        Ux_F, Uy_F = self.get_velocity_from__vorticity_F(om_F)
        Bx_F, By_F = self.get_velocity_from__vorticity_F(j_F)
        Ux_cpu = np.asarray(jnp.fft.ifft2(Ux_F).real.block_until_ready())
        Uy_cpu = np.asarray(jnp.fft.ifft2(Uy_F).real.block_until_ready())
        Bx_cpu = np.asarray(jnp.fft.ifft2(Bx_F).real.block_until_ready())
        By_cpu = np.asarray(jnp.fft.ifft2(By_F).real.block_until_ready())
        om_cpu = np.asarray(jnp.fft.ifft2(om_F).real.block_until_ready())
        j_cpu  = np.asarray(jnp.fft.ifft2( j_F).real.block_until_ready())

        om_out = om_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        j_out  =  j_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        Ux_out = Ux_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        Uy_out = Uy_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        Bx_out = Bx_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        By_out = By_cpu.reshape((self.N,self.N,1), order = 'C').copy()
        imageToVTK(
            file_name,
            cellData = {
                'om' : om_out,
                'j'  : j_out,
                'Ux' : Ux_out,
                'Uy' : Uy_out,
                'Bx' : Bx_out,
                'By' : By_out,
            },
            pointData = {} )

