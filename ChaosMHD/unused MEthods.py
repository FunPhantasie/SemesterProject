import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import os
import shutil
import csv
import matplotlib.pyplot as plt
def setup_turbulence(self):
    x = self.x_val
    u = -jnp.cos(x + 1.4) - 1. / 3. * 4 * jnp.cos(2 * x + 2.3)
    return jnp.fft.rfft(u)
def dealias(self, IN_F):
    mask = (jnp.sqrt(jnp.abs(self.k2)) <= float(self.Nx) / 3.)
    return IN_F * mask
def calc_dt(self, u_F, cfl):

    ux = jnp.fft.irfft(u_F).real
    umax = jnp.abs(ux).max()
    # print("umax = ", umax)

    # cfl = umax*dt/dx --> dt = cfl*self.dx/umax
    print("should be: dt =", cfl * self.dx / umax)
    dt = 0.0002  # hard-coded as per original
    return dt