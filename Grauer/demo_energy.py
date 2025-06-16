#!/usr/bin/env python
# demonstrates energy spectrum and inertial range

import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.use('TkAgg')  # or use 'Agg' for non-GUI environments

from helper import *
from burgers import *
from statistics import *

# simulation parameters
n = 1024 * 32
x_min = 0.0
x_max = 25.0
dt = 0.001
nu = 0.001

# forcing (Mexican Hat)
l = 1.0
strength = 0.25
chi = lambda x: (1.0 - x ** 2.0 / l ** 2.0) * np.exp(- 0.5 * x ** 2.0 / l ** 2.0)
chihat = lambda k: l ** 3.0 * k ** 2.0 * np.exp(- 0.5 * k ** 2.0 * l ** 2.0)

# creates burgers solver with the simulation and forcing parameters
burgers = Burgers(n, x_min, x_max, dt, nu, chihat, strength)

# statistics
e_spectrum = EnergySpectrum(burgers.x_len, burgers.dk, burgers.uhat.shape[0], nu)

# histogram of Fourier modes
histo = Histogram(-1,1,100)

# plotting
fig = plt.figure("Burgers Energy Cascade Demo")

# plot for the velocity field
u_line = plt.Line2D([], [])
u_plot = fig.add_subplot(3, 1, 1)
u_plot.set_xlabel("$x$")
u_plot.set_ylabel("$u(x, t)$")
u_plot.set_xlim(burgers.x_min, burgers.x_max)
u_plot.set_ylim(-np.pi, np.pi)
u_plot.add_line(u_line)
u_text = u_plot.text(0.02, 0.02, '', transform=u_plot.transAxes)

# plot for the energy spectrum
e_line = plt.Line2D([], [], label="$E(k)$")
e_plot = fig.add_subplot(3, 1, 2)
e_plot.set_xlabel("$k$")
e_plot.set_ylabel("$E(k)$")
e_plot.set_xlim(burgers.k[1], np.max(burgers.k))
e_plot.set_ylim(10.0 ** (-10.0), 10.0 ** (2.0))
e_plot.set_xscale("log")
e_plot.set_yscale("log")
e_plot.add_line(e_line)
e_plot_int_len_line = plt.axvline(x=0, linestyle="--", color="green", label="$k_0$")
e_plot_shock_len_line = plt.axvline(x=0, linestyle=":", color="red", label="$k_s$")
e_plot_k2_line = plt.Line2D([burgers.k[1:]], [(burgers.k[1:]) ** -2], color="orange", label="$k^{-2}$")
e_plot.add_line(e_plot_k2_line)

# plot for the histogram of Fourier modes
h_line = plt.Line2D([], [], label="$pdf$")
h_plot = fig.add_subplot(3, 1, 3)
h_plot.set_xlim(-0.5,0.5)
h_plot.set_ylim(0,10)
h_plot.add_line(h_line)


def step(_):
    changed = []

    for _ in range(100):
        burgers.time_step()

        e_spectrum.feed(burgers.u, burgers.uhat, burgers.dudx)

    u_line.set_data(burgers.x, burgers.u)
    changed.append(u_line)

    u_text.set_text("$t = %.2f$" % burgers.t)
    changed.append(u_text)

    e_line.set_data(burgers.k, e_spectrum.get_mean_spectral_e_density())
    changed.append(e_line)

    integral_wavenumber = e_spectrum.get_integral_wavenumber()
    shock_wavenumber = e_spectrum.get_shock_wavenumber()

    integral_length = e_spectrum.get_integral_length()
    shock_length = e_spectrum.get_shock_length()

    e_plot_int_len_line.set_data([integral_wavenumber, integral_wavenumber], [0, 1])
    e_plot_shock_len_line.set_data([shock_wavenumber, shock_wavenumber], [0, 1])

    e_plot_int_len_line.set_label("$k_0 \\approx " + ("%.2f" % integral_wavenumber) + ", l_0 \\approx " + ("%.2f" % integral_length) + "$")
    e_plot_shock_len_line.set_label("$k_s \\approx " + ("%.0f" % shock_wavenumber) + ", l_s \\approx " + ("%.3f" % shock_length) + "$")

    changed.append(e_plot_int_len_line)
    changed.append(e_plot_shock_len_line)

    histo.categorize(np.array([burgers.uhat[60].real]))
    h_line.set_data(histo.pdf())
    changed.append(h_line)

    changed.append(plt.legend())

    return changed

_ = animation.FuncAnimation(fig, step, init_func=None, interval=1, blit=True, repeat=False)

plt.legend()
plt.show()
