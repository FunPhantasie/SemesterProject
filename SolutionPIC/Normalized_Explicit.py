"""
    1D electrostatic particle-in-cell solver for electron two-stream instability.

    Les Houches by G. Lapenta.

    V. Olshevsky: sya@mao.kiev.ua

    mod by MEI

"""
import os, time
# start_time = time.clock()
import numpy as np
import pylab as plt
import matplotlib.patches as mpatches
from scipy import sparse
from scipy.sparse import linalg
import matplotlib as mpl

mpl.use('TkAgg')
# Output folder
path = 'Explicit_Solution'
if not os.path.exists(path):
   os.makedirs(path)

# Set plotting parameters
params = {'axes.labelsize': 'large',
          'xtick.labelsize': 'medium',
          'ytick.labelsize': 'medium',
          'font.size': 15,
          'font.family': 'sans-serif',
          'text.usetex': False,
          'mathtext.fontset': 'stixsans', }
plt.rcParams.update(params)
## Switch on interactive plotting mode
plt.ion()

# Simulation parameters
# original parameters -- long!!!
# L = 20*np.pi #20*np.pi # Domain size
# DT = 0.005 # Time step
# NT = 50000  # Number of time steps
# TOut = round(NT/500) # Output period
# verbose = True
# NG = 320  # Number of grid cells
# N = NG * 20 # Number of particles
# WP = 1. # Plasma frequency
# QM = -1. # Charge/mass ratio
# V0 = 0.9 # Stream velocity
# VT = 0.0000001 # Thermal speed
#  endoriginal parameters -- long!!!

# this is fast, but does not conserve energy in the end
# change parameters for better energy conservation
L = 2.5 * np.pi  # 20*np.pi #20*np.pi # Domain size
DT = 0.005 * 10  # 0.005 # Time step
NT = 500  # 50000  # Number of time steps
TOut = round(NT / 25)  # Output period
verbose = True
NG = 40  # 80 #320  # Number of grid cells
PPC = 20  # number of particles per cell
Np = NG * PPC  # total number of particles
WP = 1.  # Plasma frequency
QM = -1.  # Charge/mass ratio; normalized to to electron mass
V0 = 0.5  # 0.9 # Stream velocity
VT = 0.0000001  # Thermal speed

# perturbation
XP1 = 1.0
mode = 1

dx = L / NG  # Grid step

# with ES=true, E from Gauss
# with ES=false, E from Ampere
# ES= True
ES = False

maxE = []

# remember that N= Ng *PPC
# same as iPic3D, where it is Q = signed_rho * dV/ PPC, which is the same
# summing over (average) number of particles per cell:
# PPC * signed_rho *dV / PPC= signed rho *dV
# that's why we need to divide by dV when accumulating
Q = WP ** 2 / (QM * Np / L)

# this is the ion rho_0
# with WP=1, QOM= -1, rho_back=1
rho_back = -Q * Np / L  # proton background density

# Auxilliary vectors / Hilfs-
p = np.concatenate([np.arange(Np), np.arange(Np)])  # Some indices up to N 0 bis np-1 und dann nochmal
# Poisson is a diagonal matrix with -2 on the diag; -1 above and below used for \nabla^2
Poisson = sparse.spdiags(([1, -2, 1] * np.ones((1, NG - 1), dtype=int).T).T, \
                         [-1, 0, 1], NG - 1, NG - 1)
Poisson = Poisson.tocsc()

# Cell center coordinates
xg = np.linspace(0, L - dx, NG) + dx / 2

# electron initialization
# electrons initialized at fixed distance
xp = np.linspace(0, L - L / Np, Np).T  # Particle positions
vp = VT * (1 - VT ** 2) ** (-0.5) * np.random.randn(Np)  # Particle momentum, initially Maxwellian
pm = np.arange(Np) #Stepsize 1 Steps Np
# to get the two streams
pm = 1 - 2 * np.mod(pm + 1, 2)
vp += pm * (V0 * (1 - V0 ** 2) ** (-0.5))  # Momentum + stream velocity

# Add electron perturbation to excite the desired mode
# this perturbation is added on the position, so it will show up in the density
xp += XP1 * (L / Np) * np.sin(2 * np.pi * xp / L * mode)
xp[np.where(xp < 0)] += L
xp[np.where(xp >= L)] -= L

histEnergy, histPotE, histKinE, histMomentum, t = [], [], [], [], []

if verbose:
    plt.figure(1, figsize=(16, 9))

# Main cycle
for it in range(NT + 1):  # p3
    # for it in range(5):      #p3
    # p3
    ### PARTICLE PUSHER - part 1
    # update particle position xp
    # dx/dt= v, forward difference
    # (x^(n+1)- x^n)/ dt= v^n
    xp += vp * DT
    # Periodic boundary condition
    xp[np.where(xp < 0)] += L
    xp[np.where(xp >= L)] -= L

    # Project particles->grid
    # grid point involved
    g1 = np.floor(xp / dx - 0.5)
    g = np.concatenate((g1, g1 + 1))
    # particle fraction per grid point
    fraz1 = 1 - np.abs(xp / dx - g1 - 0.5)
    fraz = np.concatenate((fraz1, 1 - fraz1))
    # fix boundary conditions for the accumulation
    g[np.where(g < 0)] += NG
    g[np.where(g > NG - 1)] -= NG

    # particle (N) contribution to grid points (NG)
    # syntax: csc_matrix((data, (row, col)), shape=(Nrow, Ncol))
    # mat.toarray()[p] tells you to which grid point the particle p contributes
    mat = sparse.csc_matrix((fraz, (p, g)), shape=(Np, NG))

    # rho_e: electron density
    # mat.toarray().sum(axis=0)  gives ~ particles per cell
    # mat.toarray().sum(axis=0).sum(axis=0) is the number of particles
    # since Q= signed rho_e * DV/ PPC,
    # rho_e \sim -1
    rho_e = Q / dx * mat.toarray().sum(axis=0)
    # total density with neutralizing ions

    # total densirt, rho ~0
    rho = rho_e + rho_back

    #### calculation of the other moments, Jx and Pxx, needed for implicit
    ### q.v_x
    mat2 = mat.multiply(vp.reshape(Np, 1))
    J_ex = Q / dx * mat2.toarray().sum(axis=0)

    ### q.v_x.v_x - this is the stress tensor, not the thermal pressure
    ### you need this in the calculation
    mat3 = mat2.multiply(vp.reshape(Np, 1))
    P_exx = Q / dx * mat3.toarray().sum(axis=0)

    if (ES):
        # Compute electric field potential
        # electric potential
        # eq: - lap \phi= \rho/ \epsilon_0
        # lap= Poisson/ dx^2
        # here: Poisson \Phi= - (\rho/ epsilon_0) dx**2
        Phi = linalg.spsolve(Poisson, -dx ** 2 * rho[0:NG - 1])
        # fix BC
        Phi = np.concatenate((Phi, [0]))
        # Electric field on the grid
        # E = - d Phi/dx rho, central diff
        Eg = (np.roll(Phi, 1) - np.roll(Phi, -1)) / (2 * dx)
    else:
        if (it == 0):
            Eg = np.zeros(len(rho))
        # calculate E from Ampere: dE/ dt= - \mu_0/ \epsilon_0 J
        Eg -= DT * J_ex

    maxE.append(np.max(Eg))
    # mover, part 2
    # interpolation grid->particle and velocity update
    # dv/dt= q/m E_p; E_p interpolated at particle position
    # mat is the field to particle interpolation
    vp += mat * QM * Eg * DT

    Etot = 0.5 * (Eg ** 2).sum() * dx
    histEnergy.append(Etot)
    histPotE.append(0.5 * (Eg ** 2).sum() * dx)
    histKinE.append(0.5 * Q / QM * (vp ** 2).sum())
    histMomentum.append(Q / QM * vp.sum())
    t.append(it * DT)

    if (np.mod(it, TOut) == 0) and verbose:
        # Phase space
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(xp[0:-1:2], vp[0:-1:2], s=0.5, marker='.', color='blue')
        plt.scatter(xp[1:-1:2], vp[1:-1:2], s=0.5, marker='.', color='red')
        plt.xlim(0, L)
        plt.ylim(-V0 * 2, V0 * 2)
        plt.xlabel('X')
        plt.ylabel('P')
        plt.legend((mpatches.Patch(color='w'),), (r'$\omega_{pe}t=$' + str(DT * it),), loc=1, frameon=False)

        # Electric field
        plt.subplot(2, 2, 2)
        plt.xlim(0, L)
        plt.xlabel('X')
        plt.plot(xg, Eg, label='E', linewidth=2)
        plt.legend(loc=1)

        # Energies
        plt.subplot(2, 2, 3)
        plt.xlim(0, NT * DT)
        plt.xlabel('time')
        plt.yscale('log')
        plt.plot(t, histEnergy, label='Total Energy', linewidth=2)
        plt.plot(t, histPotE, label='Field', linewidth=2)
        plt.plot(t, histKinE, label='Kinetic', linewidth=2)
        plt.legend(loc=4)

        # Momentum
        plt.subplot(2, 2, 4)
        plt.xlim(0, NT * DT)
        plt.xlabel('time')
        plt.plot(t, histMomentum, label='Momentum', linewidth=2)
        plt.legend(loc=1)
        # plt.pause(0.01)
        print(it)
        plt.savefig(os.path.join(path, 'twostream_ES%i_%3.3i' % (ES, it / TOut,) + '.png'))

        plt.clf()

        # rho
        plt.subplot(2, 2, 1)
        plt.xlim(0, L)
        plt.xlabel('X')
        plt.plot(xg, rho, label='rho tot', linewidth=2)
        plt.legend(loc=1)

        # Electric field
        plt.subplot(2, 2, 2)
        plt.xlim(0, L)
        plt.xlabel('X')
        plt.plot(xg, rho_e, label='rho E', linewidth=2)
        plt.legend(loc=1)

        plt.subplot(2, 2, 3)
        plt.xlim(0, L)
        plt.xlabel('X')
        plt.plot(xg, J_ex, label='J ex', linewidth=2)
        plt.legend(loc=1)

        # Electric field
        plt.subplot(2, 2, 4)
        plt.xlim(0, L)
        plt.xlabel('X')
        plt.plot(xg, P_exx, label='P exx', linewidth=2)
        plt.legend(loc=1)

        print(it)
        plt.savefig(os.path.join(path, 'twostream_MOM_ES%i_%3.3i' % (ES, it / TOut,) + '.png'))

## use this to calculate the growth rate of the instability
## fit the linear phase, compare the slopes
plt.clf()
plt.xlim(0, NT * DT)
plt.xlabel('time')
plt.yscale('log')
plt.plot(t, maxE, label='maxE', linewidth=2)
plt.savefig(os.path.join(path, 'twostream_maxE_ES%i' % (ES) + '.png'))

# print ('Time elapsed: ', time.clock() - start_time)