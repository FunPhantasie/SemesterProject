import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np
import os as os
import matplotlib as mpl
mpl.use('TkAgg')

# After implementing boundary conditions for a purely electromagnetic code,
# we will now implement boundary conditions for particles.
# The function boundary() selects the boundary condition used for the particles.
# The periodic boundary condition is implemented by default. Check how it works,
# and then implement reflective boundary conditions, that will reflect the
# particles reaching the boundary.

# Once done, implement open boundary conditions, that will remove the particles
# reaching the boundary from the box. What is the major problem with this last
# boundary condition?
"By removing particles, you remove energy. So you break the energy conservation"
"condition."

# In this exercise, you've implemented boundary conditions for the particles, but
# didn't change the ones for the field. Is it a problem?
"Yes, it can be a problem in terms of consistency. Not necessarily numerically,"
"as it can just be seen as forcing the electromagnetic fields at the boundaries."


#-------------------------------------------------------
# Initialization of the simulation
#-------------------------------------------------------

mx = 128                        # grid points
t_b = 0.; t_e = 5.; dt = 0.1    # time beginning; time end; time step
Lx = 1; dx = Lx/mx;               # box size; space step
x = dx * np.arange(mx)

NPpCell = 20                            # particles per cell
NP = NPpCell*mx                         # total number of particles
qDm = -1                                # charge divided by mass (electrons), ions with infinite mass
amplitude = 0.01                        # perturbation amplitude in the initial conditions
omega_p = 1                             # plasma frequency
epsilon_0 = 1                           # convenient normalization
charge = omega_p**2/qDm*epsilon_0*Lx/NP  # particle charge

E = np.zeros([3,mx])
B = np.zeros([3,mx])
rho = np.zeros(mx)

vp = np.zeros([3,NP])
Ep = np.zeros([3,NP])
Bp = np.zeros([3,NP])


# two stream instability setup
# positions (xp) and velocities (vp) of the particles from population 1 and 2
xp1 = 2*Lx/NP*np.arange(NP//2)
xp2 = 2*Lx/NP*np.arange(NP//2)
vp1 =  1+amplitude*np.sin(2*np.pi/Lx*xp1)
vp2 = -1-amplitude*np.sin(2*np.pi/Lx*xp1)

# list of particle's position and velocities
xp = np.concatenate([xp1, xp2])
vp_x = np.concatenate([vp1, vp2])

vp[0,:] = vp_x




B[2,...] = 1.

#Initial state of the simulation
t = t_b
N_steps = 1
frames = int(t_e / float(N_steps * dt)) + 1
Ekin0   = np.sum(vp**2)*0.5
Ekin    = []
times = []

#-------------------------------------------------------
# Definition of the functions
#-------------------------------------------------------

def dot(A,B):
    U = np.zeros_like(A)
    U[0,...] = A[0,...]*B[0,...]
    U[1,...] = A[1,...]*B[1,...]
    U[2,...] = A[2,...]*B[2,...]
    return U

def cross(A,B):
    U = np.zeros_like(A)
    U[0,...] = A[1,...]*B[2,...] - A[2,...]*B[1,...]
    U[1,...] = A[2,...]*B[0,...] - A[0,...]*B[2,...]
    U[2,...] = A[0,...]*B[1,...] - A[1,...]*B[0,...]
    return U

#function to update the spatial position of a particle
def step_x(dt_):
    global L,xp,vp
    xp += dt_*vp[0,...]
    return

    
def Boris(dt_):
    global vp, Ep, Bp, qDm
  
    a = 0.5 * dt_ * qDm
  
    t_boris = a * Bp
    s_boris = 2*t_boris/(1+dot(t_boris,t_boris))
  
    v_min = vp + a * Ep
  
    v_quote = v_min + cross(v_min, t_boris)
  
    v_plus = v_min + cross(v_quote, s_boris)
  
    vp = v_plus + a * Ep
    return


#First order interpolation (interpolation over the boundary node of each cell)
def interpolation_rho_to_grid(rho_,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        ip1 = np.mod(i+1,mx)
        diff = zeta - i
        rho_[i]   += 1-diff
        rho_[ip1] += diff
    return None


#First order interpolation (including only the boundary node of each cell)
def interpolation_to_part(part_force,grid_force,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        ip1 = np.mod(i+1,mx)
        diff = zeta - i
        part_force[:,p_iter] = (1-diff)*grid_force[:,i]  \
                             +    diff*grid_force[:,ip1]
    return None


#function to interpolate the population density on the grid
def weight_rho():
    global NP,rho,xp,charge,dx,mx
    rho *=0

    # Interpolation of the moment of each particles on the grid
    interpolation_rho_to_grid(rho,xp)
    
    # ion background to neutralize
    rho -= NP/mx
    rho *= 2*NPpCell*charge/dx
    # print('total charge = ',np.sum(rho))


#interpolation of the force on the particle position
def force():
    global Ep,Bp,NP,rho,xp,dx,qDm,mx

    interpolation_to_part(Ep,E,xp)
    interpolation_to_part(Bp,B,xp)

#function to update the electric field (using FFT)

def calc_E():
    global rho,E,Lx
    rhohat = np.fft.rfft(rho)
    kx = 2*np.pi/Lx*np.arange(rhohat.size)
    with np.errstate(divide='ignore',invalid='ignore'):
        tmp = np.where(kx*kx > 0, rhohat/(1j*kx), 0.)
    E[0,:] = np.fft.irfft(tmp)
    return
    
#The particle boundary conditions
    
def boundary():
    global xp,vp,Lx
    
    periodic()
    #reflective()
    #open()
    
    return


def periodic():
    global xp,vp,Lx
    
    xp  = np.mod(xp,Lx)
    return
    
    
def reflective():
    global xp,vp,Lx,dx
    
    ip     = np.where(xp>Lx)
    diff   = xp[ip]-Lx
    xp[ip] = Lx-diff
    vp[0,ip] =- vp[0][ip]
    
    im     = np.where(xp<0.)
    diff   = xp[im]
    xp[im] = -diff
    vp[0][im] =-vp[0][im]
    
    return
    
def open():
    global xp,vp,Lx,dx,NP,Ep,Bp
    
    ip     = np.where(xp>Lx)[0]
    im     = np.where(xp<0.)[0]
        
    xp = np.delete(xp,ip)
    vp = np.delete(vp,ip,1)
    Ep = np.delete(Ep,ip,1)
    Bp = np.delete(Bp,ip,1)
    
    NP-= len(ip)
    
    xp = np.delete(xp,im)
    vp = np.delete(vp,im,1)
    Ep = np.delete(Ep,im,1)
    Bp = np.delete(Bp,im,1)
        
    NP-= len(im)
    return

#Definition of a single step in the PIC loop
def single_step(dt_):
    
    
    weight_rho()    # from particles to grid
    calc_E()        # calculate E-field with FFT (on the grid)
    force()         # from grid to particles
    Boris(dt_)
    step_x(dt_)
    boundary()      # implement the boundary conditions


#Looking at the fft


#-------------------------------------------------------
# Plot and animation of the simulation and its evolution
#-------------------------------------------------------

# plotting stuff
fig = plt.figure()
ax = plt.axes(xlim=(0,Lx), ylim=(-3,3))
xv_scatter1, = ax.plot(xp[:NP//2], vp[0,:NP//2],'o',markersize=0.5,color='b')
xv_scatter2, = ax.plot(xp[NP//2:], vp[0,NP//2:],'o',markersize=0.5,color='r')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
title = ax.set_title('1D PIC')

ax.set_xlabel('$x$')
ax.set_ylabel('$v$')


#------------------------------
#implementation of the PIC loop
#------------------------------

#Integration for the x VS vx plot

#animation
def init():
        xv_scatter1.set_data([], [])
        xv_scatter2.set_data([], [])
        time_text.set_text('')
        step_x(dt/2)
        boundary()
        return (xv_scatter1,xv_scatter2, time_text)

def integrate(i):
    global t,dt,dx
    for istep in range(N_steps):
        single_step(dt)
        t = t + dt
        # print("t = ",t)
        #Percentage of kinetic energy in the box compared to t=0
        Ekin.append(0.5*np.sum(vp**2)/Ekin0*100)
        times.append(t)
    xv_scatter1.set_data(xp[:NP//2], vp[0,:NP//2])
    xv_scatter2.set_data(xp[NP//2:], vp[0,NP//2:])
    time_text.set_text('time = %.2f' % t)
    return (xv_scatter1,xv_scatter2, time_text)

    
#------------------------------

anim = animation.FuncAnimation(fig, integrate, init_func=init, frames=frames,
                               interval=100, blit=False,repeat=False)

plt.show()
del open
#Plot of the time evolution of the kinetic energy
# Implement it


plt.plot(times,Ekin)
plt.show()

