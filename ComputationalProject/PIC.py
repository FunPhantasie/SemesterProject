import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


# In this task, you will implement the PIC loop for a one dimensional problem.
# For simplicity, we will assume the ions have an infinite mass, and only electrons
# move. The function integrate() runs the loop in time.
# At each time step, it calls the function single_step() that calls each step of
# the PIC loop. Implement the PIC loop in single_step().

# deposit_rho() calculates the charge density on the grid from particles.
# The further away from a node of the grid, the less influence a particle has.
# In order to get the relative weight of each particle for the calculation of
# the charge density, one need to interpolate the weight of each particle on the grid.
# Implement this interpolation in interpolation_to_grid().

# After the electric field is updated, the force it exerts on each particle
# must be interpolated from the grid to the particle's position.
# Implement this interpolation in interpolation_to_part() and complete the function calc_force()
# that updates the value of the force applied to each particle.

# Finally, implement step_x() and step_v(), that update the position and velocity of
# all the particles. Run the code and see how the system evolves.


# After running the code once, try changing dt to 3.0. What happens and why?

# Next, turn it back to 0.3 and plot the time evolution of the kinetic energy in the box.
# What do you notice?

# Can you explain your observation?

# What happens if you increase your spatial resolution keeping dx/dt constant?

#-------------------------------------------------------
# Initialization of the simulation
#-------------------------------------------------------

nx = 64                        # grid points
t_b = 0.; t_e = 50; dt = 0.075   # time beginning; time end; time step (normalized to omega_p) 
L = 10.26; dx = L/nx;          # box size (fits fastest growing mode of two-stream); space step (normalized to u_d/omega_p)
x = dx * np.arange(nx)

NPpCell = 20                            # particles per cell
NP = NPpCell*nx                         # total number of particles
qDm = -1                                # charge divided by mass (electrons), ions with infinite mass
amplitude = 0.003                        # perturbation amplitude in the initial conditions

pweight=1/NPpCell                       # particle weight

E = np.zeros(nx)
rho = np.zeros(nx)

# two stream instability setup
# positions (xp) and velocities (vp) of the particles from population 1 and 2
xp1 = 2*L/NP*np.arange(NP//2)
xp2 = 2*L/NP*np.arange(NP//2)
vp1 =  1+amplitude*np.sin(2*np.pi/L*xp1)
vp2 = -1-amplitude*np.sin(2*np.pi/L*xp1)

# list of particle's position and velocities
xp = np.concatenate([xp1, xp2])
vp = np.concatenate([vp1, vp2])
Fp = np.zeros(NP)

#Initial state of the simulation
t = t_b
frames = int(t_e / float(dt)) + 1
Ep    = []
EE    = []
times = []

#-------------------------------------------------------
# Definition of the functions
#-------------------------------------------------------

#function to update the spatial position of a particle
def step_x(dt):
    global L,xp,vp
    xp += 0
    return

#function to update the velocity of a particle
def step_v(dt):
    global vp,Fp
    vp += 0
    return


def interpolation_to_grid(field,x_p):
    for p_iter in range(NP):
        zeta = 0
        i = 0
        ip1 = 0
        diff = 0
        field[i] += 0
        field[ip1] += 0
    return None
    
def interpolation_to_part(part_force,grid_force,x_p):
    for p_iter in range(NP):
        zeta = 0
        i = 0
        ip1 = 0
        diff = 0
        part_force[p_iter] = 0
    return None

#function to interpolate the population density on the grid
def deposit_rho():
    global NP,rho,xp,dx,nx,qDm,pweight
    rho *=0

    # Interpolation of the moment of each particles on the grid
    interpolation_to_grid(0,0)
    
    # ion background to neutralize
    rho +=  rho*0  + 1.0

#interpolation of the electric field E on the particle position
def calc_force():
    global Np,Fp,xp,qDm,rho,E,nx
    interpolation_to_part(0,0,0)

#function to update the electric field (using FFT)
def calc_E():
    global rho,E,L
    rhohat = np.fft.rfft(rho)
    k = 2*np.pi/L*np.arange(rhohat.size)
    with np.errstate(divide='ignore',invalid='ignore'):
        tmp = np.where(k*k > 0, rhohat/(1j*k), 0.)
    E = np.fft.irfft(tmp)
    return

#Definition of a single step in the PIC loop
def single_step(dt):
        # deposit charge density from particles to grid
        # calculate E-field with FFT (on the grid)
        # interpolate E-field from grid to particles


#-------------------------------------------------------
# Plot and animation of the simulation and its evolution
#-------------------------------------------------------

# plotting stuff
fig = plt.figure()
ax = plt.axes(xlim=(0,L), ylim=(-3, 3))
xv_scatter, = ax.plot(xp,vp,'o',markersize=0.5)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
title = ax.set_title('1D PIC')
ax.set_xlabel('$x \omega_p/u_d$')
ax.set_ylabel('$v/u_d$')

#animation
def init():
        xv_scatter.set_data([], [])
        time_text.set_text('')
        return (xv_scatter, time_text)

#------------------------------
#implementation of the PIC loop
#------------------------------
def integrate(i):
    global t,dt,dx, pweight
    single_step(dt)
    t = t + dt
    # print("t = ",t)
    #Percentage of kinetic energy in the box compared to t=0
    Ep.append(0.5*pweight*np.sum(vp**2)*dx)
    EE.append(0.5*np.sum(E**2)*dx)
    times.append(t)
    xv_scatter.set_data(xp, vp)
    time_text.set_text('time = %.2f' % t)
    return (xv_scatter, time_text)
#------------------------------

anim = animation.FuncAnimation(fig, integrate, init_func=init, frames=frames,
                               interval=100, blit=False,repeat=False)

plt.show()


EE = (np.roll(EE,-1) + EE)/2
Etot=np.array(EE)+np.array(Ep)
#Plot of the time evolution of the kinetic energy

#energy growth
plt.yscale('log')
plt.xlabel('$t \omega_p$')
plt.ylabel('$|\Delta En|/En(0)$')
plt.ylim([1e-8,1e1])
times=np.array(times)
plt.plot(times,abs(Ep-Ep[0])/Etot[0],label="kinetic energy")
plt.plot(times,EE/Etot[0],label="electric energy")
plt.plot(times,abs(Etot - Ep[0])/Etot[0],label="total energy")
plt.plot(times,1e-6*np.exp(times*2/2.8284),label="theoretical growth") #theoretical growth rate
plt.legend(loc="lower right")
plt.show()

#energy evolution
plt.xlabel('$t \omega_p$')
plt.ylabel('$En/En(0)$')
plt.plot(times,Ep/Etot[0], label="kinetic energy")
plt.plot(times,EE/Etot[0], label="electric energy")
plt.plot(times,Etot/Etot[0], label="total energy")
plt.legend(loc="center right")
plt.show()
