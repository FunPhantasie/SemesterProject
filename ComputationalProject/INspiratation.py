import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

# In this task, we will apply the Boris pusher that we just developed
# into a PIC simulation.
# This code is a slightly modified version of last week's code.
# Look at the code and identify the main changes.

# Implement the force() function taking into account this
# difference.
# Next, run the code. How different are the results in comparison to
# last week's result?

# Now, implement the Boris pusher and use it to push particles.
# How does it change the result.

# How do you explain it?

# Now, restore the Euler pusher and add an initial magnetic
# field along z (the third dimension).
# Run the code. What do you observe?

# Try the same with the Boris pusher. Is it better?


#-------------------------------------------------------
# Initialisation of the simulation
#-------------------------------------------------------

mx = 128   *2                     # grid points
t_b = 0.; t_e = 50; dt = 0.1    # time beginning; time end; time step
Lx = 10; dx = Lx/mx;               # box size; space step
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
Fp = np.zeros([3,NP])
Ep = np.zeros([3,NP])
Bp = np.zeros([3,NP])


# two-stream instability setup
# positions (xp) and velocities (vp) of the particles from population 1 and 2
xp1 = 2*Lx/NP*np.arange(NP//2)
xp2 = 2*Lx/NP*np.arange(NP//2)
vp1 =  1+amplitude*np.sin(2*np.pi/Lx*xp1)
vp2 = -1-amplitude*np.sin(2*np.pi/Lx*xp1)

# list of particle's positions and velocities
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
    xp  = np.mod(xp,Lx) # periodic boundary conditions
    return

#function to update the velocity of a particle
def step_v(dt_):
    global vp,Fp
    vp += dt_*Fp
    return
    
def Boris(dt):
    global vp, Ep, Bp, qDm

    # Step 1: Half-acceleration by E
    v_minus = vp + 0.5 * dt * qDm * Ep

    # Step 2: Rotation by B
    t_rot = 0.5 * dt * qDm * Bp

    s = 2 * t_rot / (1 + dot(t_rot, t_rot))

    v_prime = v_minus + cross(v_minus, t_rot)
    v_plus = v_minus + cross(v_prime, s)

    # Step 3: Half-acceleration by E again
    vp = v_plus + 0.5 * dt * qDm * Ep
    return


#First-order interpolation (interpolation over the boundary node of each cell)


#First-order interpolation (including only the boundary node of each cell)
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
def interpolation_rho_to_grid(rho_,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        ip1 = np.mod(i+1,mx)
        diff = zeta - i
        rho_[i]   += 1-diff
        rho_[ip1] += diff
    return None


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
    global Fp,Ep,Bp,NP,rho,xp,dx,qDm,mx

    interpolation_to_part(Ep,E,xp)
    interpolation_to_part(Bp,B,xp)
    
    Fp = qDm * (Ep + cross(vp, Bp) )

#function to update the electric field (using FFT)

def calc_E():
    global rho,E,Lx
    rhohat = np.fft.rfft(rho)
    kx = 2*np.pi/Lx*np.arange(rhohat.size)
    with np.errstate(divide='ignore',invalid='ignore'):
        tmp = np.where(kx*kx > 0, rhohat/(1j*kx), 0.)
    E[0,:] = np.fft.irfft(tmp)
    return



#Definition of a single step in the PIC loop
def single_step(dt_):
    #step_x(dt_)

    weight_rho()    # from particles to grid
    calc_E()        # calculate E-field with FFT (on the grid)
    force()         # from grid to particles
    #step_v(dt_) boris updated V
    Boris(dt_)
    step_x(dt_)


#Looking at the fft


#-------------------------------------------------------
# Plot and animation of the simulation and its evolution
#-------------------------------------------------------

# plotting stuff
fig = plt.figure()
ax = plt.axes(xlim=(0,Lx), ylim=(-3,3))
xv_scatter, = ax.plot(xp, vp[0,:],'o',markersize=0.5)
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
        xv_scatter.set_data([], [])
        time_text.set_text('')
        #step_x(dt/2)
        return (xv_scatter, time_text)

def integrate(i):
    global t,dt,dx
    for istep in range(N_steps):
        single_step(dt)
        t = t + dt
        # print("t = ",t)
        #Percentage of kinetic energy in the box compared to t=0
        Ekin.append(0.5*np.sum(vp**2)/Ekin0*100)
        times.append(t)
    xv_scatter.set_data(xp, vp[0,:])
    time_text.set_text('time = %.2f' % t)
    return (xv_scatter, time_text)

    
#------------------------------

anim = animation.FuncAnimation(fig, integrate, init_func=init, frames=frames,
                               interval=100, blit=False,repeat=False)

plt.show()


#Plot of the time evolution of the kinetic energy

plt.plot(times,Ekin)
plt.show()
