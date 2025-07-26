import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
# In this task, we will apply the Boris pusher that we just developed
# into a PIC simulation.
# This code is a slightly modified version of last week's code.
# Look at the code and identify the main changes.
"Velocity are now defined in three directions and the"
"magnetic field can be implemented as well."
"The previous code was 1D-1V, this one is 1D-3V."

# Implement the calc_force() function taking into account this
# difference.
# Next, run the code. How different are the results in comparison to
# last week's result?
"No difference"

# Now, implement the Boris pusher and use it to push particles.
# How does it change the result.
"No changes"
# How do you explain it?
"The Boris pusher is an improved pusher for an electromagnetic code,"
"but for now, there is no magnetic field, so we don't"
"see the impact of the Boris pusher, nor the problems of"
"the Euler pusher."

# Now, restore the Euler pusher and add an initial magnetic
# field along z (the third dimension).
# Run the code. What do you observe?
"The total energy is no longer conserved."
# Try the same with the Boris pusher. Is it better?
"Here, we go back to the previous case, the simulation"
"is stable."
"The reason is that the Euler is not stable, while Boris"
"was made to be stable in an electromagnetic plasma"

"""
Note: This code was not electromagnetic.
It is an electrostatic code, made to get a magnetic field,
but by construction, this only works because the magnetic field
is perpendicular to the simulation direction, and therefore remains constant in time.
"""

#-------------------------------------------------------
# Initialization of the simulation
#-------------------------------------------------------

nx = 16                        # grid points
t_b = 0.; t_e = 50; dt = 0.3   # time beginning; time end; time step (normalized to omega_p) 
L = 10.26; dx = L/nx;          # box size (fits fastest growing mode of two-stream); space step (normalized to u_d/omega_p)
x = dx * np.arange(nx)

NPpCell = 20                            # particles per cell
NP = NPpCell*nx                         # total number of particles
qDm = -1                                # charge divided by mass (electrons), ions with infinite mass
amplitude = 0.001                        # perturbation amplitude in the initial conditions

pweight=1/NPpCell                       # particle weight

E = np.zeros([3,nx])
B = np.zeros([3,nx])
rho = np.zeros(nx)

vp = np.zeros([3,NP])
Fp = np.zeros([3,NP])
Ep = np.zeros([3,NP])
Bp = np.zeros([3,NP])

# two stream instability setup
# positions (xp) and velocities (vp) of the particles from population 1 and 2
xp1 = 2*L/NP*np.arange(NP//2)
xp2 = 2*L/NP*np.arange(NP//2)
vp1 =  1+amplitude*np.sin(2*np.pi/L*xp1)
vp2 = -1-amplitude*np.sin(2*np.pi/L*xp1)

# list of particle's position and velocities
xp = np.concatenate([xp1, xp2])
vp_x = np.concatenate([vp1, vp2])

vp[0,:] = vp_x
B[2,...] = 0.0

#Initial state of the simulation
t = t_b
frames = int(t_e / float(dt)) + 1
Ekin    = []
EE    = []
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
def step_x(dt):
    global L,xp,vp
    xp += dt*vp[0,...]
    xp = np.mod(xp,L) # periodic boundary conditions
    return

#function to update the velocity of a particle
def step_v(dt):
    global vp,Fp
    vp += dt*Fp
    return

def Boris(dt):
    global vp, Ep, Bp, qDm

    a = 0.5 * dt * qDm

    t_boris = a * Bp
    s_boris = 2*t_boris/(1+dot(t_boris,t_boris))

    v_min = vp + a * Ep

    v_quote = v_min + cross(v_min, t_boris)

    v_plus = v_min + cross(v_quote, s_boris)

    vp = v_plus + a * Ep
    return


def interpolation_to_grid(field,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        ip1 = np.mod(i+1,nx)
        diff = zeta - i
        field[i] += 1-diff
        field[ip1] += diff
    return None
    
def interpolation_to_part(part_force,grid_force,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        ip1 = np.mod(i+1,nx)
        diff = zeta - i
        part_force[:,p_iter] = (1-diff)*grid_force[:,i] + diff*grid_force[:,ip1]
    return None

'''
#Third-order interpolation (interpolation over the boundary node of the cell and its neighbor)
def interpolation_rho_to_grid(field,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        im1 = np.mod(i-1,nx)
        ip1 = np.mod(i+1,nx)
        ip2 = np.mod(i+2,nx)
        diff = zeta - i
        rho_[im1] += -           diff    *(diff-1.)*(diff-2.)/6.
        field[i]   +=  (diff+1.)*          (diff-1.)*(diff-2.)/2.
        field[ip1] += -(diff+1.)* diff    *          (diff-2.)/2.
        field[ip2] +=  (diff+1.)* diff    *(diff-1.)          /6.
    return None
#Third-order interpolation (including also the boundary node of neighbor cells)
def interpolation_to_part(part_force,grid_force,x_p):
    for p_iter in range(NP):
        zeta = x_p[p_iter]/dx
        i = int(zeta)
        im1 = np.mod(i-1,nx)
        ip1 = np.mod(i+1,nx)
        ip2 = np.mod(i+2,nx)
        diff = zeta - i
        part_force[:,p_iter] = grid_force[:,im1]* -           diff    *(diff-1.)*(diff-2.)/6. \
                             + grid_force[:,i]  *  (diff+1.)*          (diff-1.)*(diff-2.)/2. \
                             + grid_force[:,ip1]* -(diff+1.)* diff    *          (diff-2.)/2. \
                             + grid_force[:,ip2]*  (diff+1.)* diff    *(diff-1.)          /6.
    return None
'''

#function to interpolate the population density on the grid
def deposit_rho():
    global NP,rho,xp,dx,nx,qDm,pweight
    rho *=0

    # Interpolation of the moment of each particles on the grid
    interpolation_to_grid(rho,xp)
    
    rho *= np.sign(qDm)*pweight
    # ion background to neutralize
    rho +=  rho*0  + 1.0

#interpolation of the electric field E on the particle position
def calc_force():
    global NP,Fp,Ep,Bp,xp,qDm,rho,E,B,nx

    interpolation_to_part(Ep,E,xp)
    interpolation_to_part(Bp,B,xp)

    Fp = qDm * (Ep + cross(vp, Bp) )

#function to update the electric field (using FFT)
def calc_E():
    global rho,E,L
    rhohat = np.fft.rfft(rho)
    k = 2*np.pi/L*np.arange(rhohat.size)
    with np.errstate(divide='ignore',invalid='ignore'):
        tmp = np.where(k*k > 0, rhohat/(1j*k), 0.)
    E[0,:] = np.fft.irfft(tmp)
    return

#Definition of a single step in the PIC loop
def single_step(dt):
    step_x(dt/2)
    deposit_rho()    # deposit charge density from particles to grid
    calc_E()        # calculate E-field with FFT (on the grid)
    calc_force()    # interpolate E-field from grid to particles
    #step_v(dt)
    Boris(dt)
    step_x(dt/2)


#-------------------------------------------------------
# Plot and animation of the simulation and its evolution
#-------------------------------------------------------

# plotting stuff
fig = plt.figure()
ax = plt.axes(xlim=(0,L), ylim=(-3, 3))
xv_scatter, = ax.plot(xp,vp[0,:],'o',markersize=0.5)
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
    Ekin.append(0.5*pweight*np.sum(vp**2)*dx)
    EE.append(0.5*np.sum(E**2)*dx)
    times.append(t)
    xv_scatter.set_data(xp, vp[0,:])
    time_text.set_text('time = %.2f' % t)
    return (xv_scatter, time_text)
#------------------------------

anim = animation.FuncAnimation(fig, integrate, init_func=init, frames=frames,
                               interval=100, blit=False,repeat=False)

plt.show()


EE = (np.roll(EE,-1) + EE)/2
Etot=np.array(EE)+np.array(Ekin)
#Plot of the time evolution of the kinetic energy

#energy growth
plt.yscale('log')
plt.xlabel('$t \omega_p$')
plt.ylabel('$|\Delta En|/En(0)$')
plt.ylim([1e-8,1e1])
times=np.array(times)
plt.plot(times,abs(Ekin-Ekin[0])/Etot[0],label="kinetic energy")
plt.plot(times,EE/Etot[0],label="electric energy")
plt.plot(times,abs(Etot - Ekin[0])/Etot[0],label="total energy")
plt.plot(times,1e-6*np.exp(times*2/2.8284),label="theoretical growth") #theoretical growth rate
plt.legend(loc="lower right")
plt.show()

#energy evolution
plt.xlabel('$t \omega_p$')
plt.ylabel('$En/En(0)$')
plt.plot(times,Ekin/Etot[0], label="kinetic energy")
plt.plot(times,EE/Etot[0], label="electric energy")
plt.plot(times,Etot/Etot[0], label="total energy")
plt.legend(loc="center right")
plt.show()
