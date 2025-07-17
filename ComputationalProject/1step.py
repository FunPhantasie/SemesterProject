import matplotlib.pyplot as plt
from sort import PIC1D
from explicit_particle_sim import Explicit_PIC_Solver
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
# Simulation parameters
border = 1
gridpoints = 128
NPpCell = 20
dt = 0.05
nsteps = 2



# Initialize solvers
solver_test = PIC1D(border, gridpoints, NPpCell, dt)
solver_test.setEn()
solver_ref = Explicit_PIC_Solver(border, gridpoints, NPpCell, dt)



# Run simulation for n steps
for step in range(nsteps):
    solver_test.step()
    solver_ref.step()  # for comparison

# After simulation, get final states
x1 = solver_test.xp
v1_x = solver_test.vp[0]
v1_y = solver_test.vp[1]
v1_z = solver_test.vp[2]
rho_test = solver_test.rho
E_test_x = solver_test.E_prev[0]
E_test_y = solver_test.E_prev[1]
E_test_z = solver_test.E_prev[2]
B_x=solver_test.B[0]

rho_ref = solver_ref.rho
E_ref = solver_ref.E[0]

# Compute energies for the whole run to plot energy vs time
# (or you could skip this entirely to keep it minimal)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot particles (x,v)
Np = solver_test.Np
ax1.plot(x1[:Np//2], v1_x[:Np//2], 'o', markersize=0.5, label='Species A')
ax1.plot(x1[Np//2:], v1_x[Np//2:], 'o', markersize=0.5, label='Species B')
ax1.set_xlim(0, solver_test.Lx)
ax1.set_ylim(-3, 3)
ax1.set_title(f"Particles after {nsteps} steps (t = {nsteps*dt})")
ax1.set_xlabel("x")
ax1.set_ylabel("v")
ax1.legend()

# Plot rho and E_x on grid
grid_indices = np.arange(gridpoints)
ax2.plot(grid_indices, rho_test, linestyle='dashed', label='rho (Implicit)', color='g')
ax2.plot(grid_indices, B_x, linestyle='dashed', label='B_x (Implicit)', color='r')
ax2.plot(grid_indices, E_test_x, linestyle='solid', label='E_x (Implicit)', color='black')
ax2.plot(grid_indices, E_test_y, linestyle='solid', label='E_y (Implicit)', color='blue')
ax2.plot(grid_indices, E_test_z, linestyle='solid', label='E_z (Implicit)', color='magenta')
#ax2.plot(grid_indices, rho_ref, linestyle='solid', label='rho (Explicit)', color='black')
#ax2.plot(grid_indices, E_ref, linestyle='dotted', label='E_x (Explicit)', color='magenta')

ymin, ymax = ax2.get_ylim()
abs_max = max(abs(ymin), abs(ymax))

if abs_max < 0.1:
    ax1.set_ylim(-0.01, 0.01)
ax2.set_title("Charge Density and Electric Field")
ax2.set_xlabel("Grid index")
ax2.set_ylabel("Value")
ax2.legend()

plt.tight_layout()
plt.show()
