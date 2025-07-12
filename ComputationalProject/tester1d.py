import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sort import PIC1D
from sort import PIC_Explicit1D
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
mpl.use('TkAgg')
import win32api
import win32con

# Show a yes/no message box
response = win32api.MessageBox(0, "Neurechnen?", "Simulation", win32con.MB_YESNO | win32con.MB_ICONQUESTION)

border=1
gridpoints=128
NPpCell=20
dt=0.05
t_end=3
# Initialize solvers
solver_test = PIC1D(border, gridpoints, NPpCell, dt)
solver_ref = PIC_Explicit1D(border, gridpoints, NPpCell, dt)

# Create 'rendered' folder if it doesn't exist
if not os.path.exists('rendered'):
    os.makedirs('rendered')

# Define file paths for saved data
data_files = {
    'x_test': 'rendered/x_test_history.npy',
    'v_test': 'rendered/v_test_history.npy',
    'x_ref': 'rendered/x_ref_history.npy',
    'v_ref': 'rendered/v_ref_history.npy',
    't': 'rendered/t_history.npy',
    'energy_total_test': 'rendered/energy_total_history_test.npy',
    'energy_kin_test': 'rendered/energy_kin_history_test.npy',
    'energy_total_ref': 'rendered/energy_total_history_ref.npy',
    'energy_kin_ref': 'rendered/energy_kin_history_ref.npy'
}

# Check if all data files exist
data_exists = all(os.path.exists(file) for file in data_files.values())

# Handle the response
if response == win32con.IDYES:
    data_exists = False
else:

    print("User clicked No")

# Initialize data arrays
total_steps = int(t_end / solver_test.dt)

if data_exists:
    # Load data from files
    x_test_history = np.load(data_files['x_test'])
    v_test_history = np.load(data_files['v_test'])
    x_ref_history = np.load(data_files['x_ref'])
    v_ref_history = np.load(data_files['v_ref'])
    t_history = np.load(data_files['t'])
    energy_total_history_ref = np.load(data_files['energy_total_ref'])
    energy_kin_history_ref = np.load(data_files['energy_kin_ref'])
    energy_total_history_test = np.load(data_files['energy_total_test'])
    energy_kin_history_test = np.load(data_files['energy_kin_test'])
else:
    # Run simulation and store data
    x_test_history = []
    v_test_history = []
    x_ref_history = []
    v_ref_history = []
    t_history = []
    energy_total_history_ref = []
    energy_kin_history_ref = []
    energy_total_history_test = []
    energy_kin_history_test = []


    """
    Calculation
    
    """
    for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
        solver_test.step()
        solver_ref.step()

        # Store data (copy to avoid reference issues)
        x_test_history.append(solver_test.xp.copy())
        v_test_history.append(solver_test.vp[0].copy())
        x_ref_history.append(solver_ref.xp.copy())
        v_ref_history.append(solver_ref.vp[0].copy())
        t_history.append(solver_test.t)
        energy_total_history_test.append(solver_test.CalcEFieldEnergy())
        energy_kin_history_test.append(solver_test.CalcKinEnergery())
        energy_total_history_ref.append(solver_ref.CalcEFieldEnergy())
        energy_kin_history_ref.append(solver_ref.CalcKinEnergery())

    """
    Saving
    
    """


    # Convert lists to NumPy arrays and save
    x_test_history = np.array(x_test_history)
    v_test_history = np.array(v_test_history)
    x_ref_history = np.array(x_ref_history)
    v_ref_history = np.array(v_ref_history)
    t_history = np.array(t_history)
    energy_total_history_test = np.array(energy_total_history_test)
    energy_kin_history_test = np.array(energy_kin_history_test)
    energy_total_history_ref = np.array(energy_total_history_ref)
    energy_kin_history_ref = np.array(energy_kin_history_ref)

    # Save data to files
    np.save(data_files['x_test'], x_test_history)
    np.save(data_files['v_test'], v_test_history)
    np.save(data_files['x_ref'], x_ref_history)
    np.save(data_files['v_ref'], v_ref_history)
    np.save(data_files['t'], t_history)
    np.save(data_files['energy_total_test'], energy_total_history_test)
    np.save(data_files['energy_kin_test'], energy_kin_history_test)
    np.save(data_files['energy_total_ref'], energy_total_history_ref)
    np.save(data_files['energy_kin_ref'], energy_kin_history_ref)



"""
Plott Settings
"""
# Setup plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

# Plot for solver_test
ax1.set_xlim(0, solver_test.Lx)
ax1.set_ylim(-3, 3)
sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
ax1.set_title("Your PIC1D")
ax1.set_xlabel("x")
ax1.set_ylabel("v")
ax1.legend()

# Plot for solver_ref
ax2.set_xlim(0, solver_ref.Lx)
ax2.set_ylim(-3, 3)
sc2_a, = ax2.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
sc2_b, = ax2.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
ax2.set_title("Reference PIC")
ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.legend()

# Energy plot setup
ax3.set_xlim(0, 5)

ax3.set_title("Energy vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Energy")

a=energy_total_history_test
b=energy_kin_history_test

c=energy_total_history_ref
d=energy_kin_history_ref
# Plot fixed energy curves
"""ax3.plot(t_history, (a - a.min()) / (a.max() - a.min()), '-', label='Total Energy')
ax3.plot(t_history, (b - b.min()) / (b.max() - b.min()), '-', label='Kinetic Energy')
ax3.plot(t_history, (c - c.min()) / (c.max() - c.min()), '-', label='Total Energy Ref')
ax3.plot(t_history, (d - d.min()) / (d.max() - d.min()), '-', label='Kinetic Energy Ref')
"""

#ax3.set_yscale('log')
ax3.plot(t_history, a, '-', label='E Field Energy')
ax3.plot(t_history[1:], b[1:], '-', label='Kinetic Energy')
ax3.plot(t_history[1:], c[1:], '-', label='E Field Energy Ref')
ax3.plot(t_history[1:], d[1:], '-', label='Kinetic Energy Ref')
#Set y-axis limits for ax3 (replace ymin, ymax with desired values)
#ax3.set_ylim(-1.0e5, 1.0e5)  # Example: set y-axis limits from -1 to 1


ax3.legend()


"""
print("Field ENergy")
print(energy_total_history_ref)
print("Kin ENergy")
print(energy_kin_history_ref)
"""
def init():
    # Initialize particle plots as empty
    sc1_a.set_data([], [])
    sc1_b.set_data([], [])
    text1.set_text('')

    sc2_a.set_data([], [])
    sc2_b.set_data([], [])
    text2.set_text('')

    return sc1_a, sc1_b, sc2_a, sc2_b, text1, text2

def update(frame):
    Np = solver_test.Np
    Np2 = solver_ref.Np

    x1 = x_test_history[frame]
    v1 = v_test_history[frame]
    x2 = x_ref_history[frame]
    v2 = v_ref_history[frame]
    t = t_history[frame]

    sc1_a.set_data(x1[:Np // 2], v1[:Np // 2])
    sc1_b.set_data(x1[Np // 2:], v1[Np // 2:])
    text1.set_text(f"time = {t:.2f}")

    sc2_a.set_data(x2[:Np2 // 2], v2[:Np2 // 2])
    sc2_b.set_data(x2[Np2 // 2:], v2[Np2 // 2:])
    text2.set_text(f"time = {t:.2f}")

    return sc1_a, sc1_b, sc2_a, sc2_b, text1, text2

anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_steps,
                              interval=1000, blit=True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.title("rho nach der Simulation")
plt.plot(solver_test.rho, label='rho')
plt.xlabel("Grid Index")
plt.ylabel("rho")
plt.legend()

plt.subplot(1,2,2)
plt.title("E_x nach der Simulation")
plt.plot(solver_test.E[0], label='E_x')
plt.xlabel("Grid Index")
plt.ylabel("E_x")
plt.legend()

plt.tight_layout()
plt.show()
