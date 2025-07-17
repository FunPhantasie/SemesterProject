import matplotlib.pyplot as plt
from sort import PIC1D
from explicit_particle_sim import Explicit_PIC_Solver
from semi_implicit_moment_pic import SemiImplicitPIC1D
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
import matplotlib.animation as animation
mpl.use('TkAgg')
import win32api
import win32con

# Show a yes/no message box
response = win32api.MessageBox(0, "Neurechnen?", "Simulation", win32con.MB_YESNO | win32con.MB_ICONQUESTION)

t_end=5



solver_test=SemiImplicitPIC1D()
# Create 'rendered' folder if it doesn't exist
if not os.path.exists('rendered'):
    os.makedirs('rendered')

# Define file paths for saved data (only for solver_test)
data_files = {
    'x_test': 'rendered/x_test_history.npy',
    'v_test': 'rendered/v_test_history.npy',
    't': 'rendered/t_history.npy',
}

# Check if all data files exist
data_exists = all(os.path.exists(file) for file in data_files.values())

# Handle the response
if response == win32con.IDYES:
    data_exists = False
else:
    print("User clicked No")

total_steps = int(t_end / solver_test.dt)

if data_exists:
    # Load data from files
    x_test_history = np.load(data_files['x_test'])
    v_test_history = np.load(data_files['v_test'])
    t_history = np.load(data_files['t'])
else:
    # Run simulation and store data
    x_test_history = []
    v_test_history = []
    t_history = []

    """
    Calculation
    """
    for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
        solver_test.step()
         # Still run solver_ref for consistency, but don't store its data

        # Store data for solver_test (copy to avoid reference issues)
        x_test_history.append(solver_test.xp.copy())
        v_test_history.append(solver_test.vp.copy())

        t_history.append(solver_test.t)


    """
    Saving
    """
    # Convert lists to NumPy arrays and save
    x_test_history = np.array(x_test_history)
    v_test_history = np.array(v_test_history)

    t_history = np.array(t_history)


    # Save data to files
    np.save(data_files['x_test'], x_test_history)
    np.save(data_files['v_test'], v_test_history)

    np.save(data_files['t'], t_history)


"""
Plot Settings
"""
# Setup plots
fig, (ax1) = plt.subplots(1, figsize=(6, 5), sharey=False)

# Plot for solver_test particles
ax1.set_xlim(0, solver_test.Lx)
ax1.set_ylim(-3, 3)
sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
ax1.set_title("PIC1D Particles")
ax1.set_xlabel("x")
ax1.set_ylabel("v")
ax1.legend()









# Initialize current frame index
current_frame = [0]  # Use list to allow modification in callback


def update_plot(frame):
    Np = solver_test.Np
    x1 = x_test_history[frame]
    v1 = v_test_history[frame][0]
    t = t_history[frame]

    # Update particle plot
    sc1_a.set_data(x1[:Np // 2], v1[:Np // 2])
    sc1_b.set_data(x1[Np // 2:], v1[Np // 2:])
    text1.set_text(f"time = {t:.2f}")


    text1.set_text(f"time = {t:.2f}")

    return sc1_a, sc1_b,  text1





def init():
    # Initialize particle plots as empty
    sc1_a.set_data([], [])
    sc1_b.set_data([], [])
    text1.set_text('')



    return sc1_a, sc1_b,  text1,


# Initialize plot with first frame
anim = animation.FuncAnimation(fig, update_plot, init_func=init, frames=total_steps,
                              interval=100, blit=True)

plt.tight_layout()
plt.show()