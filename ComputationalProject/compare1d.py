import matplotlib.pyplot as plt
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

border = 1
gridpoints = 128
NPpCell = 20
dt = 0.05
t_end = 3

# Initialize solvers
solver_test = PIC1D(border, gridpoints, NPpCell, dt)
solver_ref = PIC_Explicit1D(border, gridpoints, NPpCell, dt)

# Create 'rendered' folder if it doesn't exist
if not os.path.exists('rendered'):
    os.makedirs('rendered')

# Define file paths for saved data (only for solver_test)
data_files = {
    'x_test': 'rendered/x_test_history.npy',
    'v_test': 'rendered/v_test_history.npy',
    't': 'rendered/t_history.npy',
    'energy_total_test': 'rendered/energy_total_history_test.npy',
    'energy_kin_test': 'rendered/energy_kin_history_test.npy',
    'rho_test': 'rendered/rho_test_history.npy',
    'E_test': 'rendered/E_test_history.npy',
    'rho_ref': 'rendered/rho_ref_history.npy',
    'E_ref': 'rendered/E_ref_history.npy'
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
    t_history = np.load(data_files['t'])
    energy_total_history_test = np.load(data_files['energy_total_test'])
    energy_kin_history_test = np.load(data_files['energy_kin_test'])
    rho_test_history = np.load(data_files['rho_test'])
    E_test_history = np.load(data_files['E_test'])
    rho_ref_history = np.load(data_files['rho_ref'])
    E_ref_history = np.load(data_files['E_ref'])
else:
    # Run simulation and store data
    x_test_history = []
    v_test_history = []
    t_history = []
    energy_total_history_test = []
    energy_kin_history_test = []
    rho_test_history = []
    E_test_history = []
    rho_ref_history = []
    E_ref_history = []
    af = lambda a: a
    solver_test.rho = solver_test.deposit_charge(solver_test.xp, af)

    """
    Calculation
    """
    for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
        solver_test.step()
        solver_ref.step()  # Still run solver_ref for consistency, but don't store its data

        # Store data for solver_test (copy to avoid reference issues)
        x_test_history.append(solver_test.xp.copy())
        v_test_history.append(solver_test.vp.copy())
        t_history.append(solver_test.t)
        energy_total_history_test.append(solver_test.CalcEFieldEnergy())
        energy_kin_history_test.append(solver_test.CalcKinEnergery())
        rho_test_history.append(solver_test.rho.copy())
        E_test_history.append(solver_test.E.copy())
        rho_ref_history.append(solver_ref.rho.copy())
        E_ref_history.append(solver_ref.E.copy())

    """
    Saving
    """
    # Convert lists to NumPy arrays and save
    x_test_history = np.array(x_test_history)
    v_test_history = np.array(v_test_history)
    t_history = np.array(t_history)
    energy_total_history_test = np.array(energy_total_history_test)
    energy_kin_history_test = np.array(energy_kin_history_test)
    rho_test_history = np.array(rho_test_history)
    E_test_history = np.array(E_test_history)
    rho_ref_history = np.array(rho_ref_history)
    E_ref_history = np.array(E_ref_history)

    # Save data to files
    np.save(data_files['x_test'], x_test_history)
    np.save(data_files['v_test'], v_test_history)
    np.save(data_files['t'], t_history)
    np.save(data_files['energy_total_test'], energy_total_history_test)
    np.save(data_files['energy_kin_test'], energy_kin_history_test)
    np.save(data_files['rho_test'], rho_test_history)
    np.save(data_files['E_test'], E_test_history)
    np.save(data_files['rho_ref'], rho_ref_history)
    np.save(data_files['E_ref'], E_ref_history)

"""
Plot Settings
"""
# Setup plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

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

# Plot for rho and E_x (solver_test only)
ax2.set_xlim(0, gridpoints)
ax2.set_ylim(-0.5, 0.5)  # Adjust based on expected rho and E ranges
rho_line_test, = ax2.plot([], [], '-', color='g', label='rho')
E_line_test, = ax2.plot([], [], '-', color='r', label='E_x')
rho_line_ref, = ax2.plot([], [], '--', color='cyan', label='rho (Ref)')

E_line_ref, = ax2.plot([], [], '--', color='m', label='E_x (Ref)')
text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
ax2.set_title("Charge Density and Electric Field")
ax2.set_xlabel("Grid Index")
ax2.set_ylabel("Value")
ax2.legend()

# Energy plot setup
ax3.set_xlim(0, t_end)
ax3.set_title("Energy vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Energy")

# Plot fixed energy curves (solver_test only)
ax3.plot(t_history, energy_total_history_test, '-', label='E Field Energy')
ax3.plot(t_history[1:], energy_kin_history_test[1:], '-', label='Kinetic Energy')
ax3.legend()

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

    # Update rho and E plot
    grid_indices = np.arange(gridpoints)
    rho_line_test.set_data(grid_indices, rho_test_history[frame])
    E_line_test.set_data(grid_indices, E_test_history[frame][0])
    rho_line_ref.set_data(grid_indices, rho_ref_history[frame])
    print(rho_test_history[frame])
    #print(rho_ref_history[frame])

    E_line_ref.set_data(grid_indices, E_ref_history[frame][0])
    text2.set_text(f"time = {t:.2f}")

    fig.canvas.draw()


def on_click(event):
    # Increment frame on click, wrap around if at end
    if event.button == 1:  # Left click
        current_frame[0] = (current_frame[0] + 1) % total_steps
        update_plot(current_frame[0])
    elif event.button == 3:  # Right click
        current_frame[0] = (current_frame[0] - 1) % total_steps
        update_plot(current_frame[0])


# Initialize plot with first frame
update_plot(current_frame[0])

# Connect click event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()