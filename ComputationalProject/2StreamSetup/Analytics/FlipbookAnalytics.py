import matplotlib.pyplot as plt

from .RenderSavings import CallItRenderer


import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os

mpl.use('TkAgg')
import win32api
import win32con



def run(solver_test,solver_ref,total_steps,t_end):
    # Catching the returned values with matching variable names
    x_test_history, v_test_history, x_ref_history, v_ref_history, t_history,\
     energy_total_history_test, energy_kin_history_test, rho_test_history,\
     E_test_history, rho_ref_history, E_ref_history, energy_total_history_ref, energy_kin_history_ref = CallItRenderer(solver_test,solver_ref,total_steps,t_end)

    """
    Plot Settings
    """
    # Setup plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

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


    # Plot for solver_test particles
    ax2.set_xlim(0, solver_ref.Lx)
    ax2.set_ylim(-3, 3)
    sc2_a, = ax2.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
    sc2_b, = ax2.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
    text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
    ax2.set_title("PIC1D Explicit")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.legend()


    # Plot for rho and E_x (solver_test only)
    ax3.set_xlim(0, solver_test.Nx)
    ax3.set_ylim(-5, 5)  # Adjust based on expected rho and E ranges

    rho_line_test, = ax3.plot([], [], linestyle='dashed', color='g', label='rho (Implicit)')
    E_line_test, = ax3.plot([], [], linestyle='dashed', color='r', label='E_x (Implicit)')
    rho_line_ref, = ax3.plot([], [], linestyle='solid', color='black', label='rho (Explicit)')
    E_line_ref, = ax3.plot([], [], linestyle='dotted', color='m', label='E_x (Explicit)')


    text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes, va='top')
    ax3.set_title("Charge Density and Electric Field")
    ax3.set_xlabel("Grid Index")
    ax3.set_ylabel("Value")
    ax3.legend()

    # Energy plot setup
    ax4.set_xlim(0, t_end)
    ax4.set_title("Energy vs Time")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Energy")

    # Plot fixed energy curves (solver_test only)
    ax4.plot(t_history, energy_total_history_test, linestyle='solid', label='E Field Energy (Implicit)')
    ax4.plot(t_history[1:], energy_kin_history_test[1:], linestyle='solid', label='Kinetic Energy (Implicit)')
    ax4.legend()

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

        x2 = x_ref_history[frame]
        v2 = v_ref_history[frame][0]
        sc2_a.set_data(x2[:Np // 2], v2[:Np // 2])
        sc2_b.set_data(x2[Np // 2:], v2[Np // 2:])
        text1.set_text(f"time = {t:.2f}")

        # Update rho and E plot
        grid_indices = np.arange(solver_test.Nx)
        rho_line_test.set_data(grid_indices, rho_test_history[frame])
        E_line_test.set_data(grid_indices, E_test_history[frame][0])

        rho_line_ref.set_data(grid_indices, rho_ref_history[frame])
        E_line_ref.set_data(grid_indices, E_ref_history[frame][0])

        #print(rho_test_history[frame])
        print(rho_ref_history[frame])

        #print(E_ref_history[frame])
        text3.set_text(f"time = {t:.2f}")

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