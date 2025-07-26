import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from .RenderManager import CallItRenderer

import matplotlib as mpl

mpl.use('TkAgg')





def run(data_test, data_ref, sim_params, plot_params):
    total_steps, t_end, Nx_test, Np_test, Np_ref = sim_params
    # Werte aus dem Dictionary extrahieren
    ymin, ymax = plot_params["stream_limits"]
    xmin_im, xmax_im = plot_params["xlims_implicit"]
    xmin_ex, xmax_ex = plot_params["xlims_explicit"]
    ymin_M, ymax_M = plot_params["moments_limits"]
    ymin_E, ymax_E = plot_params["energy_limits"]
    speed = plot_params["frame_duration_ms"]


    x_test_history_e = data_test["x_e"]
    v_test_history_e = data_test["v_e"]
    #x_test_history_i = data_test["x_i"]
    #v_test_history_i = data_test["v_i"]
    t_test_history = data_test["t"]
    energy_total_history_test = data_test["energy_total"]
    energy_kin_history_test = data_test["energy_kin"]
    rho_test_history_e = data_test["rho_e"]
    #rho_test_history_i = data_test["rho_i"]
    E_test_history = data_test["E"]
    B_test_history = data_test["B"]


    # Referenzdaten
    x_ref_history_e = data_ref["x_e"]
    v_ref_history_e = data_ref["v_e"]
    #x_ref_history_i = data_ref["x_i"]
    #v_ref_history_i = data_ref["v_i"]
    energy_total_history_ref = data_ref["energy_total"]
    t_ref_history = data_ref["t"]
    energy_kin_history_ref = data_ref["energy_kin"]
    rho_ref_history_e = data_ref["rho_e"]
    #rho_ref_history_i = data_ref["rho_i"]
    E_ref_history = data_ref["E"]
    B_ref_history = data_ref["B"]




    """
    Plot Settings
    """
    # Setup plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

    # Plot for solver_test particles
    ax1.set_xlim(xmin_im,xmax_im)
    ax1.set_ylim(ymin, ymax)
    sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
    sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
    text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
    ax1.set_title("PIC1D Implicit")
    ax1.set_xlabel("x")
    ax1.set_ylabel("v")
    ax1.legend()


    # Plot for solver_test particles
    ax2.set_xlim(xmin_ex, xmax_ex)
    ax2.set_ylim(ymin, ymax)
    sc2_a, = ax2.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
    sc2_b, = ax2.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
    text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
    ax2.set_title("PIC1D Explicit")
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.legend()


    # Plot for rho and E_x (solver_test only)
    ax3.set_xlim(0, Nx_test)
    ax3.set_ylim(ymin_M,ymax_M)  # Adjust based on expected rho and E ranges

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
    ax4.plot(t_test_history, energy_total_history_test, linestyle='solid', label='E Field Energy (Implicit)')
    ax4.plot(t_test_history[1:], energy_kin_history_test[1:], linestyle='solid', label='Kinetic Energy (Implicit)')
    ax4.legend()

    # Initialize current frame index
    current_frame = [0]  # Use list to allow modification in callback


    def update_plot(frame):

        x1 = x_test_history_e[frame]
        v1 = v_test_history_e[frame][0]
        t_test = t_test_history[frame]
        t_ref=t_ref_history[frame]
        # Update particle plot
        sc1_a.set_data(x1[:Np_test // 2], v1[:Np_test // 2])
        sc1_b.set_data(x1[Np_test // 2:], v1[Np_test // 2:])
        text1.set_text(f"time = {t_test:.2f}")

        x2 = x_ref_history_e[frame]
        v2 = v_ref_history_e[frame][0]
        sc2_a.set_data(x2[:Np_ref // 2], v2[:Np_ref // 2])
        sc2_b.set_data(x2[Np_ref // 2:], v2[Np_ref // 2:])
        text2.set_text(f"time = {t_ref:.2f}")

        # Update rho and E plot
        grid_indices = np.arange(Nx_test)
        #rho_line_test.set_data(grid_indices, rho_test_history[frame])
        #E_line_test.set_data(grid_indices, E_test_history[frame][0])

        rho_line_ref.set_data(grid_indices, rho_ref_history_e[frame])
        E_line_ref.set_data(grid_indices, E_ref_history[frame][0])



        #print(E_ref_history[frame])
        text3.set_text(f"time = {t_test:.2f}")

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