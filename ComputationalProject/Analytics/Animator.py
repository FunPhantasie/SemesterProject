import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')

def setup_plot(data_test, data_ref, sim_params, plot_params):
    total_steps, t_end, Nx_test, Np_test, Np_ref = sim_params
    ymin, ymax = plot_params["stream_limits"]
    xmin_im, xmax_im = plot_params["xlims_implicit"]
    xmin_ex, xmax_ex = plot_params["xlims_explicit"]
    ymin_M, ymax_M = plot_params["moments_limits"]
    ymin_E, ymax_E = plot_params["energy_limits"]

    # Daten auslesen
    x_test_history_e = data_test["x_e"]
    v_test_history_e = data_test["v_e"]
    t_test_history = data_test["t"]
    rho_test_history_e = data_test["rho_e"]
    E_test_history = data_test["E"]
    electric_energy_history_test = data_test["electric_energy"]
    energy_kin_history_test = data_test["energy_kin"]

    x_ref_history_e = data_ref["x_e"]
    v_ref_history_e = data_ref["v_e"]
    t_ref_history = data_ref["t"]
    rho_ref_history_e = data_ref["rho_e"]
    electric_energy_history_ref = data_ref["electric_energy"]
    energy_kin_history_ref = data_ref["energy_kin"]
    E_ref_history = data_ref["E"]

    # Subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10), sharey=False)

    # Implicit
    ax1.set_xlim(xmin_im, xmax_im)
    ax1.set_ylim(ymin, ymax)
    sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b',label='Species A')
    sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r',label='Species B')
    text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
    ax1.set_xlabel("x")
    ax1.set_ylabel("v")
    ax1.legend()
    ax1.set_title("PIC1D Implicit")

    # Explicit
    ax2.set_xlim(xmin_ex, xmax_ex)
    ax2.set_ylim(ymin, ymax)
    sc2_a, = ax2.plot([], [], 'o', markersize=0.5, color='b',label='Species A')
    sc2_b, = ax2.plot([], [], 'o', markersize=0.5, color='r',label='Species B')
    text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.legend()
    ax2.set_title("PIC1D Explicit")

    # rho/E Plot
    ax3.set_xlim(0, Nx_test)
    ax3.set_ylim(ymin_M, ymax_M)

    rho_line_test, = ax3.plot([], [], linestyle='dashed', color='g', label='rho (Implicit)')
    E_line_test, = ax3.plot([], [], linestyle='dashed', color='r', label='E_x (Implicit)')

    rho_line_ref, = ax3.plot([], [], linestyle='solid', color='black', label='rho (Explicit)')
    E_line_ref, = ax3.plot([], [], linestyle='dotted', color='m', label='E_x (Explicit)')

    text3 = ax3.text(0.02, 0.95, '', transform=ax3.transAxes)
    ax3.set_title("Charge Density and Electric Field")
    ax3.set_xlabel("Grid Index")
    ax3.set_ylabel("Value")
    ax3.legend()


    # Energy Plot
    ax4.set_xlim(0, t_end)
    ax4.set_ylim(ymin_E, ymax_E)
    ax4.set_title("Energy vs Time")
    ax4.set_xlabel("Time")
    ax3.set_ylabel("Energy")
    ax4.set_title("Energy Development")
    ax4.plot(t_test_history, electric_energy_history_test, label='E Field Energy')
    ax4.plot(t_test_history, energy_kin_history_test, label='Kinetic Energy')
    ax4.plot(t_ref_history, electric_energy_history_ref, label='E Field Energy (Explicit)', linestyle='solid')
    ax4.plot(t_ref_history, energy_kin_history_ref, label='Kinetic Energy (Explicit)', linestyle='solid')

    ax4.legend()

    artists = {
        "sc1_a": sc1_a, "sc1_b": sc1_b, "sc2_a": sc2_a, "sc2_b": sc2_b,
        "text1": text1, "text2": text2, "text3": text3,
        "rho_line_ref": rho_line_ref, "E_line_ref": E_line_ref,
        "rho_line_test":rho_line_test, "E_line_test": E_line_test
    }

    data = {
        "x_test": x_test_history_e, "v_test": v_test_history_e,
        "x_ref": x_ref_history_e, "v_ref": v_ref_history_e,
        "t_test": t_test_history, "t_ref": t_ref_history,
        "rho_ref": rho_ref_history_e, "E_ref": E_ref_history,
        "rho_test":rho_test_history_e,"E_test":E_test_history
    }

    return fig, artists, data

def run_continuous(data_test, data_ref, sim_params, plot_params):
    fig, art, data = setup_plot(data_test, data_ref, sim_params, plot_params)
    total_steps, _, Nx_test, Np_test, Np_ref = sim_params
    speed = plot_params["frame_duration_ms"]

    def update(frame):
        x1, v1 = data["x_test"][frame], data["v_test"][frame][0]
        x2, v2 = data["x_ref"][frame], data["v_ref"][frame][0]

        art["sc1_a"].set_data(x1[:Np_test//2], v1[:Np_test//2])
        art["sc1_b"].set_data(x1[Np_test//2:], v1[Np_test//2:])
        art["text1"].set_text(f"time = {data['t_test'][frame]:.2f}")

        art["sc2_a"].set_data(x2[:Np_ref//2], v2[:Np_ref//2])
        art["sc2_b"].set_data(x2[Np_ref//2:], v2[Np_ref//2:])
        art["text2"].set_text(f"time = {data['t_ref'][frame]:.2f}")

        grid = np.arange(Nx_test)
        art["rho_line_ref"].set_data(grid, data["rho_ref"][frame])
        art["E_line_ref"].set_data(grid, data["E_ref"][frame][0])
        art["rho_line_test"].set_data(grid, data["rho_test"][frame])
        art["E_line_test"].set_data(grid, data["E_test"][frame][0])

        art["text3"].set_text(f"t_ref = {data['t_ref'][frame]:.2f}")
        if frame%30==0 and False:
            frame_data = data["rho_ref"][frame]
            print(f"Rho data: Min: {frame_data.min()}, Max: {frame_data.max()}")
            frame_data = data["E_ref"][frame][0]
            print(f"Electric Field data: Min: {frame_data.min()}, Max: {frame_data.max()}")
        if frame % 30 == 0 and True:
            frame_data = data["rho_test"][frame]
            print(f"Rho data IMplicit: Min: {frame_data.min()}, Max: {frame_data.max()}")
            frame_data = data["E_test"][frame][0]
            print(f"Electric Field data IMplicit: Min: {frame_data.min()}, Max: {frame_data.max()}")

        return list(art.values())

    ani = animation.FuncAnimation(fig, update, frames=total_steps, interval=speed, blit=True)
    plt.tight_layout()
    plt.show()

def run_flipbook(data_test, data_ref, sim_params, plot_params):
    fig, art, data = setup_plot(data_test, data_ref, sim_params, plot_params)
    total_steps, _, Nx_test, Np_test, Np_ref = sim_params
    current_frame = [0]

    def update_plot(frame):
        x1, v1 = data["x_test"][frame], data["v_test"][frame][0]
        x2, v2 = data["x_ref"][frame], data["v_ref"][frame][0]

        art["sc1_a"].set_data(x1[:Np_test//2], v1[:Np_test//2])
        art["sc1_b"].set_data(x1[Np_test//2:], v1[Np_test//2:])
        art["text1"].set_text(f"time = {data['t_test'][frame]:.2f}")

        art["sc2_a"].set_data(x2[:Np_ref//2], v2[:Np_ref//2])
        art["sc2_b"].set_data(x2[Np_ref//2:], v2[Np_ref//2:])
        art["text2"].set_text(f"time = {data['t_ref'][frame]:.2f}")

        grid = np.arange(Nx_test)
        art["rho_line_ref"].set_data(grid, data["rho_ref"][frame])
        art["E_line_ref"].set_data(grid, data["E_ref"][frame][0])
        art["rho_line_test"].set_data(grid, data["rho_test"][frame])
        art["E_line_test"].set_data(grid, data["E_test"][frame][0])
        art["text3"].set_text(f"t_ref = {data['t_ref'][frame]:.2f}")

        frame_data = data["rho_ref"][frame]
        print(f"Rho data: Min: {frame_data.min()}, Max: {frame_data.max()}")
        frame_data = data["E_ref"][frame][0]
        print(f"Electric Field data: Min: {frame_data.min()}, Max: {frame_data.max()}")

        fig.canvas.draw()

    def on_click(event):
        if event.button == 1:
            current_frame[0] = (current_frame[0] + 1) % total_steps
        elif event.button == 3:
            current_frame[0] = (current_frame[0] - 1) % total_steps
        update_plot(current_frame[0])

    update_plot(current_frame[0])
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
