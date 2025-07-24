import matplotlib.pyplot as plt
import matplotlib.animation as animation

from .RenderSavings import CallItRenderer

import matplotlib as mpl

mpl.use('TkAgg')



def run(solver_test,solver_ref,total_steps,t_end):
    x_test_history, v_test_history, x_ref_history, v_ref_history, t_history, \
        energy_total_history_test, energy_kin_history_test, rho_test_history, \
        E_test_history, rho_ref_history, E_ref_history, energy_total_history_ref, energy_kin_history_ref = CallItRenderer(
        solver_test, solver_ref, total_steps, t_end)



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
    ax3.plot(t_history, b, '-', label='Kinetic Energy')
    ax3.plot(t_history, c, '-', label='E Field Energy Ref')
    ax3.plot(t_history, d, '-', label='Kinetic Energy Ref')
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
        Np = solver_test.species[0]["Np"]
        Np2 = solver_ref.Np

        x1 = x_test_history[frame]
        v1 = v_test_history[frame][0]
        x2 = x_ref_history[frame]
        v2 = v_ref_history[frame][0]
        t = t_history[frame]

        sc1_a.set_data(x1[:Np // 2], v1[:Np // 2])
        sc1_b.set_data(x1[Np // 2:], v1[Np // 2:])
        text1.set_text(f"time = {t:.2f}")

        sc2_a.set_data(x2[:Np2 // 2], v2[:Np2 // 2])
        sc2_b.set_data(x2[Np2 // 2:], v2[Np2 // 2:])
        text2.set_text(f"time = {t:.2f}")

        return sc1_a, sc1_b, sc2_a, sc2_b, text1, text2

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_steps,
                                  interval=100, blit=True)

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
