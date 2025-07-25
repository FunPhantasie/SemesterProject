import matplotlib.pyplot as plt
import matplotlib.animation as animation

from RenderManager import CallItRenderer
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')



def run(solver_test,total_steps,dt,ShowSingleStep=None):
    if ShowSingleStep is None:
        data = CallItRenderer(solver_test, total_steps)

        x_test_history_e = data["x_e"]
        v_test_history_e = data["v_e"]
        x_test_history_i = data["x_i"]
        v_test_history_i = data["v_i"]
        t_history = data["t"]
        energy_total_history_test = data["energy_total"]
        energy_kin_history_test = data["energy_kin"]
        rho_test_history_e = data["rho_e"]
        rho_test_history_i = data["rho_i"]
        E_test_history = data["E"]
        B_test_history = data["B"]


        Np = solver_test.species[0]["Np"]
        Lx =solver_test.Lx
        Continous( x_test_history_e, v_test_history_e,x_test_history_i,v_test_history_i, t_history, \
                  energy_total_history_test, energy_kin_history_test, \
                   rho_test_history_e,rho_test_history_i, \
                  E_test_history, Lx, Np,total_steps)


    elif ShowSingleStep is not None:
        #Np = solver_test.species[0]["Np"]
        #Lx = solver_test.Lx
        AfterNSteps(solver_test, ShowSingleStep,dt)


def Continous( x_test_history_e, v_test_history_e,x_test_history_i,v_test_history_i,\
               t_history, \
               energy_total_history_test, energy_kin_history_test, \
               rho_test_history_e,rho_test_history_i, \
               E_test_history, Lx, Np,total_steps):
    """
    Plott Settings
    """
    # Setup plots
    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(18, 5), sharey=False)

    # Plot for solver_test
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(-3, 3)
    sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
    sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
    text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
    ax1.set_title("Your PIC1D")
    ax1.set_xlabel("x")
    ax1.set_ylabel("v")
    ax1.legend()



    # Energy plot setup
    ax3.set_xlim(0, 5)

    ax3.set_title("Energy vs Time")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Energy")

    a=energy_total_history_test
    b=energy_kin_history_test





    #ax3.set_yscale('log')
    ax3.plot(t_history, a, '-', label='E Field Energy')

    ax3.plot(t_history, b, '-', label='Kinetic Energy')


    ax3.legend()



    def init():
        # Initialize particle plots as empty
        sc1_a.set_data([], [])
        sc1_b.set_data([], [])
        text1.set_text('')



        return sc1_a, sc1_b,  text1,

    def update(frame):

        x1_e= x_test_history_e[frame][0]
        v1_e = v_test_history_e[frame][0]
        x1_i = x_test_history_i[frame][0]
        v1_i = v_test_history_i[frame][0]
        t = t_history[frame]

        sc1_a.set_data(x1_e, v1_e)
        sc1_b.set_data(x1_i, v1_i)
        text1.set_text(f"time = {t:.2f}")



        return sc1_a, sc1_b,  text1,

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_steps,
                                  interval=100, blit=True)
    plt.tight_layout()
    plt.show()

def AfterNSteps(solver_test,total_steps,dt):
    # Run simulation for n steps
    for step in range(total_steps):
        print(step)
        #solver_test.step()


    solver_test.analyze_E_theta_RHS()

    # After simulation, get final states
    electrons=solver_test.species[0]
    x1_e = electrons["xp"][0]
    v1_x_e = electrons["vp"][0]
    ions = solver_test.species[1]
    x1_i = ions["xp"][0]
    v1_x_i = ions["vp"][0]

    rho_test = electrons["rho"][:,0,0]
    E_test_x = solver_test.E[0][:,0,0]
    E_test_y = solver_test.E[1][:,0,0]
    E_test_z = solver_test.E[2][:,0,0]
    B_x=solver_test.B[0][:,0,0]

    #rho_ref = solver_ref.rho
    #E_ref = solver_ref.E[0]

    # Compute energies for the whole run to plot energy vs time
    # (or you could skip this entirely to keep it minimal)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot particles (x,v)
    Np = solver_test.species[0]["Np"]
    ax1.plot(x1_e, v1_x_e, 'o', markersize=0.5, label='Species A')
    ax1.plot(x1_i, v1_x_i, 'o', markersize=0.5, label='Species B')

    ax1.set_xlim(0, solver_test.Lx)
    ax1.set_ylim(-3, 3)
    ax1.set_title(f"Particles after {total_steps} steps (t = {dt*total_steps:.3f})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("v")
    ax1.legend()

    # Plot rho and E_x on grid
    grid_indices = np.arange(solver_test.Nx)
    print(np.shape(grid_indices))
    print(np.shape(B_x))
    ax2.plot(grid_indices, rho_test, linestyle='dashed', label='rho (Implicit)', color='g')
    ax2.plot(grid_indices, B_x, linestyle='dashed', label='B_x (Implicit)', color='r')
    ax2.plot(grid_indices, E_test_x, linestyle='solid', label='E_x (Implicit)', color='black')
    #ax2.plot(grid_indices, E_test_y, linestyle='solid', label='E_y (Implicit)', color='blue')
    #ax2.plot(grid_indices, E_test_z, linestyle='solid', label='E_z (Implicit)', color='magenta')


    ymin, ymax = ax2.get_ylim()
    abs_max = max(abs(ymin), abs(ymax))

    if abs_max < 0.1:
        ax2.set_ylim(-0.1, 0.1)
    ax2.set_title("Charge Density and Electric Field")
    ax2.set_xlabel("Grid index")
    ax2.set_ylabel("Value")
    ax2.legend()

    plt.tight_layout()
    plt.show()