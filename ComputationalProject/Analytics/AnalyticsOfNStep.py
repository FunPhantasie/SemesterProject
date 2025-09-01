import matplotlib.pyplot as plt
import numpy as np
import os

def run_save_steps(solver_test, solver_ref, total_steps, dt, out_dir="NStepsOutput"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for step in range(total_steps):
        solver_test.step()
        solver_ref.step()

        electrons = solver_test.species[0]
        x1 = electrons["xp"]
        v1_x = electrons["vp"][0]
        rho_test = electrons["rho"] + 20
        E_test_x = solver_test.E_prev[0]
        B_x = solver_test.B[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Particles (x, v_x)
        Np = electrons["Np"]
        ax1.plot(x1[:Np//2], v1_x[:Np//2], 'o', markersize=0.5, label='Species A')
        ax1.plot(x1[Np//2:], v1_x[Np//2:], 'o', markersize=0.5, label='Species B')
        ax1.set_xlim(0, solver_test.Lx)
        ax1.set_ylim(-3, 3)
        ax1.set_title(f"Particles step {step+1}/{total_steps}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("v")
        ax1.legend()

        # Rho, E_x, B_x
        grid_indices = np.arange(solver_test.Nx)
        ax2.plot(grid_indices, rho_test, linestyle='dashed', label='rho (Implicit)', color='g')
        ax2.plot(grid_indices, B_x, linestyle='dashed', label='B_x (Implicit)', color='r')
        ax2.plot(grid_indices, E_test_x, linestyle='solid', label='E_x (Implicit)', color='black')
        ax2.set_title("Charge Density, Electric and Magnetic Field")
        ax2.set_xlabel("Grid index")
        ax2.set_ylabel("Value")
        ax2.legend()

        plt.tight_layout()
        filename = os.path.join(out_dir, f"step_{step:04d}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

        print(f"Saved {filename}")

    print(f"All {total_steps} steps saved in {out_dir}/")
