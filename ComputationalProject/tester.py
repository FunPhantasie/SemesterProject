import matplotlib.pyplot as plt
import matplotlib.animation as animation
from semi_implicit_particle_sim import PIC_Solver
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
# Initialisiere den Solver
solver = PIC_Solver()

# Zufällige Teilchenverteilung
solver.xp = np.random.rand(3, solver.Np)
solver.xp[0] *= solver.Lx
solver.xp[1] *= solver.Ly
solver.xp[2] *= solver.Lz

# Zwei gegenläufige Strahlen in x-Richtung
v_drift = 0.1
solver.vp[0, :] = v_drift * np.where(np.arange(solver.Np) < solver.Np // 2, 1, -1)
solver.vp[1:] = 0

# Felder setzen
solver.E[:] = 0
solver.B[2, :, :, :] = 1.0  # B-Feld in z

# Set up Figure
fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(solver.xp[0], solver.xp[1], s=10, c='red')
ax.set_xlim(0, solver.Lx)
ax.set_ylim(0, solver.Ly)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('2D Particle Motion')
ax.grid(True)
ax.set_aspect('equal')

# Update-Funktion für die Animation
def update(frame):
    solver.step()
    sc.set_offsets(np.c_[solver.xp[0], solver.xp[1]])
    ax.set_title(f'2D Particle Motion — Step {frame}, t = {solver.t:.2f}')
    return sc,

# Animation ausführen
ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
plt.tight_layout()
plt.show()
