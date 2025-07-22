from Simulation.twostream import twostream1D
from Simulation.explicit_particle_sim import Explicit_PIC_Solver

from Analytics.AnalyticsOfNStep import run as run_nstep
from Analytics.ContiniousAnalytics import run as run_continious
from Analytics.FlipbookAnalytics import run as run_flipbook

"""
Choose Params
"""
border = 1
gridpoints = 128
NPpCell = 20
dt = 0.05
t_end = 3
nsteps = 2
total_steps = int(t_end / dt)

# Mode:
# 1 = N-Step
# 2 = Continous
# 3 = Flipbook
mode = 3

# Initialisiere Solver
solver_test = twostream1D(border, gridpoints, NPpCell, dt)
solver_ref = Explicit_PIC_Solver(border, gridpoints, NPpCell, dt)

# Dispatcher
if mode == 1:
    total_steps=6
    t_end = total_steps*dt
    run_nstep(solver_test, solver_ref, total_steps, t_end)
elif mode == 2:
    run_continious(solver_test, solver_ref, total_steps, t_end)
elif mode == 3:
    run_flipbook(solver_test, solver_ref, total_steps,t_end)
else:
    raise ValueError("Ungültiger Modus: Wähle 1, 2 oder 3")
