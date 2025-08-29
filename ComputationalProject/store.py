# Initialisiere Solver
solver_test = twostream1D(L, NG, PPC, DT)
solver_test.species[0]["xp"], solver_test.species[0]["vp"], solver_test.B = initialize_two_stream1D(solver_test.Lx, solver_test.species[0]["Np"], solver_test.B)
solver_test.step()

solver_ref = Explicit_PIC_Solver(L, NG, PPC, DT)
solver_ref.xp, solver_ref.vp, solver_ref.B = initialize_two_stream1D(solver_ref.Lx, solver_ref.Np, solver_ref.B)
# Referenzen aktualisieren
solver_ref.species[0]["xp"] = solver_ref.xp
solver_ref.species[0]["vp"] = solver_ref.vp
solver_ref.species[0]["rho"] = solver_ref.rho
solver_ref.Ekin0 = np.sum(solver_ref.vp ** 2) * 0.5



print("Chosen Parameters:")
print(f"dx: {L / NG:.6f}")
print("Starting Parameters for Explicit Solver:")

solver_ref.weight_rho()
print(f"Normalization Charge (qp): {solver_ref.charge:.6f}")
print("Density (rho):")
print(solver_ref.rho)






# ---------------------
# Kompakte Parameterpakete
# ---------------------

# Simulation & Auflösung
Nx_test = solver_test.Nx
Np_test = solver_test.species[0]["Np"]
Np_ref = solver_ref.Np
sim_params = (NT, DT, Nx_test, Np_test, Np_ref)

# Plot-Grenzen
plot_params = {
    "stream_limits": (-0.5, 0.5),
    "xlims_implicit": (0, solver_test.Lx),
    "xlims_explicit": (0, solver_ref.Lx),
    "moments_limits": (-1, 1),
    "energy_limits": (0, 5),
    "frame_duration_ms": 100
}

# Display Result / Calculating

if mode == 1:
    run_nstep(solver_test, solver_ref, NT, DT)
elif mode == 1:
    data_test = CallItRenderer(solver_test, NT, "TwoStreamRender/Implicit",step=False)
    data_ref = CallItRenderer(solver_ref, NT,"TwoStreamRender/Explicit",step=True)

    run_continuous(data_test,data_ref,sim_params, plot_params)
elif mode == 2:
    data_test = CallItRenderer(solver_test, NT, "TwoStreamRender/Implicit",step=False)
    data_ref = CallItRenderer(solver_ref, NT,"TwoStreamRender/Explicit",step=True)

    run_flipbook(data_test,data_ref,sim_params, plot_params)
else:
    raise ValueError("No Valid Mode: Choose 1, 2 or 3")
