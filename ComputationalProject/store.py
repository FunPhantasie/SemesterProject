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
def initialize_two_stream1D(Lx, Np,B,VT=0.005,V0=0.05, XP1=0.01,mode=1):
    """
    Initialize particle positions and velocities for a two-stream instability.

    Args:
        Lx (float): System length
        Np (int): Total number of particles
        amplitude (float): Amplitude of velocity perturbation

    Returns:
        tuple: (xp, vp_x) where xp is particle positions and vp_x is x-component of velocities
    """

    #xg = np.linspace(0, L - dx, NG) + dx / 2
    #Number of Grid Creates 0 bis L-dx
    #xp = np.linspace(0, L - L / Np, Np).T
    # Eavenly spaced bis L - L / Np
    vp = np.zeros([ Np])
    xp1 = 2 * Lx / Np * np.arange(Np // 2)
    xp2 = 2 * Lx / Np * np.arange(Np // 2)


    vp1 = V0 + XP1 * np.sin(2 *mode* np.pi / Lx * xp1)+sample_maxwellian_anisotropic(VT,Np//2)
    vp2 = -V0 - XP1 * np.sin(2 *mode* np.pi / Lx * xp1)+sample_maxwellian_anisotropic(VT,Np//2)
    xp = np.concatenate([xp1, xp2])
    vp_x = np.concatenate([vp1, vp2])
    vp = vp_x
    #B[2, ...] = 1
    print("Non Normalized Sampling")
    return xp, vp,B

def sample_maxwellian_anisotropic(vth_par, Np):
    # Sampling für anisotrope Maxwell-Verteilung (par = x, perp = y/z)

    vx = np.random.normal(loc=0.0, scale=vth_par, size=Np)
    return vx
