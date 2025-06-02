import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2 * np.pi  # Domain length
N = 128  # Number of grid points
nu = 0.01  # Viscosity
T = 1.0  # Total time
dt = 0.001  # Time step
n_steps = int(T / dt)

# Grid and wavenumbers
x = np.linspace(0, L, N, endpoint=False)  # Periodic domain [0, 2pi)
dx = L / N
k = np.fft.fftfreq(N, d=dx) * 2 * np.pi  # Wavenumbers
k2 = k ** 2  # For dissipation term

# Initial condition in physical space
u0 = np.sin(x)  # Example: u(x,0) = sin(x)

# FFT of initial condition
u_hat = np.fft.fft(u0)

# Forcing term (set to zero for simplicity)
f_hat = np.zeros_like(u_hat)

# Dealiasing: 2/3 rule (zero out highest 1/3 wavenumbers)
kmax = 2 * np.pi / (3 * dx)  # Maximum wavenumber for dealiasing
dealias = np.abs(k) <= kmax

# Time-stepping loop (Euler method)
u_hat_new = u_hat.copy()
for n in range(n_steps):
    # Dealias
    u_hat_dealiased = u_hat_new * dealias

    # Transform to physical space
    u = np.fft.ifft(u_hat_dealiased).real

    # Compute nonlinear term in physical space: tmp(x) = (1/2) * u^2
    tmp = 0.5 * u ** 2

    # FFT of nonlinear term
    tmp_hat = np.fft.fft(tmp)

    # Compute right-hand side in Fourier space
    rhs = -0.5j * k * tmp_hat + f_hat

    # Euler step: u_hat^(1) = e^(-nu k^2 dt) * [u_hat^(0) + dt * rhs]
    u_hat_new = np.exp(-nu * k2 * dt) * (u_hat_new + dt * rhs)

    # Optional: Store or visualize solution
    if n % 100 == 0:
        u = np.fft.ifft(u_hat_new).real
        plt.plot(x, u, label=f't={n * dt:.2f}')

# Plot final result
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.show()