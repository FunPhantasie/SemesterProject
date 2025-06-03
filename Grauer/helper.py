import numpy as np

# helper functions

# integrate: discrete integral
# norm:      discrete $L^2$ norm (euclidean norm) 
# std:       standard deviation

def integrate(y, dx):
    return np.sum(y) * dx

def norm(y, dx):
    return np.sqrt(np.sum(np.abs(y) ** 2) * dx)

def std(mu, x, px, dx):
    return np.sqrt(integrate((x - mu) ** 2.0 * px, dx))

# fourier transform
# helper functions for computing Fourier transforms on a 1-D grid
# x:         cell values
# n:         number of cells
# a:         start coordinate
# b:         end coordinate

# ft:        Fourier transform
# rft:       real Fourier transform

# <>:        Fourier transform
# <>_k:      k-space coordinates
# <>_dk:     cell size in k-space
# i<>:       inverse Fourier transform

def ft_k(n, a=0.0, b=1.0):
    return np.linspace(- (n // 2), n // 2 - 1 + (n % 2), n) * 2.0 * np.pi / (b - a)

def ft_dk(a=0.0, b=1.0):
    return 2.0 * np.pi / (b - a)

def ft(x, a=0.0, b=1.0):
    n = len(x)
    xhat = np.fft.fftshift(np.fft.fft(x)) * ((b - a) / n / np.sqrt(2.0 * np.pi))

    if a != 0.0:
        k = ft_k(n, a, b)
        xhat = xhat * np.exp(- 1.0j * k * a)

    return xhat

def ift(xhat, a=0.0, b=1.0):
    n = len(xhat)

    if a != 0.0:
        k = ft_k(n, a, b)
        xhat = xhat * np.exp(1.0j * k * a)

    x = np.fft.ifft(np.fft.ifftshift(xhat * (n * np.sqrt(2.0 * np.pi) / (b - a))))

    return x

def rft_k(n, a=0.0, b=1.0):
    return np.linspace(0, n - 1, n) * 2.0 * np.pi / (b - a)

def rft_dk(a=0.0, b=1.0):
    return 2.0 * np.pi / (b - a)

def rft(x, a=0.0, b=1.0):
    n = len(x)
    xhat = np.fft.rfft(x) * ((b - a) / n / np.sqrt(2.0 * np.pi))

    if a != 0.0:
        k = rft_k(n // 2 + 1, a, b)
        xhat = xhat * np.exp(- 1.0j * k * a)

    return xhat

def irft(xhat, a=0.0, b=1.0):
    n = len(xhat)

    if a != 0.0:
        k = rft_k(n, a, b)
        xhat = xhat * np.exp(1.0j * k * a)

    x = np.fft.irfft(xhat * ((n - 1.0) * 2.0 * np.sqrt(2.0 * np.pi) / (b - a)))

    return x
