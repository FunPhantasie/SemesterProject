# implementations of multiple statistic classes

import numpy as np


class Histogram:
    # creates a histogram object with n bins from min to max
    def __init__(self, min, max, n):
        self.min = min
        self.max = max
        self.n = n

        self.delta = (self.max - self.min) / self.n
        self.bins = np.linspace(self.min - 0.5 * self.delta, self.max + 0.5 * self.delta, self.n + 1)
        self.histogram = np.zeros(self.n)
        self.count = 0

    # categorizes the sample into the bins and increments the corresponding counters
    def categorize(self, sample):
        self.histogram += np.histogram(sample, bins=self.bins)[0]
        self.count += len(sample)

    # creates and returns a pdf from the histogram data
    def pdf(self):
        return (np.linspace(self.min, self.max, self.n), self.histogram / self.count / self.delta)

    # calculates and returns the standard deviation of the pdf
    def std(self):
        x, p = self.pdf()

        mu=0.0
        arg=(x - mu) ** 2.0 * p
        integ=np.sum(arg)* self.delta
        return np.sqrt(integ)


class EnergySpectrum:
    # creates an object which provides energy and energy related statistics
    # x_length: spacial interval length
    # dk:       cell size in Fourier space
    # n_k:      number of Fourier modes
    # nu:       viscosity constant
    def __init__(self, x_length, dk, n_k, nu):
        self.x_length = x_length
        self.dk = dk
        self.n_k = n_k
        self.u2_samples = 0.0
        self.uhat2_samples = np.zeros(n_k)
        self.dudx2_samples = 0.0
        self.count = 0
        self.nu = nu

    # feeds samples for averaging
    # for this purpose, the velocity field u, its Fourier transform uhat and its gradient dudx are squared and added up
    def feed(self, u, uhat, dudx):
        self.u2_samples += np.mean(np.square(u))
        self.uhat2_samples += np.square(np.abs(uhat))
        self.dudx2_samples += np.mean(np.square(dudx))
        self.count += 1

    # calculates and returns the root mean square of the velocity u
    def get_u_rms(self):
        return np.sqrt(self.u2_samples / self.count)

    # calculates and returns the mean kinetic energy
    def get_mean_kin_e(self):
        return 0.5 * self.get_u_rms() ** 2.0

    # calculates and returns the mean spectral energy density
    def get_mean_spectral_e_density(self):
        return 0.5 * self.uhat2_samples / self.count

    # calculates and returns the mean energy dissipation rate
    def get_mean_e_diss(self):
        return 2.0 * self.nu * self.dudx2_samples / self.count

    # calculates and returns the average shock length
    def get_shock_length(self):
        return 12.0 * self.nu / self.get_u_rms()

    # calculates and returns the wavenumber corresponding to the average shock length
    def get_shock_wavenumber(self):
        return self.length_to_wavenumber(self.get_shock_length())

    # calculates and returns the average integral length
    def get_integral_length(self):
        return self.get_u_rms() ** 3.0 / self.get_mean_e_diss()

    # calculates and returns the wavenumber corresponding to the average integral length
    def get_integral_wavenumber(self):
        return self.length_to_wavenumber(self.get_integral_length())

    # takes a length which will be converted into a wavenumber
    def length_to_wavenumber(self, length):
        return self.x_length / length * self.dk

    # calculates and returns the reynolds number
    def get_reynolds_number(self):
        return self.get_u_rms() * self.get_integral_length() / self.nu
