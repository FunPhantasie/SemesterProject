# implementations of multiple statistic classes

import numpy as np
from helper import *

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
        return std(0.0, x, p, self.delta)

class Filter:
    # creates an object that filters samples with length n for observable values in the specified range from min to max
    # the range from min to max is divided into n_bins bins
    # this serves to find the mean value of samples that have a certain observable value
    # see filter(...) for more details
    def __init__(self, n, min, max, n_bins):
        self.n = n
        self.min = min
        self.max = max
        self.n_bins = n_bins

        self.delta = (self.max - self.min) / (self.n_bins - 1)
        self.bins = np.linspace(self.min, self.max, self.n_bins)
        self.samples = np.zeros((self.n_bins, self.n))
        self.counters = np.zeros(self.n_bins)

    # assigns all the elements of observable to the corresponding bins and sorts the sample accordingly
    # the sample will be shifted as if the observable had taken the value at index n // 2 (center) and then added up for averaging
    # optional: specify indices for the elements of observable to be considered
    def filter(self, sample, observable, indices = None):
        bins = self.get_bin(observable)
        valid = self.is_valid_index(bins)

        n = len(observable)

        if indices is None:
            indices = range(n)

        for i in indices:
            if valid[i]:
                self.samples[bins[i]] += np.roll(sample, n // 2 - i)
                self.counters[bins[i]] += 1

    # returns the index that corresponds to the value of the observable
    # observable can be a numpy array
    def get_bin(self, observable):
        return np.round((observable - self.min) / self.delta).astype('int')

    # returns True if index is a valid bin index
    # index can be a numpy array
    def is_valid_index(self, index):
        greater = 0 <= index
        less = index < self.n_bins

        return np.logical_and(less, greater)

    # calculates and returns the mean value for every bin
    def get_mean(self):
        return self.samples / np.maximum(self.counters[:, None], 1)

class StructureFunctions:
    # creates an object with which the structure functions and their exponents can be calculated
    # the r's will be spaced exponentially with basis r_basis to evenly distribute them in log coordinates
    # n:            number of cells in location space
    # n_p:          requested number of structure functions (p = 0, 1, ..., n_p - 1)
    # x_length:     spacial interval length
    # r_min, r_max: range for r ([r_min, r_max] intersect [dx, (n // 2 - 1) * dx])
    # r_basis:      ratio of two consecutive r's (default: 2)
    def __init__(self, n, n_p, x_length, r_min, r_max, r_basis=2):
        self.n = n

        self.n_p = n_p
        self.p = np.arange(self.n_p)
        self.p_min = np.min(self.p)
        self.p_max = np.max(self.p)

        self.x_length = x_length

        # clamp r_min, r_max
        self.r_min = np.max([self.x_length / self.n, r_min])
        self.r_max = np.min([(n // 2 - 1) * self.x_length / self.n, r_max])
        self.r_basis = r_basis
        # calculate the required number of r's
        self.n_r = int(np.round((np.log(self.r_max) - np.log(self.r_min)) / np.log(self.r_basis))) + 1

        # distribute the r's evenly in log coordinates
        self.log_r = np.linspace(np.log(self.r_min), np.log(self.r_max), self.n_r)
        # calculate the r's and their corresponding indices
        self.r = np.exp(self.log_r)
        self.r_index = np.round((self.r / self.x_length * self.n)).astype('int')
        # remove any duplicates
        self.r_index = np.unique(self.r_index)
        # recalculate the r's and log r's from the indices
        self.r = self.r_index * self.x_length / self.n
        self.log_r = np.log(self.r)

        # update n_r, r_max, r_min
        self.n_r = len(self.r_index)
        self.r_max = np.max(self.r)
        self.r_min = np.min(self.r)

        self.samples = np.zeros((self.n_p, self.n_r))
        self.count = 0

    # feeds samples for averaging
    # for this purpose, the absolute increments for all specified r's are calculated from sample, exponentiated with all p's and then added up for averaging
    def feed(self, sample):
        for i_r in range(self.n_r):
            increment = np.abs(np.roll(sample, - self.r_index[i_r]) - sample)
            increments = np.tile(increment, (self.n_p, 1))

            self.samples[:, i_r] += np.mean(increments ** self.p[:, None], axis=1)

        self.count += 1

    # calculates and returns all structure functions
    def get_mean(self):
        return self.samples / self.count

    # calculates the exponents for all structure functions by linear regression
    # optional: specify the range for linear regression with r_min, r_max
    def get_exponents(self, r_min=-1.0, r_max=-1.0):
        if r_min == -1.0:
            r_min = self.r_min

        if r_max == -1.0:
            r_max = self.r_max

        greater = self.r >= r_min
        less = self.r <= r_max

        condition = np.logical_and(less, greater)

        s = self.samples / self.count
        log_r = np.extract(condition, self.log_r)

        exponents = np.zeros(self.n_p)

        for p in range(self.n_p):
            log_s_p = np.log(np.extract(condition, s[p]))

            if len(log_s_p) > 1:
                #linear regression
                line = np.polyfit(log_r, log_s_p, 1)

                slope = line[0]
                exponents[p] = slope

        return exponents

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
