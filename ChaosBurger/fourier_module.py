import numpy as np

class FourierSolver:
    def __init__(self, mode: str = "1d"):
        if mode == "1d":
            self.forward = self.fourier_1d
            self.inverse = self.inverse_fourier_1d
        elif mode == "2d":
            self.forward = self.solve_2d
            self.inverse = self.inverse_fourier_2d
        else:
            raise ValueError("Unknown mode")

    def fourier_1d(self, data):

        return np.fft.fft(data)

    def inverse_fourier_1d(self, data):

        return  np.fft.ifft(data).real
    def solve_2d(self, data):
        print("Solving in 2D")

    def inverse_fourier_2d(self, data):
        print("Solving in 1D")
        return np.fft.ifft(data).real
    def dealise(self,arr,N):
        cutoff = N // 3
        arr[cutoff:-cutoff] = 0
        return arr