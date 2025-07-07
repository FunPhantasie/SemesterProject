############ Exercise 3 ###########
# (1) Fill in the blanks to complete the compression and decompression of a 4d tensor.
# (2) Adjust the parameters and initial condition and note any changes.


import numpy as np
import matplotlib.pyplot as plt


def svd(A):
    try:
        return np.linalg.svd(A, full_matrices=False)
    except:  # LinAlgError
        try:
            print("SVD failed")
            return np.linalg.svd(A + 1e-12 * np.linalg.norm(A, 1), full_matrices=False)
        except:
            print("SVD failed twice")
            return np.linalg.svd(A + 1e-8 * np.linalg.norm(A, 1), full_matrices=False)


class htuck4d:

    def __init__(self, A=None, eps=1e-14):
        if A is None:
            self.core = 0
            self.u = [0, 0, 0, 0]
            self.n = [0, 0, 0, 0]
            self.r = [0, 0, 0, 0]
            return

        N1, N2, N3, N4 = A.shape

        # A --> U12, U34, G1234

        B12 = np.reshape(A, (N1 * N2, -1))  # B: Matricization, in slides called M
        # A 4d zu B12 n12xn34

        B34 = np.transpose(B12)

        U12, V12, r12 = svd_trunc(B12, eps)
        U34, V34, r34 = svd_trunc(B34, eps)

        G1234 = np.tensordot(B12, U34, (1, 0))  # in slides denoted by fancy B
        G1234 = np.tensordot(G1234, U12, (0, 0)) #wtih pre step

        # U12 --> U1, U2, G12
        B1= np.reshape(U12,(N1,-1))
        B2 =np.reshape(U12,(N2,-1))
        U1,V1,r1 =svd_trunc(B1,eps)
        U2, V2, r2 = svd_trunc(B2, eps)
        U12r=np.reshape(U12,(N1,N2,r12))

        G12 = np.tensordot(U12r, U2, (1, 0))  # in slides denoted by fancy B

        G12 = np.tensordot(G12, U1, (0, 0))  # wtih pre step
        G12=np.transpose(G12,[2,1,0])

        # TODO

        # U34 --> U3, U4, G34

        B3 = np.reshape(U34, (N3, -1))
        B4 = np.reshape(U34, (N4, -1))
        U3, V3, r3 = svd_trunc(B3, eps)
        U4, V4, r4 = svd_trunc(B4, eps)
        U34r = np.reshape(U34, (N3, N4, r34))
        G34 = np.tensordot(U34r, U4, (1, 0))  # statt (1, 0)
        G34 = np.tensordot(G34, U3, (0, 0))  # statt (0, 0)
        G34=np.transpose(G34, [2, 1, 0])
        # TODO

        self.n = (N1, N2, N3, N4)
        self.r = [r1, r2, r3, r4, r12, r34]
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U4 = U4
        self.U12 = U12
        self.U34 = U34
        self.G12 = G12
        self.G34 = G34
        self.G1234 = G1234

    def size(self):
        size = self.G12.size + self.G34.size + self.G1234.size \
               + self.U1.size + self.U2.size + self.U3.size + self.U4.size
        sizefull = self.n[0] * self.n[1] * self.n[2] * self.n[3] * self.n[4]
        return size, sizefull, sizefull / float(size)

    def __repr__(self):
        res = "This is a 4D tensor in the HTucker format with \n"

        print("n = ", self.n)
        print("r = ", self.r)

        return res

    def full(self):
        N1, N2, N3, N4 = self.n

        # U1, U2, G12 --> U12
        G12 = self.G12
        U1 = self.U1
        U2 = self.U2

        print(np.shape(G12))
        print(np.shape(U2))
        print(np.shape(U1))

        U12 = np.tensordot(G12, U2, (1, 1))
        U12 = np.tensordot(U12, U1, (0, 1))
        U12 = U12.transpose([2, 1, 0])

        U12 = U12.reshape(N1 * N2, -1)

        # U3, U4, G34 --> U34



        # TODO

        print("U34.shape = ", U34.shape)
        print("should be: ", self.U34.shape)

        # U12, U34, G1234 --> A
        G1234 = self.G1234

        U34 = U34.reshape(N3 * N4, -1)
        A = np.tensordot(G1234, U34, (0, 1))
        A = np.tensordot(A, U12, (0, 1))
        A = A.transpose()

        A = A.reshape(N1, N2, N3, N4)

        return A


def svd_trunc(A, eps=1e-14):
    u, s, v = svd(A)

    N1, N2 = A.shape

    eps_svd = eps * s[0] / np.sqrt(3)
    r = min(N1, N2)
    for i in range(min(N1, N2)):
        if s[i] <= eps_svd:
            r = i
            break
        # print s/s[0]
    u = u[:, :r].copy()
    v = v[:r, :].copy()
    s = s[:r].copy()

    return u, H(v), r


def H(A):
    return np.transpose(A)


###################################################### main ####################################################
N = 32
eps = 1e-15  # the smaller eps, the more singular values are kept
print("eps: ", eps)
# A = np.random.rand(N,N,N,N)

# Gaussian blob with added anisotropy

from scipy.stats import multivariate_normal

mean_main = [0, 0, 0, 0]
covariance_main = [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]]

mean_secondary = [1, -1, 1, -1]
covariance_secondary = [[0.5, 0, 0, 0],
                        [0, 0.5, 0, 0],
                        [0, 0, 0.5, 0],
                        [0, 0, 0, 0.5]]

shape = (N, N, N, N)  # Example shape

x, y, z, w = np.meshgrid(np.linspace(-3, 3, shape[0]),
                         np.linspace(-3, 3, shape[1]),
                         np.linspace(-3, 3, shape[2]),
                         np.linspace(-3, 3, shape[3]), indexing='ij')

# Reshape the coordinates into a (n_points, n_dims) array
coords = np.column_stack([x.flatten(), y.flatten(), z.flatten(), w.flatten()])

values_main = multivariate_normal.pdf(coords, mean=mean_main, cov=covariance_main)
values_secondary = multivariate_normal.pdf(coords, mean=mean_secondary, cov=covariance_secondary)
values_combined = values_main + values_secondary

# Reshape the values into the shape of the 4D array
A = values_combined.reshape(shape)

# print(A.max(), A.min())
htuck = htuck4d(A=A, eps=eps)
A_tuck = htuck.full()
print("L_max difference from input:", np.abs(A - A_tuck).max())

fig, axs = plt.subplots(1, 3)
im_full = axs[0].imshow(A[N // 2, N // 2, :, :])
im_tuck = axs[1].imshow(A_tuck[N // 2, N // 2, :, :])
im_diff = axs[2].imshow(A[N // 2, N // 2, :, :] - A_tuck[N // 2, N // 2, :, :])

axs[0].set_title('Input')
axs[1].set_title('after HTucker')
axs[2].set_title('Difference to input after HTucker')
# fig.colorbar(im_full, ax=axs.ravel().tolist(), location='left', shrink=0.5)
fig.colorbar(im_full, location='bottom', shrink=0.5)
fig.colorbar(im_tuck, location='bottom', shrink=0.5)
fig.colorbar(im_diff, location='bottom', shrink=0.5)

plt.show()