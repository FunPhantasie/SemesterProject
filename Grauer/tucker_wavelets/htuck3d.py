import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def svd(A):
    return np.linalg.svd(A, full_matrices = False)

def svd_trunc(A, eps = 1e-14):

    u, s, v = svd(A)

    N1, N2 = A.shape

    eps_svd = eps*s[0]/np.sqrt(3)
    r = min(N1, N2)
    for i in range(min(N1, N2)):
        if s[i] <= eps_svd:
            r = i
            break
        #print s/s[0]
    u = u[:,:r].copy()
    v = v[:r,:].copy()
    s = s[:r].copy()

    return u, H(v), r

def H(A):
    return np.transpose(A)

class htuck3d:

    def __init__(self, A = None, eps = 1e-14):

        if A is None:
            self.core = 0
            self.u = [0, 0, 0]
            self.n = [0, 0, 0]
            self.r = [0, 0, 0]
            return

# notation as in the slides from the lecture
        n1, n2, n3 = A.shape

        M12 = np.reshape(A,(n1*n2,-1))
        M3 = np.transpose(M12)

        U12, _, r12 = svd_trunc(M12, eps)                       # shape (n1*n2, r12)
        U3,  _, r3  = svd_trunc(M3,  eps)                       # shape (n3, r3)

        B123 = np.tensordot(U12, M12, axes=(0,0))               # shape (r12, n3)
        B123 = np.tensordot(B123, U3, axes=(1,0))               # shape (r12, r3)

        # U12 --> U1, U2, G12
        U12 = U12.reshape(n1,-1)                                # shape (n1, n2*r12)
        U1, _, r1 = svd_trunc(U12, eps)                         # shape (n1, r1)

        tmp = np.reshape(U12,(n1, n2, -1))
        tmp = np.transpose(tmp, axes=(1, 0, 2))                 # shape (n2, n1, r1)
        tmp = tmp.reshape(n2, -1)                               # shape (n2, n1*r1)
        U2, _, r2 = svd_trunc(tmp, eps)                         # shape (n2, r2)

        B12 = np.reshape(U12,(n1,n2,-1))                        # shape: (n1, n2, r12)
        B12 = np.tensordot(U2, B12, axes=(0, 1))                # shape: (r2, n1, r12)
        B12 = np.tensordot(U1, B12, axes=(0, 1))                # shape: (r1, r2, r12)
        B12 = B12.reshape(r1*r2,-1)                             # shape: (r1*r2, r12)


        self.n = (n1,n2,n3)
        self.r = [r1, r2, r3, r12]
        self.U1 = U1
        self.U2 = U2
        self.U3 = U3
        self.U12 = U12
        self.B12 = B12
        self.B123 = B123

    def size(self):
        size = self.B12.size+self.B123.size \
              +self.U1.size+self.U2.size+self.U3.size
        sizefull = self.n[0]*self.n[1]*self.n[2]
        return size, sizefull, size/sizefull

    def __repr__(self):
        res = "This is a 3D tensor in the HTucker format with \n"
        print("n = ", self.n)
        print("r = ", self.r)
        return res

    def full(self):
        n1, n2, n3 = self.n
        [r1, r2, r3, r12] = self.r

        # U1, U2, B12 --> U12
        B12 = self.B12
        U1 = self.U1
        U2 = self.U2

        U12 = np.reshape(B12, (r1, r2, -1))                     # shape: (r1, r2, r12)
        U12 = np.tensordot(U2, U12, axes = (1,1))               # shape: (n2, r1, r12)
        U12 = np.tensordot(U1, U12, axes = (1,1))               # shape: (n1, n2, r12)

        # U12, U3, B123 --> U123

        B123 = self.B123
        U3 = self.U3
        # U12 = U12.reshape(n1*n2,-1)

        U123 = np.tensordot(B123,U3,(1,1))
        U123 = np.tensordot(U12,U123,(2,0))
        # A = U123.transpose()

        return U123

def mexhat(x, lam=1):
    c = 2./np.sqrt(3*np.abs(lam)**(1))*np.pi**0.25
    return c*(1-x**2/lam**2)*np.exp(-x**2/(2*lam**2))

def along_the_sheet(x, lam = 1):
    return np.exp(-(x/lam)**4)

def model_instanton3d(x, y, z, lam=1):
    factor = 5
    xi = factor*np.abs(lam)**(3./4.)
    l  = factor*np.abs(lam)**(2./4.)  # Boldyrev scaling
    return mexhat(x, lam=lam) * along_the_sheet(y, lam=xi) *  along_the_sheet(z, lam=l)

Nx = 256; Ny = 128; Nz = 64
x = np.linspace(-np.pi, np.pi, Nx)
y = np.linspace(-np.pi, np.pi, Ny)
z = np.linspace(-np.pi, np.pi, Nz)
X,Y,Z = np.meshgrid(x, y, z, indexing='ij')

instanton = model_instanton3d(X, Y, Z, lam=0.2)
instanton += model_instanton3d(X-Y, X+Y, Z, lam=0.5)
instanton += model_instanton3d(Y, Z, X, lam=0.5)

grid = pv.wrap(instanton)
grid = pv.wrap(ht3.full())
iso = grid.contour(isosurfaces=[1])
iso.plot()
ht3 = htuck3d(instanton, eps=1e-6)
ht3.size()
np.max(np.abs(instanton-ht3.full()))

import pywt
signal = ht3.U1[:,0]

# Wavelet transform
wavelet = 'bior4.4'
coeffs = pywt.wavedec(signal, wavelet)
coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)

# Define epsilon threshold
epsilon = 0.01

# Hard thresholding
coeff_array[np.abs(coeff_array) < epsilon] = 0

# Convert back to coeffs format
truncated_coeffs = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format='wavedec')
# truncated_coeff_array, truncated_coeff_slices = pywt.coeffs_to_array(truncated_coeffs)


# Reconstruct the signal
reconstructed = pywt.waverec(truncated_coeffs, wavelet)

# Plot
plt.plot(signal, label='Original')
plt.plot(reconstructed, label=f'Thresholded (ε={epsilon})')
plt.legend()
plt.show()

#-------------------------------------------------------------------------
def shannon_entropy(x):
    x = np.asarray(x)
    p = np.abs(x)**2
    p = p / np.sum(p)
    return -np.sum(p * np.log2(p + 1e-12))  # small epsilon to avoid log(0)

def sizewp(nodes):
    size = 0
    for node in nodes:
        size += node.data.size
    return size

# Wavelet packet decomposition
maxlevel = 4
wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=maxlevel)

# Get all terminal nodes at max level
nodes = wp.get_level(4, order='natural')

# Select nodes with low entropy
threshold = 1.3
best_nodes = [node for node in nodes if shannon_entropy(node.data) < threshold]
# eps_wp = 0.1
# best_nodes = [node for node in nodes if np.any(np.abs(node.data) >= eps_wp)]

# Create a new packet tree for reconstruction
wp2 = pywt.WaveletPacket(data=None, wavelet=wavelet, maxlevel=maxlevel)

# Insert selected nodes
for node in best_nodes:
    wp2[node.path] = node.data

# Reconstruct compressed signal
reconstructed = wp2.reconstruct(update=True)

compressed_size = sizewp(best_nodes)
original_size = sizewp(nodes)
compression_ratio = compressed_size / original_size

print(f"Original size:   {original_size}")
print(f"Compressed size: {compressed_size}")
print(f"Compression ratio: {compression_ratio:.6f}")

reconstructed = wp2.reconstruct(update=True)

plt.plot(signal, label='Original')
plt.plot(reconstructed, label=fr'Thresholded ($\epsilon={eps_wp:.2e}$)')
plt.legend()
plt.show()