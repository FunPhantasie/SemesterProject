import numpy as np
import matplotlib.pyplot as plt
# import pyvista as pv
import matplotlib as mpl
mpl.use('TkAgg')
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
        B12 = B12.reshape(r1*r2,-1)                              # shape: (r1*r2, r12)


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

# grid = pv.wrap(instanton)
# grid = pv.wrap(ht3.full())
# iso = grid.contour(isosurfaces=[1])
# iso.plot()
ht3 = htuck3d(instanton, eps=1e-6)
ht3.size()
np.max(np.abs(instanton-ht3.full()))
plt.plot(ht3.U1[:,1])
plt.show()