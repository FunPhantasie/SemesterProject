import numpy as np
import matplotlib.pyplot as plt

def svd(A):
    try:
        return np.linalg.svd(A, full_matrices = False)
    except: #LinAlgError
        try:
            print("SVD failed")
            return np.linalg.svd(A + 1e-12*np.linalg.norm(A, 1), full_matrices = False)
        except:
            print("SVD failed twice")
            return np.linalg.svd(A + 1e-8*np.linalg.norm(A, 1), full_matrices = False)

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

class htuck2d:

    def __init__(self, A = None, eps = 1e-6):

        if A is None:
            self.core = 0
            self.u = [0, 0]
            self.n = [0, 0]
            self.r = [0, 0]
            return
        
        N1, N2 = A.shape

        # A --> U1, U2, G12

        U1, V1, r1 = svd_trunc(A, eps)
        U2, V2, r2 = svd_trunc(np.transpose(A), eps)

        B12 = np.tensordot(A, U2, (1, 0))
        B12 = np.tensordot(B12, U1, (0, 0))
        B12 = B12.transpose([1,0])

        self.n = (N1, N2)
        self.r = [r1, r2]
        self.U1 = U1
        self.U2 = U2
        self.B12 = B12

    def size(self):
        size = self.B12.size + self.U1.size + self.U2.size
        sizefull = self.n[0] * self.n[1]
        return size, sizefull, size / sizefull
    
    def __repr__(self):
        
        res = "This is a 2d tensor in the Htucker format with \n"

        print("n = ", self.n)
        print("r = ", self.r)

        return res
    
    def full(self):

        N1, N2 = self.n

        # U1, U2, G12 --> A
        U1 = self.U1
        U2 = self.U2
        B12 = self.B12

        U12 = np.tensordot(B12, U2,(1,1))
        U12 = np.tensordot(U12, U1,(0,1))
        U12 = U12.transpose([1,0])
        
        A = np.reshape(U12, (N1, N2))

        return A
    
def mexhat(x, lam=1):
    c = 2./np.sqrt(3*np.abs(lam)**(1))*np.pi**0.25
    return c*(1-x**2/lam**2)*np.exp(-x**2/(2*lam**2))

def along_the_sheet(x, lam = 1):
    return np.exp(-(x/lam)**4)

def model_instanton2d(x, y, lam=1):
    factor = 5
    xi = factor*np.abs(lam)**(2./4.)  # Boldyrev scaling
    return mexhat(x, lam=lam) * along_the_sheet(y, lam=xi)

Nx = 512
x = np.linspace(-np.pi, np.pi, Nx)
X,Y = np.meshgrid(x, x)

instanton = model_instanton2d(X, Y, lam=0.2)

ht2 = htuck2d(instanton, eps=1)
print(ht2.size())

fig, axs = plt.subplots(1, 3, figsize=(24, 6))

# Plot 1: Model Instanton
cf1 = axs[0].contourf(X, Y, instanton, levels=50, cmap='viridis')
fig.colorbar(cf1, ax=axs[0], label='Amplitude')
axs[0].set_title('Model Instanton')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

# Plot 2: Htucker Full Representation
cf2 = axs[1].contourf(X, Y, ht2.full(), levels=50, cmap='viridis')  # use `cmap=`, not `colors=`
fig.colorbar(cf2, ax=axs[1], label='Amplitude (Htucker Full)')
axs[1].set_title('Htucker Full Representation')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')

# Plot 3: Difference
cf3 = axs[2].contourf(X, Y, instanton - ht2.full(), levels=50, cmap='viridis')
fig.colorbar(cf3, ax=axs[2], label='Amplitude (Difference)')
axs[2].set_title('Difference')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y') 

plt.tight_layout()
plt.show()

