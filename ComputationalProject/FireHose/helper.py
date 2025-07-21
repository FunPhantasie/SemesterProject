import numpy as np
class MathTools():
    def __init__(self,dimension,steps):
        self.steps=steps
        if dimension==1:
            self.gradient=self.gradient1d
            self.cross = self.cross1d
            self.dot=self.dot
            self.laplace_vector = self.laplace_vector
            self.divergence=self.divergence1d
            self.curl=self.curl1d

            self.dx=steps
            self.dk = [steps] #For laplace
        elif dimension==3:
            self.gradient=self.gradient3d
            self.cross=self.cross3d
            self.laplace_vector=self.laplace_vector3d
            self.divergence = self.divergence3d
            self.curl=self.curl3d

            self.dx,self.dy,self.dz=steps
            self.dk=list[steps]
        else:
            raise NotImplementedError("This Dimension is invalid:"+str(dimension) )
    """Math Operations"""

    def laplace_vector3d(self, A):  # [3,..]
        lap_A = np.zeros_like(A)
        dk = self.dk

        for index, a_k in enumerate(A):
            print(index)
            rolledFor = np.roll(a_k, shift=-1, axis=index)  # Shift along x-axis
            rolledBack = np.roll(a_k, shift=1, axis=index)  # Shift along y-axis
            lap_A[index] = (rolledFor + rolledBack - 2 * a_k) / (dk[index] ** 2)
        return lap_A

    def laplace_vector(self, A):

        lap_A = np.zeros_like(A)
        dk = self.dk[0]

        for index in range(3):

            rolledFor = np.roll(A[index], shift=-1)  # Shift along x-axis
            rolledBack = np.roll(A[index], shift=1)  # Shift along y-axis
            lap_A[index] = (rolledFor + rolledBack - 2 * A[index]) / (dk ** 2)
        return lap_A


    def cross3d(self, A, B):
        print("Cross")
        print(np.shape(A))
        print(np.shape(B))
        U = np.zeros_like(A)
        U[0, ...] = A[1, ...] * B[2, ...] - A[2, ...] * B[1, ...]
        U[1, ...] = A[2, ...] * B[0, ...] - A[0, ...] * B[2, ...]
        U[2, ...] = A[0, ...] * B[1, ...] - A[1, ...] * B[0, ...]
        """
        Or Var:

        result = np.cross(a.T, b.T)
        3 packs of vec  mal particles 
        """
        return U
    def cross1d(self, A, B):
        #print("Cross")
        #print(np.shape(A))
        #print(np.shape(B))
        U = np.zeros_like(A)
        U[0, ...] = A[1, ...] * B[2, ...] - A[2, ...] * B[1, ...]
        U[1, ...] = A[2, ...] * B[0, ...] - A[0, ...] * B[2, ...]
        U[2, ...] = A[0, ...] * B[1, ...] - A[1, ...] * B[0, ...]

        #Or Var:


        #3 packs of vec  mal particles

        #result = np.cross(A.T, B.T)
        return U
    def dot(self, A, B):
        """
        U = np.zeros_like(A)
        U[0, ...] = A[0, ...] * B[0, ...]
        U[1, ...] = A[1, ...] * B[1, ...]
        U[2, ...] = A[2, ...] * B[2, ...]

        Or Var:

        result = np.sum(a * b, axis=0)  # Shape (self.Np,)
        doesnt work
        """
        return np.sum(A * B, axis=0)

    # Finite difference functions
    # For Scalars Return Vector
    def gradient3d(self, f):
        # Compute shifted arrays using np.roll for finite differences

        grad = np.zeros([3, *f.shape])
        rolledx = np.roll(f, shift=-2, axis=0)  # Shift along x-axis
        rolledy = np.roll(f, shift=-2, axis=1)  # Shift along y-axis
        rolledz = np.roll(f, shift=-2, axis=2)  # Shift along z-axis

        grad[0] = (rolledx - f) / (2 * self.dx)
        grad[1] = (rolledy - f) / (2 * self.dy)
        grad[2] = (rolledz - f) / (2 * self.dz)
        return grad

        # For Vector Fields
    def gradient1d1(self, f):

        # Compute shifted arrays using np.roll for finite differences
        grad = np.zeros([3, *f.shape])
        rolledx = np.roll(f, shift=-2, axis=0)  # Shift along x-axis


        grad[0] = (rolledx - f) / (2 * self.dx)

        return grad

    def gradient1d(self, f):
        grad = np.zeros([3, *f.shape])
        rolled_forward = np.roll(f, -1, axis=0)
        rolled_backward = np.roll(f, 1, axis=0)
        grad[0] = (rolled_forward - rolled_backward) / (2 * self.dx)
        return grad

         # For Vector Fields
    def divergence3d(self, A):
        div_A = np.zeros_like(A[0])  # At B points (reduced size)
        rolledx = np.roll(A, shift=-1, axis=1)[0]  # Shift along x-axis
        rolledy = np.roll(A, shift=-1, axis=2)[1]  # Shift along y-axis
        rolledz = np.roll(A, shift=-1, axis=3)[2]  # Shift along z-axis

        div_A[:, :, :] = (rolledx - A[0]) / self.dx + \
                         (rolledy - A[1]) / self.dy + \
                         (rolledz - A[2]) / self.dz
        return div_A
    def divergence1d1(self, A):

        div_A = np.zeros_like(A[0])  # At B points (reduced size)
        rolledx = np.roll(A, shift=-1, axis=1)[0]  # Shift along x-axis


        div_A[:] = (rolledx - A[0]) / self.dx
        return div_A

    def divergence1d(self, A):
        div_A = np.zeros_like(A[0])
        rolled_forward = np.roll(A[0], -1)
        div_A = (rolled_forward - A[0]) / self.dx
        return div_A

    def curl3d(self, A):
        curl = np.zeros_like(A)  # Initialize curl array with same shape as A

        # Compute shifted arrays using np.roll for finite differences
        rolledx = np.roll(A, shift=-1, axis=1)  # Shift along x-axis
        rolledy = np.roll(A, shift=-1, axis=2)  # Shift along y-axis
        rolledz = np.roll(A, shift=-1, axis=3)  # Shift along z-axis

        # Compute finite differences
        # Forward difference: (A(x+dx) - A(x)) / dx

        dAy_dx = (rolledx[1] - A[1]) / self.dx  # ∂Ay/∂x
        dAz_dx = (rolledx[2] - A[2]) / self.dx  # ∂Az/∂x

        dAx_dy = (rolledy[0] - A[0]) / self.dy  # ∂Ax/∂y

        dAz_dy = (rolledy[2] - A[2]) / self.dy  # ∂Az/∂y

        dAx_dz = (rolledz[0] - A[0]) / self.dz  # ∂Ax/∂z
        dAy_dz = (rolledz[1] - A[1]) / self.dz  # ∂Ay/∂z

        # Compute curl components
        # curl_x = ∂Az/∂y - ∂Ay/∂z
        curl[0] = dAz_dy - dAy_dz
        # curl_y = ∂Ax/∂z - ∂Az/∂x
        curl[1] = dAx_dz - dAz_dx
        # curl_z = ∂Ay/∂x - ∂Ax/∂y
        curl[2] = dAy_dx - dAx_dy

        return curl

    def curl1d(self, A):

        curl = np.zeros_like(A)  # Initialize curl array with same shape as A

        # Compute shifted arrays using np.roll for finite differences
        rolledx = np.roll(A, shift=-1, axis=1)  # Shift along x-axis
        #rolledy = np.roll(A, shift=-1, axis=2)  # Shift along y-axis
        #rolledz = np.roll(A, shift=-1, axis=3)  # Shift along z-axis

        # Compute finite differences
        # Forward difference: (A(x+dx) - A(x)) / dx

        dAy_dx = (rolledx[1] - A[1]) / self.dx  # ∂Ay/∂x
        dAz_dx = (rolledx[2] - A[2]) / self.dx  # ∂Az/∂x


        # Compute curl components
        # curl_x = ∂Az/∂y - ∂Ay/∂z
        curl[0] = 0
        # curl_y = ∂Ax/∂z - ∂Az/∂x
        curl[1] =  - dAz_dx
        # curl_z = ∂Ay/∂x - ∂Ax/∂y
        curl[2] = dAy_dx

        return curl


