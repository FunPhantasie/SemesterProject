import numpy as np

class Integrator():
    def __init__(self,method,dgl_function):
        if method == "EulerForward":
            self.step = self.EulerForward
        elif method == "EulerBackward":
            self.step = self.EulerBackward
        elif method == "RK4":
            self.step = self.RK4
        elif method == "Propagator":
            self.step = self.Propagator
        elif method.step == "RK2":
            self.step =self.RK2()
        else:
            raise ValueError("Unknown mode")
        self.dgl = dgl_function

    def EulerForward(self, dt,*x):

        return x[0]+dt*self.dgl(*x)
    def EulerBackward(self, dt,*x):
        pass
    def RK2(self, dt,*x): #Heun
        u_hat, u2_hat, f_hat=x
        u_step=self.dgl(u_hat, u2_hat, f_hat)

        return x[0]+0.5*dt*(self.dgl(u_hat, u2_hat, f_hat)+self.dgl(u_hat, u2_hat, f_hat))
    def RK4(self, x, t):
        pass
    def Propagator(self, x, t):
        pass