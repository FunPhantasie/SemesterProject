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

    def EulerForward(self, t,*x):

        return x[0]+t*self.dgl(*x)
    def EulerBackward(self, x, t):
        pass
    def RK2(self, x, t):
        pass
    def RK4(self, x, t):
        pass
    def Propagator(self, x, t):
        pass