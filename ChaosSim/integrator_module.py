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

    def EulerForward(self,t,y,dt ):

        return y+dt*self.dgl(y,t)
    def EulerBackward(self,t,y,dt):
        pass
    def RK2(self,t,y,dt): #Heun

        y_step=self.dgl(y,t)

        return y+0.5*dt*((y_step)+self.dgl(y_step,t+dt))
    def RK4(self,t,y,dt):
        pass
