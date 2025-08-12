import jax
import jax.numpy as jnp
from jaxopt import BacktrackingLineSearch
from jaxopt import LBFGS, GradientDescent, ScipyMinimize
import copy
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, T=1.0, num_steps=100, a=-3.0, mu=1, F=1):
        self.T = T
        self.num_steps = num_steps
        self.dt = T / (num_steps - 1)
        self.t = jnp.linspace(0, T, num_steps)
        self.a = a  # terminal constraint target value
        self.mu = mu
        self.F = F
        self.chi = jnp.array([1,1./4.])
        self.eps = 1

    def set_a(self, a):
        self.a = a

    def set_mu(self, mu):
        self.mu = mu

    def set_F(self, F):
        self.F = F

    def solve_forward(self, p):
        eps = 1
        u = [jnp.array([0.0, 0.0])] # start from zero at negative starting time
        for i in range(self.num_steps - 1):
            u1, u2 = u[-1]
            p1, p2 = p[i]
            du1dt = -u1   - u1*u2 + self.eps*self.chi[0]*p1
            du2dt = -4*u2 + u1**2 + self.eps*self.chi[1]*p2
            u_next = u[-1] + self.dt * jnp.array([du1dt, du2dt]) # Euler step
            u.append(u_next)
        return jnp.stack(u)

    def solve_adjoint(self, p, u):
        # starting condition at t=0
        gradObsT = (self.F - self.mu*(self.observable(u))) * jnp.array([1.0, 2.0])
        z = [gradObsT]

        for i in reversed(range(self.num_steps - 1)):
            u1, u2 = u[i]
            z1, z2 = z[0]

            dz1dt = z1 + u2*z1 - 2*u1*z2
            dz2dt = 4*z2 + u1*z1

            z1_new = z1 - self.dt*dz1dt
            z2_new = z2 - self.dt*dz2dt

            z.insert(0, jnp.array([z1_new, z2_new]))

        return jnp.stack(z)
    
    def observable(self, u):
        u0 = u[-1]
        return -(u0[0] + 2 * u0[1]) - self.a
   
    def actionDensity(self, p):
        p1 = p[:,0]
        p2 = p[:,1]
        return self.eps/2.*(p1*self.chi[0]*p1 + p2*self.chi[1]*p2)

    def compute_L_A(self, p, u): # augmented Lagrangian
        # action S[p]
        S = jnp.sum(self.actionDensity(p)[:1])*self.dt
        # linear and quadratic constraint
        return S, S + self.F*self.observable(u) + self.mu/2.*self.observable(u)**2

    def compute_grad(self, p, z):
        return p-z

    def objective_and_grad(self, p_flat):
        p = p_flat.reshape((self.num_steps, 2))
        u = self.solve_forward(p)
        z = self.solve_adjoint(p, u)
        S, loss = self.compute_L_A(p, u)
        grad = self.compute_grad(p, z)
        return loss, grad
    
def run_CS(sim: Simulation, p=None):
    if p is None:
        p = jnp.zeros((sim.num_steps, 2))

    A = 0.
    epsilon = 1.e-10
    sigma = 1.
    delta_A = epsilon+1
    while(jnp.abs(delta_A) > epsilon):
        u = sim.solve_forward(p)
        z = sim.solve_adjoint(p, u)
        S, loss = sim.compute_L_A(p, u)
        grad = sim.compute_grad(p, z)
        print("observable =",sim.observable(u), "L_A = ", S, "sigma = ", sigma)
        A_bckp = A
        A = sim.observable(u)
        delta_A_bckp = delta_A
        delta_A = (A_bckp - A)/sigma/A_bckp
        if(delta_A_bckp*delta_A < 0.):
            sigma *= 0.8 #0.95
        p = p - sigma*grad
    return p, u

def run_CS_Armijo(sim: Simulation, p=None):
    if p is None:
        p = jnp.zeros((sim.num_steps, 2))
    
    # Wrap objective so it works with flat parameters
    def loss_fn(p_flat):
        p_reshaped = p_flat.reshape((sim.num_steps, 2))
        u = sim.solve_forward(p_reshaped)
        _, L_A = sim.compute_L_A(p_reshaped, u)
        return L_A
    
    def grad_fn(p_flat):
        p_reshaped = p_flat.reshape((sim.num_steps, 2))
        u = sim.solve_forward(p_reshaped)
        z = sim.solve_adjoint(p_reshaped, u)
        grad = sim.compute_grad(p_reshaped, z)
        return grad #.reshape(-1)
    
    def loss_and_grad(p_flat):
        return loss_fn(p_flat), grad_fn(p_flat)
    
    p_flat = p.reshape(-1)
    maxiter = 100

    for i in range(maxiter):
        L_A = loss_fn(p_flat)
        g = grad_fn(p_flat)
        direction = -g

        # Run line search to get step size
        ls = BacktrackingLineSearch(fun=L_A, condition='armijo')
        result = ls.run(p_flat, direction)
        sigma = result.stepsize

        # Update parameters
        p_flat = p_flat + sigma * direction

        # Monitor
        p_current = p_flat.reshape((sim.num_steps, 2))
        u_current = sim.solve_forward(p_current)
        observable = sim.observable(u_current)
        loss = loss_fn(p_flat)

        print(f"iter {i:03d}: loss = {loss:.6f}, step_size = {sigma:.3e}, observable = {observable:.6f}")

        # Optional convergence check (can refine this)
        if jnp.linalg.norm(g) < 1e-6:
            break

    # Final reshape and forward simulation
    p_final = p_flat.reshape((sim.num_steps, 2))
    u_final = sim.solve_forward(p_final)

    return p_final, u_final

def run_LBFGS(sim: Simulation, p0=None):
    if p0 is None:
        p0 = jnp.zeros((sim.num_steps, 2))
    
    # Wrap objective so it works with flat parameters
    def loss_fn(p_flat):
        p_reshaped = p_flat.reshape((sim.num_steps, 2))
        u = sim.solve_forward(p_reshaped)
        _, L_A = sim.compute_L_A(p_reshaped, u)
        return L_A
    
    def grad_fn(p_flat):
        p_reshaped = p_flat.reshape((sim.num_steps, 2))
        u = sim.solve_forward(p_reshaped)
        z = sim.solve_adjoint(p_reshaped, u)
        grad = sim.compute_grad(p_reshaped, z)
        return grad.reshape(-1)
    
    def loss_and_grad(p_flat):
        return loss_fn(p_flat), grad_fn(p_flat)

    # solver = LBFGS(fun=loss_and_grad, value_and_grad=True, tol=1.e-8)
    solver = ScipyMinimize(method='CG',fun=loss_and_grad, value_and_grad=True)
    result = solver.run(p0.reshape(-1))
    p_opt = result.params.reshape((sim.num_steps, 2))
    u_opt = sim.solve_forward(p_opt)
    return p_opt, u_opt

# Create simulation
sim = Simulation(a=-3.2,mu=0,F=2.75)

# optimiser = 'LBFGS'
# if optimiser == 'CS': # Chernyk-Stepanov
p_CS, u_CS = run_CS(sim)
# elif optimiser == 'CS_Armijo': # Chernyk-Stepanov with Armijo line search
#     p_opt, u_opt = run_CS_Armijo(sim)
# elif optimiser == 'LBFGS':

p_LBFGS, u_LBFGS = run_LBFGS(sim)

plt.plot(sim.actionDensity(p_CS),'p',label='CS')
plt.plot(sim.actionDensity(p_LBFGS),label='LBDGS')
plt.legend()
plt.show()
plt.plot(u_CS[:,0], u_CS[:,1],label='CS')
plt.plot(u_LBFGS[:,0], u_LBFGS[:,1],label='LBDGS')
plt.legend()
plt.show()

sim.observable(u_CS)
sim.observable(u_LBFGS)