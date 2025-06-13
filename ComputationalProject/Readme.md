## 🔧 Semi-Implicit PIC Solver in Electromagnetic Fields

This project implements a **semi-implicit Particle-in-Cell (PIC) solver** for simulating particle motion in **electric** and **magnetic fields**, inspired by the formulations from **Vu & Brackbill (1982)** and **Lapenta (2006)**.

---

### 📘 Method Overview

The solver uses a semi-implicit time integration scheme that improves stability and allows larger time steps compared to fully explicit methods. The particle motion is coupled self-consistently with the electric and magnetic fields.

---

### 🧩 Core Algorithm Steps

1. **Initialization**
   - Set simulation parameters (`dt`, `Lx`, `Ly`, `Nx`, `Ny`, `Np`, etc.).
   - Initialize particle positions and velocities.
   - Define and initialize fields (`E`, `B`) on a grid.

2. **Field Solver**
   - Compute the **electric potential** using Poisson's equation from the charge density `ρ`.
   - Compute the **electric field** via:
     ```
     \begin{equation}
       \rho = - \nabla^2 \phi
       
     \end{equation}
     ```
   - Magnetic field `B` can be externally defined or evolved (e.g., via Maxwell's equations).
     ```
     \begin{enumerate}
          \item \textbf{Field Solver (Poisson’s Equation):}
              \begin{equation}
                  \frac{\phi^{n+1}_{i+1} - 2\phi^{n+1}_i + \phi^{n+1}_{i-1}}{\Delta x^2} = -\rho^{n+1}_i
              \end{equation}
    
          \item \textbf{Electric Field Calculation:}
              \begin{equation}
                 E^{n+1}_i = -\frac{\phi^{n+1}_{i+1} - \phi^{n+1}_{i-1}}{2\Delta x}
              \end{equation}
    
          \item \textbf{Field to Particle Interpolation:}
              \begin{equation}
                   E_p = \sum_i E_i W(x_i - x_p)
              \end{equation}
    
          \item \textbf{Particle Mover (Position Update):}
              \begin{equation}
                  \frac{x^{n+1}_p - x^n_p}{\Delta t} = v^{n+1/2}_p
              \end{equation}
    
          \item \textbf{Particle Mover (Velocity Update):}
              \begin{equation}
                   \frac{v^{n+1/2}_p - v^{n-1/2}_p}{\Delta t} = \frac{q_p}{m_p} E^n_p
              \end{equation}
    
          \item \textbf{Moment Gathering (Charge Density):}
              \begin{equation}
                    \rho_i = \sum_p q_p W(x_i - x_p)
              \end{equation}
     \end{enumerate}
     
     
     
     
     
     
     
     
     
     ```





3. **Particle Mover (Semi-Implicit)**
   - Update particle velocities with a semi-implicit scheme:
     \[
     \frac{\vec{v}^{n+1} - \vec{v}^n}{\Delta t} = \frac{q}{m}\left( \vec{E}^{n+\theta} + \frac{\vec{v}^{n+1} + \vec{v}^n}{2} \times \vec{B} \right)
     \]
     - Typically, use `θ = 0.5` for midpoint rule.

   - Update particle positions:
     \[
     \vec{x}^{n+1} = \vec{x}^n + \vec{v}^{n+1} \Delta t
     \]

4. **Charge Deposition**
   - Deposit particle charge onto the grid using a scheme like **Cloud-in-Cell (CIC)**.
   - Compute the charge density `ρ` on the grid.

5. **Field Interpolation**
   - Interpolate electric and magnetic fields from the grid back to particle positions.

6. **Loop**
   - Repeat:
     - Deposit charge
     - Solve fields
     - Interpolate fields
     - Move particles
     - Apply boundary conditions

---

### 📚 References
##### Chatgpt
- Vu, H. X., & Brackbill, J. U. (1982). A numerical solution method for the two-dimensional magnetohydrodynamic equations. *Journal of Computational Physics*.
- Lapenta, G. (2006). Particle simulations of space weather. *Journal of Computational Physics*.
##### Own Literature
- Ott, Tobias & Pfeiffer, Marcel. (2023). PIC schemes for multi-scale plasma simulations. 10.13009/EUCASS2023-770. 
---
