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
     E = -∇φ
     ```
   - Magnetic field `B` can be externally defined or evolved (e.g., via Maxwell's equations).

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

- Vu, H. X., & Brackbill, J. U. (1982). A numerical solution method for the two-dimensional magnetohydrodynamic equations. *Journal of Computational Physics*.
- Lapenta, G. (2006). Particle simulations of space weather. *Journal of Computational Physics*.

---
