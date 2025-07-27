# Two-Stream PIC Simulation

At the Moment there is only the **Two-Stream instability**.  To run use the main file **TwoStream**.py.
## 🔧 Setup

 You set the **simulation parameters** directly in the main file, such as:

- `border` (simulation domain limits)
- `dx` (grid spacing)
- `dt`, `t_end` (time step and simulation end time)
- `NPpCell` (particles per cell)

In the same file, you also choose the **presentation mode** of the analysis:
- Either as a continuous animation
- Or as a step-by-step flipbook

You will also find the **initialization** of the problem here, including:
- The **velocity distribution** of the particles (e.g., Two-Stream setup)
- The **initial positions** of the particles

## 📁 Structure
- **Simulation**
  - The **explicit** code
  - The **semi-implicit** code 
  - Both are executed using the same initial conditions Set in Main.
- **Analytics**
  - Animator to change Plot Settings
  - RenderManager to set new data   

## ⚙️ Simulation Details

- The magnetic field **B** is initially set to `0`, but is updated during the simulation.
- The normalization of charge is done via:

  ```python
  weight = 1 / (Nx * dx)
