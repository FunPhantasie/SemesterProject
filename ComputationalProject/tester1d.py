import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sort import PIC1D
from sort import PIC_Explicit1D
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
mpl.use('TkAgg')
import win32api
import win32con

# Show a yes/no message box
response = win32api.MessageBox(0, "Neurechnen?", "Simulation", win32con.MB_YESNO | win32con.MB_ICONQUESTION)

border=1
gridpoints=128
NPpCell=20
dt=0.05
t_end=2
# Initialize solvers
solver_test = PIC1D(border, gridpoints, NPpCell, dt)
solver_ref = PIC_Explicit1D(border, gridpoints, NPpCell, dt)

# Create 'rendered' folder if it doesn't exist
if not os.path.exists('rendered'):
    os.makedirs('rendered')

# Define file paths for saved data
data_files = {
    'x_test': 'rendered/x_test_history.npy',
    'v_test': 'rendered/v_test_history.npy',
    'x_ref': 'rendered/x_ref_history.npy',
    'v_ref': 'rendered/v_ref_history.npy',
    't': 'rendered/t_history.npy',
    'energy_total_test': 'rendered/energy_total_history_test.npy',
    'energy_kin_test': 'rendered/energy_kin_history_test.npy',
    'energy_total_ref': 'rendered/energy_total_history_ref.npy',
    'energy_kin_ref': 'rendered/energy_kin_history_ref.npy'
}

# Check if all data files exist
data_exists = all(os.path.exists(file) for file in data_files.values())

# Handle the response
if response == win32con.IDYES:
    data_exists = False
else:

    print("User clicked No")

# Initialize data arrays
total_steps = int(t_end / solver_test.dt)

if data_exists:
    # Load data from files
    x_test_history = np.load(data_files['x_test'])
    v_test_history = np.load(data_files['v_test'])
    x_ref_history = np.load(data_files['x_ref'])
    v_ref_history = np.load(data_files['v_ref'])
    t_history = np.load(data_files['t'])
    energy_total_history_ref = np.load(data_files['energy_total_ref'])
    energy_kin_history_ref = np.load(data_files['energy_kin_ref'])
    energy_total_history_test = np.load(data_files['energy_total_test'])
    energy_kin_history_test = np.load(data_files['energy_kin_test'])
else:
    # Run simulation and store data
    x_test_history = []
    v_test_history = []
    x_ref_history = []
    v_ref_history = []
    t_history = []
    energy_total_history_ref = []
    energy_kin_history_ref = []
    energy_total_history_test = []
    energy_kin_history_test = []


    """
    Calculation
    
    """
    for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
        solver_test.step()
        solver_ref.step()

        # Store data (copy to avoid reference issues)
        x_test_history.append(solver_test.xp.copy())
        v_test_history.append(solver_test.vp[0].copy())
        x_ref_history.append(solver_ref.xp.copy())
        v_ref_history.append(solver_ref.vp[0].copy())
        t_history.append(solver_test.t)
        energy_total_history_test.append(solver_test.CalcEFieldEnergy())
        energy_kin_history_test.append(solver_test.CalcKinEnergery())
        energy_total_history_ref.append(solver_ref.CalcEFieldEnergy())
        energy_kin_history_ref.append(solver_ref.CalcKinEnergery())

    """
    Saving
    
    """


    # Convert lists to NumPy arrays and save
    x_test_history = np.array(x_test_history)
    v_test_history = np.array(v_test_history)
    x_ref_history = np.array(x_ref_history)
    v_ref_history = np.array(v_ref_history)
    t_history = np.array(t_history)
    energy_total_history_test = np.array(energy_total_history_test)
    energy_kin_history_test = np.array(energy_kin_history_test)
    energy_total_history_ref = np.array(energy_total_history_ref)
    energy_kin_history_ref = np.array(energy_kin_history_ref)

    # Save data to files
    np.save(data_files['x_test'], x_test_history)
    np.save(data_files['v_test'], v_test_history)
    np.save(data_files['x_ref'], x_ref_history)
    np.save(data_files['v_ref'], v_ref_history)
    np.save(data_files['t'], t_history)
    np.save(data_files['energy_total_test'], energy_total_history_test)
    np.save(data_files['energy_kin_test'], energy_kin_history_test)
    np.save(data_files['energy_total_ref'], energy_total_history_ref)
    np.save(data_files['energy_kin_ref'], energy_kin_history_ref)



"""
Plott Settings
"""
# Setup plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

# Plot for solver_test
ax1.set_xlim(0, solver_test.Lx)
ax1.set_ylim(-3, 3)
sc1_a, = ax1.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
sc1_b, = ax1.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, va='top')
ax1.set_title("Your PIC1D")
ax1.set_xlabel("x")
ax1.set_ylabel("v")
ax1.legend()

# Plot for solver_ref
ax2.set_xlim(0, solver_ref.Lx)
ax2.set_ylim(-3, 3)
sc2_a, = ax2.plot([], [], 'o', markersize=0.5, color='b', label='Species A')
sc2_b, = ax2.plot([], [], 'o', markersize=0.5, color='r', label='Species B')
text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top')
ax2.set_title("Reference PIC")
ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.legend()

# Energy plot setup
ax3.set_xlim(0, 5)

ax3.set_title("Energy vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Energy")

a=energy_total_history_test
b=energy_kin_history_test

c=energy_total_history_ref
d=energy_kin_history_ref
# Plot fixed energy curves
"""ax3.plot(t_history, (a - a.min()) / (a.max() - a.min()), '-', label='Total Energy')
ax3.plot(t_history, (b - b.min()) / (b.max() - b.min()), '-', label='Kinetic Energy')
ax3.plot(t_history, (c - c.min()) / (c.max() - c.min()), '-', label='Total Energy Ref')
ax3.plot(t_history, (d - d.min()) / (d.max() - d.min()), '-', label='Kinetic Energy Ref')
"""

#ax3.set_yscale('log')
ax3.plot(t_history, a, '-', label='E Field Energy')
ax3.plot(t_history[1:], b[1:], '-', label='Kinetic Energy')
ax3.plot(t_history[1:], c[1:], '-', label='E Field Energy Ref')
ax3.plot(t_history[1:], d[1:], '-', label='Kinetic Energy Ref')
#Set y-axis limits for ax3 (replace ymin, ymax with desired values)
#ax3.set_ylim(-1.0e5, 1.0e5)  # Example: set y-axis limits from -1 to 1


ax3.legend()


"""
print("Field ENergy")
print(energy_total_history_ref)
print("Kin ENergy")
print(energy_kin_history_ref)
"""
def init():
    # Initialize particle plots as empty
    sc1_a.set_data([], [])
    sc1_b.set_data([], [])
    text1.set_text('')

    sc2_a.set_data([], [])
    sc2_b.set_data([], [])
    text2.set_text('')

    return sc1_a, sc1_b, sc2_a, sc2_b, text1, text2

def update(frame):
    Np = solver_test.Np
    Np2 = solver_ref.Np

    x1 = x_test_history[frame]
    v1 = v_test_history[frame]
    x2 = x_ref_history[frame]
    v2 = v_ref_history[frame]
    t = t_history[frame]

    sc1_a.set_data(x1[:Np // 2], v1[:Np // 2])
    sc1_b.set_data(x1[Np // 2:], v1[Np // 2:])
    text1.set_text(f"time = {t:.2f}")

    sc2_a.set_data(x2[:Np2 // 2], v2[:Np2 // 2])
    sc2_b.set_data(x2[Np2 // 2:], v2[Np2 // 2:])
    text2.set_text(f"time = {t:.2f}")

    return sc1_a, sc1_b, sc2_a, sc2_b, text1, text2

anim = animation.FuncAnimation(fig, update, init_func=init, frames=total_steps,
                              interval=300, blit=True)

plt.tight_layout()
plt.show()

"""
[1.40127572e-001 2.99464250e+000 3.22682821e+001 2.78357741e+002
 2.39720799e+003 4.05052541e+004 1.77649590e+007 2.41004470e+010
 3.36319139e+013 4.75233153e+016 6.77671758e+019 9.73425380e+022
 1.40698277e+026 2.04490899e+029 2.98706318e+032 4.38367687e+035
 6.46140749e+038 9.56317187e+041 1.42091814e+045 2.11906631e+048
 3.17140644e+051 4.76233702e+054 7.17434825e+057 1.08411529e+061
 1.64300339e+064 2.49696799e+067 3.80489357e+070 5.81262523e+073
 8.90121473e+076 1.36622369e+080 2.10155081e+083 3.23932882e+086
 5.00288249e+089 7.74088057e+092 1.19983501e+096 1.86282376e+099
 2.89668151e+102 4.51096925e+105 7.03464611e+108 1.09845630e+112]
"""
"""
[1.40127572e-001 2.99464250e+000 3.22682821e+001 2.78357741e+002
 2.39720799e+003 4.05052541e+004 1.77649590e+007 2.41004470e+010
 3.36319139e+013 4.75233153e+016 6.77671758e+019 9.73425380e+022
 1.40698277e+026 2.04490899e+029 2.98706318e+032 4.38367687e+035
 6.46140749e+038 9.56317187e+041 1.42091814e+045 2.11906631e+048
 3.17140644e+051 4.76233702e+054 7.17434825e+057 1.08411529e+061
 1.64300339e+064 2.49696799e+067 3.80489357e+070 5.81262523e+073
 8.90121473e+076 1.36622369e+080 2.10155081e+083 3.23932882e+086
 5.00288249e+089 7.74088057e+092 1.19983501e+096 1.86282376e+099
 2.89668151e+102 4.51096925e+105 7.03464611e+108 1.09845630e+112
 Explodiert weiter
 1.71734699e+115 2.68804405e+118 4.21199431e+121 6.60668943e+124
 1.03728477e+128 1.63006503e+131 2.56378157e+134 4.03556230e+137
 6.35701463e+140 1.00209520e+144 1.58071336e+147 2.49498874e+150
 3.94039226e+153 6.22659865e+156 9.84439396e+159 1.55718741e+163
 2.46430547e+166 3.90156419e+169 6.17963896e+172 9.79169491e+175
 1.55207975e+179 2.46106548e+182 3.90370918e+185 6.19397723e+188
 9.83088981e+191 1.56077682e+195 2.47860687e+198 3.93720356e+201
 6.25570621e+204 9.94187505e+207 1.58037183e+211 2.51272589e+214
 3.99596720e+217 6.35603146e+220 1.01119305e+224 1.60902517e+227
 2.56076263e+230 4.07615536e+233 6.48939798e+236 1.03330347e+240]
"""
"""
[1.40127572e-001 2.99518522e+000 3.23099717e+001 2.79408522e+002
 2.41926648e+003 3.21093835e+004 1.69937392e+005 1.05783552e+006
 7.41340026e+006 5.29122809e+007 4.02282773e+008 8.01250873e+009
 5.16665535e+012 3.73187026e+018 1.21878282e+030 1.70538178e+056
 5.11362331e+105 4.59772469e+204 4.13795222e+205 3.72415700e+206
 3.35174130e+207 3.01656717e+208 2.71491045e+209 2.44341941e+210
 2.19907747e+211 1.97916972e+212 1.78125275e+213 1.60312747e+214
 1.44281473e+215 1.29853325e+216 1.16867993e+217 1.05181194e+218
 9.46630742e+218 8.51967668e+219 7.66770901e+220 6.90093811e+221
 6.21084430e+222 5.58975987e+223 5.03078388e+224 4.52770549e+225
 4.07493494e+226 3.66744145e+227 3.30069730e+228 2.97062757e+229
 2.67356482e+230 2.40620834e+231 2.16558750e+232 1.94902875e+233
 1.75412588e+234 1.57871329e+235 1.42084196e+236 1.27875776e+237
 1.15088199e+238 1.03579379e+239 9.32214410e+239 8.38992969e+240
 7.55093672e+241 6.79584305e+242 6.11625874e+243 5.50463287e+244
 4.95416958e+245 4.45875262e+246 4.01287736e+247 3.61158963e+248
 3.25043066e+249 2.92538760e+250 2.63284884e+251 2.36956395e+252
 2.13260756e+253 1.91934680e+254 1.72741212e+255 1.55467091e+256
 1.39920382e+257 1.25928344e+258 1.13335509e+259 1.02001958e+260
 9.18017625e+260 8.26215863e+261 7.43594277e+262 6.69234849e+263]
"""