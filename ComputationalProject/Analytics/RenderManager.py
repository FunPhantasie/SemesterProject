import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
mpl.use('TkAgg')
import win32api
import win32con

def CallItRenderer(solver_test, total_steps, path,step):
    # Show a yes/no message box
    last_part = os.path.basename(path)
    response = win32api.MessageBox(0, "Neurechnen?", "Simulation von "+last_part, win32con.MB_YESNO | win32con.MB_ICONQUESTION)

    # Create output folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Define species names
    species_names = [s["name"] for s in solver_test.species]

    # Define file paths for saved data
    data_files = {}
    for name in species_names:
        data_files[f'x_{name}'] = os.path.join(path, f'x_history_{name}.npy')
        data_files[f'v_{name}'] = os.path.join(path, f'v_history_{name}.npy')
        data_files[f'rho_{name}'] = os.path.join(path, f'rho_history_{name}.npy')
    data_files['E'] = os.path.join(path, 'E_history.npy')
    data_files['B'] = os.path.join(path, 'B_history.npy')
    data_files['t'] = os.path.join(path, 't_history.npy')
    data_files['electric_energy'] = os.path.join(path, 'electric_energy_history.npy')
    data_files['energy_kin'] = os.path.join(path, 'energy_kin_history.npy')

    # Check if all data files exist
    data_exists = all(os.path.exists(file) for file in data_files.values())

    if response == win32con.IDYES:
        data_exists = False
    else:
        print("User clicked No")

    # Daten laden oder neu berechnen
    if data_exists:
        data = {key: np.load(file) for key, file in data_files.items()}
    else:
        data = {key: [] for key in data_files}
        for s in solver_test.species:
            name = s["name"]
            data[f'x_{name}'].append(s["xp"].copy())
            data[f'v_{name}'].append(s["vp"].copy())
            data[f'rho_{name}'].append(s["rho"].copy())

        data['E'].append(solver_test.E.copy())
        data['B'].append(solver_test.B.copy())
        data['t'].append(solver_test.t)
        data['electric_energy'].append(solver_test.CalcEFieldEnergy())
        data['energy_kin'].append(solver_test.CalcKinEnergery())

        for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
            if step:
                solver_test.step()

            for s in solver_test.species:
                name = s["name"]
                data[f'x_{name}'].append(s["xp"].copy())
                data[f'v_{name}'].append(s["vp"].copy())
                data[f'rho_{name}'].append(s["rho"].copy())

            data['E'].append(solver_test.E.copy())
            data['B'].append(solver_test.B.copy())
            data['t'].append(solver_test.t)
            data['electric_energy'].append(solver_test.CalcEFieldEnergy())
            data['energy_kin'].append(solver_test.CalcKinEnergery())

        # Save all data
        for key, values in data.items():
            data[key] = np.array(values)
            np.save(data_files[key], data[key])
        print("Simulation abgeschlossen und Daten gespeichert.")

    return data
    return tuple(data[k] for k in sorted(data.keys()))
