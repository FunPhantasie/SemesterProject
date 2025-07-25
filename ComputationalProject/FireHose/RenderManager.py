import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
mpl.use('TkAgg')
import win32api
import win32con

def CallItRenderer(solver_test,total_steps):

    # Show a yes/no message box
    response = win32api.MessageBox(0, "Neurechnen?", "Simulation", win32con.MB_YESNO | win32con.MB_ICONQUESTION)

    # Create 'rendered' folder if it doesn't exist
    if not os.path.exists('rendered'):
        os.makedirs('rendered')

    # Define species names
    species_names = [s["name"] for s in solver_test.species]
    # Define file paths for saved data (only for solver_test)
    data_files = {}
    for name in species_names:
        data_files[f'x_{name}'] = f'rendered/x_history_{name}.npy'
        data_files[f'v_{name}'] = f'rendered/v_history_{name}.npy'
        data_files[f'rho_{name}'] = f'rendered/rho_history_{name}.npy'
    # Global fields
    data_files['E'] = 'rendered/E_history.npy'
    data_files['B'] = 'rendered/B_history.npy'
    data_files['t'] = 'rendered/t_history.npy'
    data_files['energy_total'] = 'rendered/energy_total_history.npy'
    data_files['energy_kin'] = 'rendered/energy_kin_history.npy'

    # Check if all data files exist
    data_exists = all(os.path.exists(file) for file in data_files.values())

    # Handle the response
    if response == win32con.IDYES:
        data_exists = False
    else:
        print("User clicked No")

    # Initialize data arrays

    if data_exists:
        data = {}
        for key, path in data_files.items():
            data[key] = np.load(path)


    else:
        # Initialize storage
        data = {}
        for name in species_names:
            data[f'x_{name}'] = []
            data[f'v_{name}'] = []
            data[f'rho_{name}'] = []

        data['E'] = []
        data['B'] = []
        data['t'] = []
        data['energy_total'] = []
        data['energy_kin'] = []



        """
        Calculation
        """
        for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
            solver_test.step()

            for s in solver_test.species:
                name = s["name"]
                data[f'x_{name}'].append(s["xp"].copy())
                data[f'v_{name}'].append(s["vp"].copy())
                data[f'rho_{name}'].append(s["rho"].copy())

            data['E'].append(solver_test.E.copy())
            data['B'].append(solver_test.B.copy())
            data['t'].append(solver_test.t)
            data['energy_total'].append(solver_test.CalcEFieldEnergy())
            data['energy_kin'].append(solver_test.CalcKinEnergery())


        """
        Saving
        """
        # Save all data
        for key, values in data.items():
            data[key] = np.array(values)
            np.save(data_files[key], data[key])
        print("call")
        print(data)
        # Rückgabe: alle gesammelten Daten als Tupel
        return tuple(data[k] for k in sorted(data.keys()))