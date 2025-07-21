import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import os
mpl.use('TkAgg')
import win32api
import win32con

def CallItRenderer(solver_test,solver_ref,total_steps,t_end):

    # Show a yes/no message box
    response = win32api.MessageBox(0, "Neurechnen?", "Simulation", win32con.MB_YESNO | win32con.MB_ICONQUESTION)

    # Create 'rendered' folder if it doesn't exist
    if not os.path.exists('rendered'):
        os.makedirs('rendered')

    # Define file paths for saved data (only for solver_test)
    data_files = {
        'x_test': 'rendered/x_test_history.npy',
        'v_test': 'rendered/v_test_history.npy',
        'x_ref': 'rendered/x_ref_history.npy',
        'v_ref': 'rendered/v_ref_history.npy',
        't': 'rendered/t_history.npy',
        'energy_total_test': 'rendered/energy_total_history_test.npy',
        'energy_kin_test': 'rendered/energy_kin_history_test.npy',
        'rho_test': 'rendered/rho_test_history.npy',
        'E_test': 'rendered/E_test_history.npy',
        'rho_ref': 'rendered/rho_ref_history.npy',
        'E_ref': 'rendered/E_ref_history.npy',
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

    if data_exists:
        # Load data from files
        x_test_history = np.load(data_files['x_test'])
        v_test_history = np.load(data_files['v_test'])
        x_ref_history = np.load(data_files['x_ref'])
        v_ref_history = np.load(data_files['v_ref'])
        t_history = np.load(data_files['t'])
        energy_total_history_test = np.load(data_files['energy_total_test'])
        energy_kin_history_test = np.load(data_files['energy_kin_test'])
        rho_test_history = np.load(data_files['rho_test'])
        E_test_history = np.load(data_files['E_test'])
        rho_ref_history = np.load(data_files['rho_ref'])
        E_ref_history = np.load(data_files['E_ref'])
        energy_total_history_ref = np.load(data_files['energy_total_ref'])
        energy_kin_history_ref = np.load(data_files['energy_kin_ref'])

    else:
        # Run simulation and store data
        x_test_history = []
        v_test_history = []
        x_ref_history = []
        v_ref_history = []
        t_history = []
        energy_total_history_test = []
        energy_kin_history_test = []
        energy_total_history_ref = []
        energy_kin_history_ref = []
        rho_test_history = []
        E_test_history = []
        rho_ref_history = []
        E_ref_history = []


        """
        Calculation
        """
        for _ in tqdm(range(total_steps), desc="Simulating", unit="step"):
            solver_test.step()
            solver_ref.step()  # Still run solver_ref for consistency, but don't store its data

            # Store data for solver_test (copy to avoid reference issues)
            electrons=solver_test.species[0]
            x_test_history.append(electrons["xp"].copy())
            v_test_history.append(electrons["vp"].copy())
            x_ref_history.append(solver_ref.xp.copy())
            v_ref_history.append(solver_ref.vp.copy())
            t_history.append(solver_test.t)
            energy_total_history_test.append(solver_test.CalcEFieldEnergy())
            energy_kin_history_test.append(solver_test.CalcKinEnergery())
            rho_test_history.append(electrons["rho"].copy() + 20)
            E_test_history.append(solver_test.E.copy())
            rho_ref_history.append(solver_ref.rho.copy() - 5080)
            E_ref_history.append(solver_ref.E.copy())
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
        rho_test_history = np.array(rho_test_history)
        E_test_history = np.array(E_test_history)
        rho_ref_history = np.array(rho_ref_history)
        E_ref_history = np.array(E_ref_history)

        # Save data to files
        np.save(data_files['x_test'], x_test_history)
        np.save(data_files['v_test'], v_test_history)
        np.save(data_files['x_ref'], x_ref_history)
        np.save(data_files['v_ref'], v_ref_history)
        np.save(data_files['t'], t_history)
        np.save(data_files['energy_total_test'], energy_total_history_test)
        np.save(data_files['energy_kin_test'], energy_kin_history_test)
        np.save(data_files['rho_test'], rho_test_history)
        np.save(data_files['E_test'], E_test_history)
        np.save(data_files['rho_ref'], rho_ref_history)
        np.save(data_files['E_ref'], E_ref_history)
        np.save(data_files['energy_total_ref'], energy_total_history_ref)
        np.save(data_files['energy_kin_ref'], energy_kin_history_ref)
    return (x_test_history, v_test_history, x_ref_history, v_ref_history, t_history,
            energy_total_history_test, energy_kin_history_test, rho_test_history,
            E_test_history, rho_ref_history, E_ref_history,energy_total_history_ref,energy_kin_history_ref)