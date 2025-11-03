import pickle 
import matplotlib.pyplot as plt
import numpy as np
# Define a mapping from variable names to output paths
# Define a mapping from variable names to output paths
run_configs = {
    "400stiff_healthy": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-08/23-11-53/20250822_181403_evaluation_results_1000steps_ankleStiff400.pkl",
    "500stiff_healthy": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-08/23-11-53/20250822_181422_evaluation_results_1000steps_ankleStiff500.pkl",
    "300stiff_healthy": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-08/23-11-53/20250822_181455_evaluation_results_1000steps_ankleStiff300.pkl",
    # "250stiff_healthy": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-08/23-11-53/20250822_184637_evaluation_results_1000steps_ankleStiff250.pkl",
    # "200stiff_healthy": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-08/23-11-53/20250822_184651_evaluation_results_1000steps_ankleStiff200.pkl"
}

def load_data(run_paths):
    loaded = {}
    for name, path in run_paths.items():
        with open(path, "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded

all_loaded_data = load_data(run_configs)

def get_joint_angles(loaded_data, angle_key_r="ankle_angle_r_angle", angle_key_l="ankle_angle_l_angle", deg=True):
    angles = {}
    for name, data in loaded_data.items():
        r = data.get(angle_key_r)
        l = data.get(angle_key_l)
        if deg:
            r = np.rad2deg(r)
            l = np.rad2deg(l)
        angles[name] = {"r": r, "l": l}
    return angles

joint_angles = get_joint_angles(all_loaded_data)

plt.figure(figsize=(12, 6))
for name, angles in joint_angles.items():
    plt.plot(angles["l"], label=f"{name} l")
    plt.plot(angles["r"], label=f"{name} r")

plt.legend()
plt.grid()
plt.savefig('ankle_angle_plot.png')
