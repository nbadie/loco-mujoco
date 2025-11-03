import pickle 
import matplotlib.pyplot as plt
import numpy as np

# Plot socket_ty displacement, socket_ty force_constraint, smooth and  GRF in one plot with 4 subplots 

run_configs = {
    # # "T1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/06-50-25/20250901_132507_evaluation_results_1000steps.pkl",
    # # "T2": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/06-50-25/20250901_132518_evaluation_results_1000steps.pkl",
    # # # "T3": "",
    # # "T1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_174409_evaluation_results_1000steps_0seed.pkl",
    # # "T2": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_174446_evaluation_results_1000steps_0seed.pkl",
    # "T3": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_174549_evaluation_results_1000steps_0seed.pkl",
    # # "T4": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_220802_evaluation_results_1000steps_0seed.pkl",
    # "T5": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_220929_evaluation_results_1000steps_0seed.pkl",
    "Org": '/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-27/09-05-19/20250827_222323_evaluation_results_1000steps.pkl',
    # "Neg": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_224645_evaluation_results_1000steps_0seed.pkl",
    # # "Gemini": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_225844_evaluation_results_1000steps_0seed.pkl",
    # "All_Neg": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_230702_evaluation_results_1000steps_0seed.pkl",
    # # "WOExtra-": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_231108_evaluation_results_1000steps_0seed.pkl",
    # # # "GeminiWOExtra-": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-08-26/23-28-37/20250901_231210_evaluation_results_1000steps_0seed.pkl",
    "TN":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_092239_evaluation_results_1000steps.pkl",
#     # "TN_Chat":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_101138_evaluation_results_1000steps_0seed.pkl",
#     # # "TN_Chat_Neg_-1":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_102443_evaluation_results_1000steps_0seed.pkl",
#     # "TN_Chat_Neg_Ext":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_102459_evaluation_results_1000steps_0seed.pkl",
#     # # # "TN_Chat_Pos_Ext":"",
#     # # "TN_Chat_Pos_-1":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_102612_evaluation_results_1000steps_0seed.pkl",
#     # "T20":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_102553_evaluation_results_1000steps_0seed.pkl",
#     # # "T21":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_102612_evaluation_results_1000steps_0seed.pkl",

#     # # "TN_Swap": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_113632_evaluation_results_500steps_0seed.pkl",
#     # # "TN_Swap_01": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_113648_evaluation_results_500steps_0seed.pkl",
#     # # "TN_Swap_01_Ext": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_114519_evaluation_results_500steps_0seed.pkl",
#     # "TN_3x": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_115405_evaluation_results_500steps_0seed.pkl",
#     # "TN_3x_L-1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_115707_evaluation_results_500steps_0seed.pkl",
#     # "TN_3x_LR-1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_115854_evaluation_results_500steps_0seed.pkl",
#     # "TN_05x": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_121851_evaluation_results_500steps_0seed.pkl",
#     # "TN_02x": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_121903_evaluation_results_500steps_0seed.pkl",
#     # "1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_130145_evaluation_results_500steps_0seed.pkl",
#     # "2": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_130248_evaluation_results_500steps_0seed.pkl",
#     # # "3":"/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_130355_evaluation_results_500steps_0seed.pkl",
#     # "4": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_135902_evaluation_results_500steps_0seed.pkl",
#     # "5": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_140035_evaluation_results_500steps_0seed.pkl",
#     # "7": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_143006_evaluation_results_500steps_0seed.pkl",
# #     "8": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_143045_evaluation_results_500steps_0seed.pkl",
# #     # "9": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_143408_evaluation_results_500steps_0seed.pkl",
# #     "10_005": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_145901_evaluation_results_500steps_0seed.pkl",
# #     "11_001": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_150037_evaluation_results_500steps_0seed.pkl",
#     # "8_1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_151735_evaluation_results_500steps_0seed.pkl",
#     # "9_1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_151745_evaluation_results_500steps_0seed.pkl",
#     "11_1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_153313_evaluation_results_500steps_0seed.pkl",
#     # "12-1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_153313_evaluation_results_500steps_0seed.pkl",
#     "13-1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_201425_evaluation_results_500steps_0seed.pkl",
#     # "14-1": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_212203_evaluation_results_500steps_0seed.pkl",

    # "Gemini_org": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_214408_evaluation_results_500steps_0seed.pkl",
    # "Gemini_pap": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_214849_evaluation_results_500steps_0seed.pkl", 
    "Auto_7": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_223943_evaluation_results_500steps_0seed.pkl",
    "Auto_5": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_225849_evaluation_results_500steps_0seed.pkl",
    "Auto_6": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_225938_evaluation_results_500steps_0seed.pkl",
    # "Auto_4": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_230017_evaluation_results_500steps_0seed.pkl",
    # "Auto_3": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_230017_evaluation_results_500steps_0seed.pkl",
    "Auto_8": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_232435_evaluation_results_500steps_0seed.pkl",
    "Auto_9": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_232448_evaluation_results_500steps_0seed.pkl",
    "Auto_10": "/home/nadinebadie/loco-mujoco/prosthesis_rand_save/outputs/2025-09-01/23-34-16/20250902_232505_evaluation_results_500steps_0seed.pkl",

}

# Load and process data for each run configuration
for label, file_path in run_configs.items():
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

# Parameters to plot 
ty_name = "socket_ty_l_"
angle_name = ty_name  + "angle"
force_constraint_name = ty_name  +"forces_constraint"
forces_smooth_name = ty_name  + "forces_smooth"
grf_l_name = "all_grf_l"

def extract_data(loaded_data, key):
    return loaded_data.get(key, [])

# Store data for all runs
all_angles = {}
all_force_constraints = {}
all_forces_smooth = {}
all_grf_l = {}
all_forces_applied = {}

for label, file_path in run_configs.items():
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    all_angles[label] = extract_data(loaded_data, angle_name)
    all_force_constraints[label] = extract_data(loaded_data, force_constraint_name)
    all_forces_smooth[label] = extract_data(loaded_data, forces_smooth_name)
    all_grf_l[label] = extract_data(loaded_data, grf_l_name)
    all_forces_applied[label] = extract_data(loaded_data, ty_name + "forces_applied")

# Plot
fig, axs = plt.subplots(4, 1, figsize=(12, 10)) #, sharex=True)

for label in run_configs.keys():
    axs[0].plot(all_angles[label], label=label)
    # axs[1].plot(all_force_constraints[label], label=label)
    # axs[2].plot(all_forces_smooth[label], label=label)
    axs[1].plot([x[0] for x in all_force_constraints[label]], label=label)
    # axs[3].plot(all_grf_l[label][:,5], label=label)
    axs[2].plot(all_angles[label], [x[0] for x in all_force_constraints[label]], label=label)
    # axs[3].plot([x[5] for x in all_grf_l[label]],label=label)
    

    # Calculate Stiffness which is Force / Displacement
    # Flatten all_angles[label] if it contains nested lists
    # Ensure flat_angles and all_forces_applied[label] are 1D arrays of scalars
    flat_angles = np.array([a if np.isscalar(a) else a[0] for a in all_angles[label]]).flatten()
    forces_applied_flat = np.array([f if np.isscalar(f) else f[0] for f in all_forces_applied[label]]).flatten()
    # Calculate stiffness safely
    stiffness = np.array([f / d if d != 0 else 0 for f, d in zip(forces_applied_flat, flat_angles)])
    axs[3].plot(flat_angles, stiffness, label=label)

axs[0].set_title("Socket Ty Angle")
axs[0].set_ylabel("Angle")
axs[0].legend()
axs[0].grid()

# axs[1].set_title("Socket Ty Force Constraint")
# axs[1].set_ylabel("Force Constraint")
# axs[1].legend()
# axs[1].grid()

# axs[2].set_title("Socket Ty Forces Smooth")
# axs[2].set_ylabel("Forces Smooth")
# axs[2].legend()
# axs[2].grid()

# axs[3].set_title("GRF L")
# axs[3].set_ylabel("GRF")
# axs[3].legend()
# axs[3].grid()
# axs[3].set_xlabel("Step")

axs[1].set_title("Socket Ty Force Applied")
axs[1].set_ylabel("Force Applied")
axs[1].legend()
axs[1].grid()

axs[2].set_title("Socket Ty Force Applied over Displacement")
axs[2].set_ylabel("Force Applied")
axs[2].set_xlabel("Displacement")
axs[2].legend()
axs[2].grid()

axs[3].set_title("Socket Ty Stiffness")
axs[3].set_ylabel("Stiffness")
axs[3].set_xlabel("Angle")
axs[3].legend()
axs[3].grid()

fig.tight_layout()
fig.savefig("socket_ty_displacement_and_forces_compare__.png")
plt.show()
