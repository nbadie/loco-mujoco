import pickle 
import matplotlib.pyplot as plt

# # with open("/home/nadinebadie/loco-mujoco/prosthesis_test/outputs/2025-06-11/16-30-59/20250625_140700_evaluation_results_1000steps.pkl", "rb") as f:
# #     loaded_data = pickle.load(f)

# # all_sensor_data = loaded_data.get("all_sensor_force")
# # left_knee_torque_sensor = all_sensor_data["left_knee_mimic_torque_sensor"]
# # print("left_knee_torque_sensor", left_knee_torque_sensor[0:10])
# # right_knee_torque_sensor = all_sensor_data["right_knee_mimic_torque_sensor"]
# # right_foot_torque_sensor = all_sensor_data["right_foot_mimic_torque_sensor"]
# # left_foot_torque_sensor = all_sensor_data["left_foot_mimic_torque_sensor"]

# # plt.figure(figsize=(12, 6))
# # plt.plot(left_knee_torque_sensor, label='Left Knee Torque Sensor', alpha=0.5)
# # plt.plot(right_knee_torque_sensor, label='Right Knee Torque Sensor', alpha=0.5)
# # plt.plot(left_foot_torque_sensor, label='Left Foot Torque Sensor', alpha=0.5)
# # plt.plot(right_foot_torque_sensor, label='Right Foot Torque Sensor', alpha=0.5)
# # plt.xlabel('Time Step')
# # plt.ylabel('Torque Sensor Values')
# # plt.title('Torque Sensor Values Over Time')
# # plt.legend()
# # plt.grid()
# # plt.savefig('torque_sensor_plot.png')

# file_name = "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-11/04-13-27/20250711_163756_evaluation_results_1000steps.pkl"
# # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-11/04-13-27/20250711_162757_evaluation_results_1000steps.pkl"
# # "/home/nadinebadie/loco-mujoco/prosthesis_test/outputs/2025-06-11/16-30-59/20250625_140700_evaluation_results_1000steps.pkl"
# with open(file_name, "rb") as f:
#     loaded_data = pickle.load(f)
# file_name_1= "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-11/15-40-47/20250711_225012_evaluation_results_1000steps.pkl"
# with open(file_name_1, "rb") as f:
#     loaded_data_1 = pickle.load(f)

# file_name_2 = "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-12/01-00-33/20250712_132758_evaluation_results_1000steps.pkl"
# with open(file_name_2, "rb") as f: 
#     loaded_data_2 = pickle.load(f)


# file_name_3 = "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-12/01-00-33/20250712_140420_evaluation_results_1000steps.pkl"
# with open(file_name_3, 'rb') as f: 
#     loaded_data_3 = pickle.load(f)

# tx = 'socket_tx_l'
# ty = 'socket_ty_l'
# tz = 'socket_tz_l'
# flexion = 'socket_flexion_l'
# rotation = 'socket_rotation_l'
# adduction = 'socket_adduction_l'

# parameter = '_angle'

# all_parameters = [tx, ty, tz, flexion, adduction, rotation]
# all_parameters = [name +parameter for name in all_parameters]





# # file_name1= "/home/nadinebadie/loco-mujoco/prosthesis_test/outputs/2025-06-11/16-30-59/20250625_140700_evaluation_results_1000steps.pkl"
# # with open(file_name1, "rb") as f:
# #     loaded_data1 = pickle.load(f)
# # variable_name = "all_grf_l"
# # variable_data = [data[5] for data in loaded_data[variable_name]]
# # variable_data1 = [data[5] for data in loaded_data1[variable_name]]
# # plt.figure(figsize=(12,6))
# # plt.plot(variable_data, label= f"{variable_name} new")
# # plt.plot(variable_data1, label= variable_name)
# # plt.xlabel('Time')
# # plt.ylabel(variable_name)
# # plt.legend()
# # plt.grid()
# # plt.savefig(f'{variable_name}.png')




# def plot_from_loaded_data(loaded_data,loaded_data_1,loaded_data_2, loaded_data_3,variable_name):
#     # variable_data = [data[5] for data in loaded_data[variable_name]] #loaded_data[variable_name]
#     variable_data = loaded_data[variable_name]#loaded_data[variable_name]
#     variable_data_1 = loaded_data_1[variable_name] #loaded_data[variable_name]
#     variable_data_2 = loaded_data_2[variable_name] #loaded_data[variable_name]
#     variable_data_3 = loaded_data_3[variable_name] #loaded_data[variable_name]
#     plt.figure(figsize=(12,6))
#     plt.plot(variable_data, label= variable_name)
#     plt.plot(variable_data_1, label= f'{variable_name}_med_gas')
#     plt.plot(variable_data_2, label= f'{variable_name}_med_gas_TalusStiffDampSame')
#     plt.plot(variable_data_3, label= f'{variable_name}_med_gas_TalusStiffDampSame_stiffSocket')
#     plt.xlabel('Time')
#     plt.ylabel(variable_name)
#     plt.legend()
#     plt.grid()
#     plt.savefig(f'{variable_name}_2cont.png')

# for name in all_parameters: 
#     plot_from_loaded_data(loaded_data, loaded_data_1,loaded_data_2,loaded_data_3, name)
# # plot_from_loaded_data(loaded_data, "all_grf_r")



def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f) #, encoding='latin1')

def plot_from_loaded_data(data_list, labels, variable_name, save_folder="plots"):
    """
    Plots a variable from multiple loaded datasets.

    Args:
        data_list: List of dicts containing variable_name as key.
        labels: List of labels for each dataset.
        variable_name: The name of the variable to plot.
        save_folder: Folder where to save the plots (default: 'plots').
    """
    plt.figure(figsize=(12, 6))
    for data, label in zip(data_list, labels):
        if variable_name in data:
            plt.plot(data[variable_name], label=label)
        else:
            print(f"Warning: '{variable_name}' not found in dataset labeled '{label}'")

    plt.xlabel("Time")
    plt.ylabel(variable_name)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{variable_name}.png")
    plt.close()



file_paths = [
    # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-11/04-13-27/20250711_163756_evaluation_results_1000steps.pkl",
    # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-11/15-40-47/20250711_225012_evaluation_results_1000steps.pkl",
    # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-12/01-00-33/20250712_132758_evaluation_results_1000steps.pkl",
    # # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-12/01-00-33/20250712_132758_evaluation_results_1000steps.pkl",
    # # # # # #"/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-12/01-00-33/20250712_140420_evaluation_results_1000steps.pkl",
    # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/12-30-23/20250713_224320_evaluation_results_1000steps.pkl",
    # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250714_102933_evaluation_results_1000steps.pkl",
    # # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_090024_evaluation_results_1000steps.pkl",
    
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_092202_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_092319_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_093639_evaluation_results_1000steps.pkl",
    # # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_093719_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_105317_evaluation_results_1000steps.pkl",
    # # # #"/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_105413_evaluation_results_1000steps.pkl"
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_120650_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_120729_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_123233_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_123401_evaluation_results_1000steps.pkl"
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_130842_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_133516_evaluation_results_1000steps.pkl", 
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_133609_evaluation_results_1000steps.pkl",
   
   
   
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-13/22-55-54/20250715_140255_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_001754_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-16/01-07-33/20250716_090607_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-16/01-07-33/20250716_121606_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-16/01-07-33/20250716_124352_evaluation_results_1000steps.pkl",
    # # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_125436_evaluation_results_1000steps.pkl"
    
    
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_164439_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_170058_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_170224_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_172237_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_172308_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_180002_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_182613_evaluation_results_1000steps.pkl",
    
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_184409_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_184450_evaluation_results_1000steps.pkl",
    # # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_185042_evaluation_results_1000steps.pkl",

    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_190212_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-15/15-13-07/20250716_190609_evaluation_results_1000steps.pkl",


    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_125329_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_163622_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_163700_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_163727_evaluation_results_1000steps.pkl",


    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_165510_evaluation_results_1000steps.pkl",
    "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_165934_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_165957_evaluation_results_1000steps.pkl",


    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_171827_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_171853_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_172047_evaluation_results_1000steps.pkl",
    # "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_172135_evaluation_results_1000steps.pkl",


    # "",
    "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_173931_evaluation_results_1000steps.pkl",
    "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_174122_evaluation_results_1000steps.pkl",
    "/home/nadinebadie/loco-mujoco/prosthesis_randomization/outputs/2025-07-17/02-11-17/20250717_174144_evaluation_results_1000steps.pkl",

]

labels = [
    # # # # "baseline",
    # # # # "med_gas",
    # # # # "med_gas_TalusStiffDampSame",
    # # # # # "med_gas_TalusStiffDampSame_stiffSocket",
    # # # # "med_gas_KneeMimic_TalusStiffDampSame_stiffSocket",
    # # # "med_gas_KneeMimic_TalusStiffDampSame_stiffSocket_NoAnkle", 
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp10_NoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp10all_NoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100_NoAnkle",
    # # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff10000_NoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff40000_NoAnkle",
    # # #"med_gas_KneeMimic_TalusStiffDampSame_damp400stiff40000_NoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_txdamp100stiff40000_tz21st100damp_NoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff40000_tz10st100dampNoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff40000_tz30st100dampNoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff40000_tz30st50dampNoAnkle",
    # # "med_gas_KneeMimic_TalusStiffDampSame_damp100stiff40000_tz43st50dampNoAnkle",
    # "tzstif30damp100_tystif43damp400",
    # "tzstif43damp100_tystif43damp400",
    # "tzstif43damp100_tystif43damp100",
    # "tzstif43damp40_tystif43damp40",
    
    # "walking",
    
    # # # "walking_noRand",
    # # # "walking_noRand_Knee0",
    # # # "walking_noRand_meanStiff",
    # # # "walking_meanStiff",
    
    # # "walking_solimp",
    # # "walking_solimp_20",
    # # "walking_solimp_30",
    # # "walking_solimp_10",
    # # "walking_solimp_8",
    # # "walking_solimp_0",
    # "walking_solimp_0_halfRange_halfWidth",
    
    # # "walking_solimp_0_halfRange_018Width",
    # # "walking_solimp_0_halfRange_02Width",
    # # "walking_solimp_0_018Range_018Width",

    # "walking_solimp_0_0020-0015Range_001Width",
    # "walking_solimp_0_0019-001Range_0015Width"

    # "085solimp0018_Rrange_mar0018",
    # "085solimp0018_Rrange_mar003",
    # "085solimp0018_Rrange_mar005",
    # "085solimp0018_Rrange_mar007",

    # "085solimp001_Rrange_mar005",
    "096solimp0008_Rrange_mar0008",
    # "088solimp0004_Rrange_mar0004",

    # "096solimp0007_Rrange_mar0007",
    # "096solimp0006_Rrange_mar0006",
    # "094solimp0006_Rrange_mar0006",
    # "092solimp0002_Rrange_mar001",


    "096solimp0008_Rrange_mar0008",    
    "092solimp0008_Rrange_mar008",    
    "092solimp0008_Rrange_mar005",    





    


]

loaded_datasets = [load_pickle(fp) for fp in file_paths]

tx = 'socket_tx_l'
ty = 'socket_ty_l'
tz = 'socket_tz_l'
flexion = 'socket_flexion_l'
rotation = 'socket_rotation_l'
adduction = 'socket_adduction_l'
parameter_suffix = '_angle'

all_parameters = [name + parameter_suffix for name in [tx, ty, tz, flexion, adduction, rotation]]

for name in all_parameters:
    plot_from_loaded_data(loaded_datasets, labels, name)