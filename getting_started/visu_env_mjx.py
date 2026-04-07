import time 
import jax 
import mujoco
import mujoco.viewer
from loco_mujoco import ImitationFactory
from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesis 
# from loco_mujoco.environments.humanoids.skeleton_prosthesis import MjxSkeletonMuscleProsthesisNoArms
# from mujoco import mjx
import numpy as np

def main(): 
    # env = ImitationFactory.make("MjxSkeletonMuscle", default_dataset_conf=dict(task="walk"))
    # env = MjxSkeletonMuscleProsthesisNoArms(prosthesis_side = "left_side", #"left_side",  #"bilateral", #"left_side", 
    # env = SkeletonMuscleProsthesis(prosthesis_side = "right_side", #"left_side",  #"bilateral", #"left_side", 
    env = MjxSkeletonMuscleProsthesis(prosthesis_side = "right_side", #"left_side",  #"bilateral", #"left_side", 
                                      prosthesis_type= "transtibial", 
                                      prosthesis_subtype = "SACH",
                                      disable_arms = False,
                                      delete_joints= False, remove_joint_names=['subtalar_angle'], #,'ankle_angle'],
                                      limit_knee_extension= True,
                                      knee_extension_limit = {
                                          'ankle_angle_l': [-20, 20], 
                                          'ankle_angle_r': [-10, 10], 
                                          'hip_flexion_r': [-30, 35],
                                          'hip_flexion_l': [-30, 35],
                                          'knee_angle_r': [-120, 5],
                                          'knee_angle_l': [-120, 5]
                                      },
                                      tibia_socket_offset_z = 0, tibia_socket_offset_x = 0, tibia_socket_overlap= 0.2, amputated_tibia_length= 0.2,
                                      #joint_stiffness = {'ankle_angle': 500, 'mtp_angle': 10}, joint_damping = {'ankle_angle': 10, 'mtp_angle': 0.01},
                                      reattach_muscle = True, #True, #True, #True, 
                                      replace_joint =True, 
                                      socket_joint = True, #False,
                                    #   prosthesis_visualization = True, #True,
                                      reattach_muscles_offset = {'med_gas': [0,0.0,0]},
                                      contact_solref= [-900,-300],
                                      add_pos_ori_to_observation = True,
                                      joint_stiffness = {'ankle_angle': 900},

                                      contact_geom_type= '2boxes_new', #'sphere_scone_exact', #_exact', #'3spheres_scone', #'2boxes_new', #'2box', #'2boxes_new' , #'2boxes_small', #'2boxes_new' , #'3spheres_scone', #'sphere_scone_exact', # 'sphere_tiptoe', #'sphere_scone', #'cylinderHeel_sphereToe', #'2box', # 'sphere', #'cylinder', #'sphere', #, 'lat_gas':[0,0,0]},
                                    #   stiffen_and_dampen_joint_names = ['ankle_angle','mtp_angle'],joint_stiffness=100, joint_damping=1, 
                                      use_2_box_per_foot= True, #False, #True, #False, #True ,
                                      #use_box_feet = False,
                                      add_sensors = True )
    
# # # # Alternative 
    model = env.model #spec.compile() #mujoco.MjModel.from_xml_path(xml_path)
    data = env.data #mujoco.MjData(model)

# create keys
    key = jax.random.key(0)
    n_envs = 100
    keys = jax.random.split(key, n_envs + 1)
    key, env_keys = keys[0], keys[1:]


    with mujoco.viewer.launch(model, data) as viewer:


        # viewer.sync()
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data) 

            # Pick up changes to the physics state, apply perturbations, update options from GUI
            viewer.sync()

            # Time keeping 
            time_until_new_step = model.opt.timestep - (time.time() - step_start)
            if time_until_new_step > 0:
                time.sleep(time_until_new_step)

    pass


if __name__ == "__main__":
    main()