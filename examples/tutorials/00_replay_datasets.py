import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf

env_params = {
    # Only required parameters, rest will be defaulted
    "prosthesis_side": "right",
    "prosthesis_type": "transtibial", 
    "prosthesis_subtype": "SACH",
    "amputated_tibia_length": 0.2,
    "tibia_socket_overlap": 0.2, 
}
# # example --> you can add as many datasets as you want in the lists!
env = ImitationFactory.make("MjxSkeletonMuscleProsthesis", **env_params,
                            default_dataset_conf=DefaultDatasetConf(["walk"]),
                            # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4"]),
                            # if SMPL and AMASS are installed, you can use the following:
                            # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
                            #                                     "KIT/12/WalkInClockwiseCircle11_poses",
                            #                                     "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
                            #                                     'KIT/359/walking_fast05_poses']),
                            n_substeps=20)

env.play_trajectory(n_episodes=3, n_steps_per_episode=500, render=True)


