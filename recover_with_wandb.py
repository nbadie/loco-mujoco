import wandb
import os

api = wandb.Api()
# Replace 'your_entity' and 'your_project' with your W&B details
run = api.run("deepmimic/xtktql4f") # run-20260118_111734-xtktql4f")
# ("/home/nadinebadie/loco-mujoco/prosthesis_rand_save/wandb/run-20260118_111734-xtktql4f")
# run = api.run("deepmimic/2026-01-18_11-17-33_SACHFoot_ProsRight_RandSocketPosY_Horizon2000_MimTorsoHandsJointRootQPos12QVel06_LatRange05Coeff01_ActPenSquare002_2BoxNew_solref_900_300_KN-120_5_HP-30_35_AnkL-20_20_AnkR-10_10_AnkStiffL900_Gas_SlackRange002_SocketTySlack_ObsPosOriSocketTalus_RandPosOriSocketTalus_LaPreStiffDamp_BothSideToeSameStiffDamp_NoSubTal")
# Create a local directory to match your old structure
os.makedirs("recovered_checkpoints", exist_ok=True)

for file in run.files():
    if file.name.endswith(".pkl") or "ckpt" in file.name:
        print(f"Downloading {file.name}...")
        file.download(root="recovered_checkpoints", replace=True)