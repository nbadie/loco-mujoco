# Prosthetic Socket Fitting and Alignment Effects on Lower-Limb Amputee Gait: A Simulation Study
<p align="center">
  <img width="70%" src="https://github.com/robfiras/loco-mujoco/assets/69359729/bd2a219e-ddfd-4355-8024-d9af921fb92a">
</p>

![continous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/loco-mujoco/badge/?version=latest)](https://loco-mujoco.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Us-7289DA?style=flat&logo=discord&logoColor=white)](https://discord.gg/gEqR3xCVdn)

# An **end-to-end framework** transforming musculoskeletal models into prosthetic simulations through digital surgery, modular prosthesis fitting, and gait synthesis. Our work expands LocoMuJoCo which is an **imitation learning benchmark** for **whole-body control** with validated, modular design. For more information on LocoMuJoCo See [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) and [original repository](https://github.com/robfiras/loco-mujoco) for more details on them. 

Bridging the gap between musculoskeletal modeling and prosthetic engineering. This framework provides an automated, end-to-end pipeline that transforms generic musculoskeletal models into functional prosthetic simulations. By leveraging GPU-accelerated physics (MJX) and Reinforcement Learning, we enable researchers to optimize hardware and alignment in a high-fidelity virtual environment.

Built as an extension of LocoMuJoCo (https://github.com/robfiras/loco-mujoco), this tool is designed for the rapid iteration of prosthetic designs and the study of human-device interaction.

### Key Features
✅ **Virtual Innovation Lab** – Open-source platform for risk-free hardware and alignment optimization  
✅ **Validated pipeline** – From digital surgery and socket fitting to robust, synthesized gait policies
# ✅ **Validated Pipeline** – Transforms musculoskeletal models into models with prosthesis using MjSpec  
# ✅ **Curriculum Learning** - Applies curriculum learning to model with prosthesis to learn dynamic prosthetic simulations with rigorously validated gaits  
✅ **Precision Alignment** – Quantify the biomechanical sensitivity of prosthetic shifts beyond the resolution of clinical observation   
✅ **Modular design** – Easily swap components, reward functions, and domain randomization parameters
✅ **MJX-accelerated** – Harness the power of JAX for massively parallel gait synthesis and curriculum training

```
## Installation
Follow https://github.com/robfiras/loco-mujoco: 
1. Clone and install the core framework:
2. (Optional) Enable GPU acceleration with JAX:
```

## Quick Start
We provide comprehensive tutorials in the [examples folder]{./examples} to get you running in minutes:
* Curriculum Training: Train an RL agent to adapt to a new prosthetic configuration.
* MJX Environments: Run massive batches of simulations in parallel.
* Domain Randomization: Test the robustness of your prosthetic alignment across varying conditions.



## Citation
#```
#@inproceedings{alhafez2023b,
#title={LocoMuJoCo: A Comprehensive Imitation Learning Benchmark for Locomotion},
#author={Firas Al-Hafez and Guoping Zhao and Jan Peters and Davide Tateo},
#booktitle={6th Robot Learning Workshop, NeurIPS},
#year={2023}
#}
```




