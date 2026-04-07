# Prosthetic Socket Fitting and Alignment Effects on Lower-Limb Amputee Gait: A Simulation Study

![continous integration](https://github.com/robfiras/loco-mujoco/actions/workflows/continuous_integration.yml/badge.svg?branch=dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **end-to-end framework** transforming musculoskeletal models into prosthetic simulations through digital surgery, modular prosthesis fitting, and gait synthesis. Our work expands LocoMuJoCo which is an **imitation learning benchmark** for **whole-body control** with validated, modular design. For more information on LocoMuJoCo See [LocoMuJoCo datasets](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets) and [original repository](https://github.com/robfiras/loco-mujoco) for more details on them. 

Bridging the gap between musculoskeletal modeling and prosthetic engineering. This framework provides an automated, end-to-end pipeline that transforms generic musculoskeletal models into functional prosthetic simulations. By leveraging GPU-accelerated physics (MJX) and Reinforcement Learning, we enable researchers to optimize hardware and alignment in a high-fidelity virtual environment.

Built as an extension of LocoMuJoCo (https://github.com/robfiras/loco-mujoco), this tool is designed for the rapid iteration of prosthetic designs and the study of human-device interaction.

### Key Features
✅ **Virtual Innovation Lab** – Open-source platform for risk-free hardware and alignment optimization.<br/>
✅ **Validated pipeline** – From digital surgery and socket fitting to robust, synthesized gait policies.<br/>
✅ **Precision Alignment** – Quantify the biomechanical sensitivity of prosthetic shifts beyond the resolution of clinical observation.<br/>   
✅ **Modular design** – Easily swap components, reward functions, and domain randomization parameters.<br/>
✅ **MJX-accelerated** – Harness the power of JAX for massively parallel gait synthesis and curriculum training.<br/>


## Installation
Follow https://github.com/robfiras/loco-mujoco: 
  1. Clone and install the core framework
  2. (Optional) Enable GPU acceleration with JAX


## Quick Start
We provide comprehensive tutorials in the [examples folder]{./examples} to get you running in minutes:
* Curriculum Training: Train an RL agent to adapt to a new prosthetic configuration. 
* MJX Environments: Run massive batches of simulations in parallel. 
* Domain Randomization: Test the robustness of your prosthetic alignment across varying conditions. 
* Systematic Alignment Testing: Evaluate trained control policies across a range of prosthetic alignments. 



## Citation
This framework is currently part of a manuscript under review. If you use this code or the associated models in your research, please contact Nadine Badie (nadine.badie@imsb.uni-stuttgart.de) for the appropriate citation details.




