import pandas as pd
import matplotlib.pyplot as plt
import json
import re


# Plot data from custom csv format

file_path = '/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Kinetmatics_Turcot_2013.csv'
#'/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Kinetics_Turcot_2013.csv'
# '/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Kinetmatics_Turcot_2013.csv'
#'/home/nadinebadie/loco-mujoco/prosthesis_evaluation/Alignment_Data_Schmalz_2002.csv'

# Read the file and parse the custom format
data = {}
current_label = None
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Match label line, e.g. FOOT_PLA = [
        if 'Schmalz' in file_path:
            label_match = re.match(r'^([A-Z_]+)\s*=\s*\[?', line)
        else:
            # It is Knee, Ankle etc
            label_match = re.match(r'^([A-Za-z_]+)\s*=\s*\[?', line)


        if label_match:
            current_label = label_match.group(1)
            data[current_label] = []
            continue
        # Match data line, e.g. 0.41848739495798526;15.405462184873954
        if current_label:
            vals = line.split(';')
            if len(vals) == 2:
                try:
                    x, y = map(float, vals)
                    data[current_label].append((x, y))
                except ValueError:
                    continue

# Plot each label's data
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

for label, points in data.items():
    if points:
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker='o', linestyle='-', label=label)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Custom CSV Data Plot')
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig('/home/nadinebadie/loco-mujoco/prosthesis_evaluation/plots/custom_data_kinematics.png', dpi=300, bbox_inches='tight')
# plt.savefig('/home/nadinebadie/loco-mujoco/prosthesis_evaluation/plots/custom_data_plot.png', dpi=300, bbox_inches='tight')


