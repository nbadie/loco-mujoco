# README for Kinematic Data

"""
This README provides an overview of the saved kinematic and ground reaction force data in CSV format and instructions on how to load and print the data.

## Overview
The kinematic and ground reaction force data is saved in CSV files, where each file contains interpolated data for specific metrics (e.g., hip flexion, knee angle, ankle angle, ground reaction forces). The data is taken over all steps within a 10 s window from three rollouts across three different seeds. Each row in the CSV file corresponds to a metric, and the data column contains the interpolated values for that metric. We include three csv files for three different gait speeds: 'walking_data.csv' for 1.2 m/s, 'running_2ms-1' for 2 m/s, and 'running_3ms-1' for 3 m/s. 

## File Structure
- Parameter: The name of the parameter (e.g., hip_flexion, knee_angle).
- Data: A list of interpolated data points for the corresponding parameter.

## Loading and Ploting the Data
To load and plot the data, use the provided script ('read_gait_data.py'). 
