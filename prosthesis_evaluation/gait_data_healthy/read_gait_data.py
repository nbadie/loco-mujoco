import pandas as pd

import matplotlib.pyplot as plt

def plot_mean_std_from_csv(csv_file_name):
    # Read the CSV file
    df = pd.read_csv(f"{csv_file_name}.csv")

    # Create subplots for each metric
    metrics = df['Parameter'].unique()
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    # Ensure axs is iterable even if there's only one subplot
    if num_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        # Extract data for the current metric
        metric_data = df[df['Parameter'] == metric]['Data'].values[0]
        import numpy as np  # Ensure numpy is imported for eval to recognize 'array'
        metric_data = eval(metric_data.replace('array', 'np.array'))  # Replace 'array' with 'np.array' for proper evaluation

        # Calculate mean and std
        metric_data = np.array(metric_data)
        mean_values = np.mean(metric_data, axis=0)
        std_values = np.std(metric_data, axis=0)

        # Plot the mean and std
        axs[i].plot(mean_values, label='Mean', color='blue')
        axs[i].fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.2, label='STD', color='blue')
        axs[i].set_title(f"{metric.capitalize()} Mean and STD")
        axs[i].set_xlabel("Time Points")
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()

    plt.tight_layout()
    plt.savefig('kinematics_test.png')
    plt.show()





plot_mean_std_from_csv("walking_data")