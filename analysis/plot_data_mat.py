import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pickle
import argparse
from numba import njit
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import median_abs_deviation
from scipy.interpolate import griddata
import shutil

from towertask.config import DATA_DIR, FIGURE_DIR, get_figure_path

def smooth_and_threshold_in_time(signal, sigma=1.0, threshold_factor=2.0):
    """
    Smooth a 1D signal using a Gaussian filter and then threshold values
    below threshold_factor * robust_sigma (MAD).
    """
    # 1D Gaussian smoothing across time
    smoothed = gaussian_filter1d(signal, sigma=sigma, 
                                 mode='reflect',)
    
    # Calculate robust sigma. By default, median_abs_deviation returns the MAD,
    # which is approximately sigma * 0.6745. We can simply treat that as "robust sigma"
    # or multiply by 1.4826 if you prefer the traditional scaling to approximate stdev.
    mad_val = median_abs_deviation(smoothed, scale='normal')  # scale='normal' uses 1.4826 factor
    cutoff = threshold_factor * mad_val
    
    # Threshold
    smoothed[smoothed < cutoff] = 0
    return smoothed

def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_name, a=minval, b=maxval), new_colors)

def normalize_column(col):
    # scale to [-1, 1]
    # return 2 * (col - col.min()) / (col.max() - col.min()) - 1
    
    # scale to [0,1]
    return (col - col.min()) / (col.max() - col.min())

def standardize_column(col):
    return (col - col.mean()) / col.std()

@njit
def shift_array(arr, shift_amount):
    return np.roll(arr, shift_amount)

def circular_shift_neuron_activity(df, neuron_columns):
    """
    Circularly shifts the activity of each neuron by a random interval within each trial.

    Args:
    - df: DataFrame containing the trials and neuron activities.

    Returns:
    - DataFrame with circularly shifted neuron activities.
    """
    df_shifted = df.copy()
    for episode in df['episode'].unique():
        mask = df['episode'] == episode
        episode_df = df.loc[mask, neuron_columns]
        num_rows = episode_df.shape[0]
        
        for neuron in neuron_columns:
            shift_amount = np.random.randint(num_rows)
            df_shifted.loc[mask, neuron] = np.roll(episode_df[neuron].values, shift_amount)
    
    return df_shifted

def compute_mean_mutual_information_evidence_shuffled(df, neuron_columns, num_shuffles=100):
    """
    Computes the mean and standard deviation of mutual information for each neuron
    across a specified number of shuffles, with respect to a given variable.

    Args:
    - df: DataFrame with the neural activity and contextual variables.
    - neuron_columns: List of neuron column names.
    - num_shuffles: Number of shuffles to perform.
    - variable: The variable to compute mutual information against ('position', 'evidence', or both).

    Returns:
    - A dictionary with neurons as keys and tuples (mean, sd) of mutual information as values.
    """
    # mi_pos_shuffles = {neuron: [] for neuron in neuron_columns}
    mi_evi_shuffles = {neuron: [] for neuron in neuron_columns}
    for _ in tqdm(range(num_shuffles)):
        shuffled_df = circular_shift_neuron_activity(df, neuron_columns)

        mi_evi = compute_mutual_information(shuffled_df, variable='accumulated_evidence')
        for neuron in neuron_columns:
            mi_evi_shuffles[neuron].append(mi_evi[neuron])
    
    # mi_mean_sd_pos = {neuron: (np.mean(values), np.std(values)) for neuron, values in mi_pos_shuffles.items()}
    mi_mean_sd_evi = {neuron: (np.mean(values), np.std(values)) for neuron, values in mi_evi_shuffles.items()}
    return mi_mean_sd_evi

def compute_mean_mutual_information_shuffled(df, neuron_columns, num_shuffles=100):
    """
    Computes the mean and standard deviation of mutual information for each neuron
    across a specified number of shuffles, with respect to a given variable.

    Args:
    - df: DataFrame with the neural activity and contextual variables.
    - neuron_columns: List of neuron column names.
    - num_shuffles: Number of shuffles to perform.
    - variable: The variable to compute mutual information against ('position', 'evidence', or both).

    Returns:
    - A dictionary with neurons as keys and tuples (mean, sd) of mutual information as values.
    """
    mi_pos_shuffles = {neuron: [] for neuron in neuron_columns}
    mi_evi_shuffles = {neuron: [] for neuron in neuron_columns}
    mi_joint_shuffles = {neuron: [] for neuron in neuron_columns}
    for _ in tqdm(range(num_shuffles)):
        shuffled_df = circular_shift_neuron_activity(df, neuron_columns)
        mi_pos = compute_mutual_information(shuffled_df, variable='position')
        mi_evi = compute_mutual_information(shuffled_df, variable='accumulated_evidence')
        mi_joint = compute_mutual_information(shuffled_df, variable=('position', 'evidence'))
        for neuron in neuron_columns:
            mi_pos_shuffles[neuron].append(mi_pos[neuron])
            mi_evi_shuffles[neuron].append(mi_evi[neuron])
            mi_joint_shuffles[neuron].append(mi_joint[neuron])
    mi_mean_sd_pos = {neuron: (np.mean(values), np.std(values)) for neuron, values in mi_pos_shuffles.items()}
    mi_mean_sd_evi = {neuron: (np.mean(values), np.std(values)) for neuron, values in mi_evi_shuffles.items()}
    mi_mean_sd_joint = {neuron: (np.mean(values), np.std(values)) for neuron, values in mi_joint_shuffles.items()}
    return mi_mean_sd_pos, mi_mean_sd_evi, mi_mean_sd_joint

def compute_mutual_information(df, variable='position'):
    """
    Computes mutual information for a specified variable: 'position', 'evidence', or ('position', 'evidence').

    Args:
    - df: DataFrame with neuron activations and contextual variables.
    - variable: The variable(s) to base the mutual information calculation on. 
                Can be 'position', 'evidence', or ('position', 'evidence').

    Returns:
    - mutual_info: Dictionary of mutual information values for each neuron.
    """
    if variable == ('position', 'evidence'):
        group_vars = ['position', 'accumulated_evidence']
    else:  # For 'position' or 'evidence'
        group_vars = [variable]

    # Group by the specified variable(s) and calculate mean neuron activity
    lambda_x = df.groupby(group_vars).mean()

    # Calculate the probability density p(x) or p(e) or p(x, e)
    counts = df.groupby(group_vars).size()
    p_x = counts / counts.sum()

    # Calculate overall lambda
    lambda_total = (lambda_x.multiply(p_x, axis=0)).sum()

    # Calculate mutual information I for each neuron
    mutual_info = {}
    for neuron in [col for col in df.columns if col.startswith('neuron_')]:
        integrand = lambda_x[neuron] * np.log2((lambda_x[neuron] / lambda_total[neuron]).replace([np.inf, -np.inf, np.nan], np.nan) + 1) * p_x
        mutual_info[neuron] = integrand.sum(skipna=True)

    return mutual_info

def randomize_dimension(df, dimension='evidence'):
    """
    Randomizes one dimension (evidence or position) in the dataframe.

    Args:
    - df: DataFrame with columns ['episode', 'step', 'accumulated_evidence', 'position', 'neuron_0', ..., 'neuron_399']
    - dimension: 'evidence' to randomize evidence, 'position' to randomize position

    Returns:
    - df_randomized: DataFrame with the specified dimension randomized
    """
    df_randomized = df.copy()
    
    if dimension == 'evidence':
        # Randomize evidence dimension
        for position in df['position'].unique():
            mask = df['position'] == position
            evidence_values = df.loc[mask, 'accumulated_evidence'].values
            randomized_evidence = np.random.choice(evidence_values, size=mask.sum()) #with replacement
            df_randomized.loc[mask, 'accumulated_evidence'] = randomized_evidence
            
    elif dimension == 'position':
        # Randomize position dimension
        for evidence in df['accumulated_evidence'].unique():
            mask = df['accumulated_evidence'] == evidence
            position_values = df.loc[mask, 'position'].values
            randomized_position = np.random.choice(position_values, size=mask.sum()) #with replacement
            df_randomized.loc[mask, 'position'] = randomized_position
            
    return df_randomized

def average_readouts(df):
    """
    [Averages place cell activations for each neuron across the same episode, accumulated evidence, position.]

    Args:
    - df: Pandas DataFrame containing P-readouts for each neuron, position, and accumulated evidence.

    Returns:
    - df_avg: DataFrame with averaged P-readouts grouped by same posistion and episode.
    """
    # Define the columns to group by
    group_cols = ['position', 'episode']
    
    # Define the P-readout columns to average
    p_readout_cols = [col for col in df.columns if col.startswith('neuron_')]
    avg_cols = p_readout_cols + ['accumulated_evidence', 'L_R_label'] 
    
    # Group by position and accumulated evidence, then calculate the mean for each P-readout
    df_avg = df.groupby(group_cols)[avg_cols].mean().reset_index()
    
    return df_avg

def plot_neuron_evidence_activation_heatmap(df, output_dir=''):
    """
    Plots a heatmap of average neuron activations across distinct accumulated evidence values.

    Args:
    - df: Pandas DataFrame containing neuron activations, 'position', 'episode', and 'accumulated_evidence'.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Extract neuron columns
    neuron_columns = [col for col in df.columns if col.startswith('neuron_')]
    # Group by 'accumulated_evidence' and calculate mean for each neuron
    mean_activations = df.groupby('accumulated_evidence')[neuron_columns].mean()
    
    # Normalize neuron activations within each column to range [0, 1]
    normalized_activations = mean_activations.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # Find the evidence value where each neuron has its peak normalized activity
    peak_evidence_positions = normalized_activations.idxmax().sort_values()

    # Sort neurons by their peak evidence position
    sorted_neurons = peak_evidence_positions.index.tolist()
    sorted_normalized_activations = normalized_activations[sorted_neurons]

    # Transpose for the heatmap (neurons as rows, accumulated evidence as columns)
    sorted_normalized_activations_transposed = sorted_normalized_activations.transpose()
    # Reset the index starting from 1
    sorted_normalized_activations_transposed.index = range(1, len(sorted_normalized_activations_transposed) + 1)

    # Define the ytick labels to show only specific indices
    ylabel_positions = [pos for pos, label in enumerate(sorted_normalized_activations_transposed.index) if pos % 200 == 100]
    # Create the labels list with empty strings for non-multiples of 50
    ylabels = [str(label) for label in ylabel_positions]

    # Plotting the heatmap
    num_neurons = sorted_normalized_activations_transposed.shape[0]
    num_evidence_values = sorted_normalized_activations_transposed.shape[1]


    plt.figure(figsize=(5.5, 12))
    ax = sns.heatmap(sorted_normalized_activations_transposed, cmap='viridis', cbar_kws={'orientation': 'horizontal',
                                                                                         'label': 'Normalized average activity'})
    ax.set_xlabel('Evidence')
    ax.set_ylabel('Neuron')
    ax.set_yticks(ylabel_positions)
    ax.set_yticklabels(ylabels, rotation=0)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    # Only show yticks for the specified indices
    plot_path = os.path.join(output_dir, 'evidence_neuron_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in interactive environments
    print(f'Saved evidence heatmap by neurons to {plot_path}')

def create_position_preference_heatmaps(df, preference_file_path, output_dir=''):
    # Load the neuron preference data
    with open(preference_file_path, 'rb') as f:
        neuron_preference = pickle.load(f)
    
    df_left = df[df['L_R_label'] == 0]
    df_right = df[df['L_R_label'] == 1]

    # Aggregate neuron activity by position for each neuron and normalize
    neuron_columns = [col for col in df.columns if col.startswith('neuron_')]
    left_neuron_activity = df_left.groupby('position')[neuron_columns].mean()
    left_normalized_neuron_activity = left_neuron_activity.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
    right_neuron_activity = df_right.groupby('position')[neuron_columns].mean()
    right_normalized_neuron_activity = right_neuron_activity.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    left_preferring_neurons = [neuron for neuron, pref in neuron_preference.items() if pref == 'left']
    left_peak_activities = left_normalized_neuron_activity[left_preferring_neurons].idxmax()
    sorted_left_preferring_neurons = left_peak_activities.sort_values().index

    # Select and sort right-preferring neurons based on peak activity in right-choice trials
    right_preferring_neurons = [neuron for neuron, pref in neuron_preference.items() if pref == 'right']
    right_peak_activities = right_normalized_neuron_activity[right_preferring_neurons].idxmax()
    sorted_right_preferring_neurons = right_peak_activities.sort_values().index

    # Select and sort non-preferring neurons based on peak activity in left-choice trials
    non_preferring_neurons = [neuron for neuron, pref in neuron_preference.items() if pref == 'non-preferring']
    non_peak_activities = left_normalized_neuron_activity[non_preferring_neurons].idxmax()
    sorted_non_preferring_neurons = non_peak_activities.sort_values().index
    
    insignificant_neurons = [neuron for neuron, pref in neuron_preference.items() if pref == 'insignificant']
    
    print(f'#left: {len(left_preferring_neurons)}, #right:{len(right_preferring_neurons)}, #non-perf:{len(non_preferring_neurons)}, #insign:{len(insignificant_neurons)}')

    # Plot the heatmaps
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Left-preferring neurons heatmap for left and right choice trials
    sns.heatmap(left_normalized_neuron_activity.loc[:, sorted_left_preferring_neurons].T, ax=axes[0, 0], cmap=cmap)
    sns.heatmap(right_normalized_neuron_activity.loc[:, sorted_left_preferring_neurons].T, ax=axes[0, 1], cmap=cmap)
    # axes[0, 0].set_title('Left-preferring neurons - Left-choice trials')
    # axes[0, 1].set_title('Left-preferring neurons - Right-choice trials')

    # Right-preferring neurons heatmap for left and right choice trials
    sns.heatmap(left_normalized_neuron_activity.loc[:, sorted_right_preferring_neurons].T, ax=axes[1, 0], cmap=cmap)
    sns.heatmap(right_normalized_neuron_activity.loc[:, sorted_right_preferring_neurons].T, ax=axes[1, 1], cmap=cmap)
    # axes[1, 0].set_title('Right-preferring neurons - Left-choice trials')
    # axes[1, 1].set_title('Right-preferring neurons - Right-choice trials')

    # Non-preferring neurons heatmap for left and right choice trials
    sns.heatmap(left_normalized_neuron_activity.loc[:, sorted_non_preferring_neurons].T, ax=axes[2, 0], cmap=cmap)
    sns.heatmap(right_normalized_neuron_activity.loc[:, sorted_non_preferring_neurons].T, ax=axes[2, 1], cmap=cmap)
    # axes[2, 0].set_title('Non-preferring neurons - Left-choice trials')
    # axes[2, 1].set_title('Non-preferring neurons - Right-choice trials')
    
    # Customize y-axis labels to show neuron counts every 100 neurons
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
        neuron_count = int(ax.get_ylim()[0]) # Get the number of neurons (rows) in the heatmap
        neuron_indices = list(np.arange(100, neuron_count+1, 100))  # Create an array of indices every 100 neurons
        # Ensure the last neuron index is included if it's not already in the list
        if neuron_indices == []:
            neuron_indices.append(neuron_count)
        
        ax.set_yticks(neuron_indices)  # Set the y-axis ticks to these indices
        ax.set_yticklabels(neuron_indices)  # Use these indices as the y-axis labels
        ax.set_xticks([0,13,19])
        ax.set_xticklabels([0, 13, 19])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

    # Adjust the layout
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'position_preference_neuron_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Saved position heatmap by neurons to {plot_path}')

def plot_3d_neuron_activation(df, neuron_column_index, output_dir=''):
    """
    Creates a 3D plot of average neuron activation for a specified neuron column.

    Args:
    - df: Pandas DataFrame with neuron activations, 'position', 'episode', and 'accumulated_evidence'.
    - neuron_column: neuron column index to plot.
    """
    # Group by 'position' and 'accumulated_evidence' and calculate mean activation for the neuron
    grouped = df.groupby(['position', 'accumulated_evidence'])['neuron_' + str(neuron_column_index)].mean().reset_index()

    # Prepare data for interpolation
    X = grouped['accumulated_evidence'].values
    Y = grouped['position'].values
    Z = grouped['neuron_' + str(neuron_column_index)].values
    
    # Create grid to interpolate onto
    xi = np.linspace(X.min(), X.max(), 200)  # finer grid on x-axis, evidence
    yi = np.linspace(Y.min(), Y.max(), 200)  # finer grid on z-axis, position
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate Y values onto the new grid
    zi = griddata((X, Y), Z, (xi, yi), method='cubic')
    zi = np.clip(zi, 0, zi.max())  # Ensure no negative values

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', linewidth=0, antialiased=True)

    # Add color bar which maps values to colors
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Average Activation')
    start, end = int(Y.min()), int(Y.max())
    middle = (start + end) // 2
    ax.set_yticks([start, middle, end])
 
    # Label the axes
    ax.set_xlabel('Accumulated Evidence')
    ax.set_ylabel('Position')
    ax.set_zlabel('Mean Activation for neuron ' + str(neuron_column_index))
    
    plot_path = os.path.join(output_dir, f'3d_neurons/3d_activation_neuron{neuron_column_index}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in interactive environments
    print(f'Saved 3d neuron plot for neuron {neuron_column_index} to {plot_path}')

def plot_2d_neuron_heatmap(df, neuron_column_index, output_dir='', normalize=True):
    """
    Plots a heatmap for a specified neuron column showing activation as a function
    of position (y-axis) and accumulated evidence (x-axis).

    Args:
    - df: Pandas DataFrame with neuron activations, 'position', 'episode', and 'accumulated_evidence'.
    - neuron_column_index: Index of the neuron column to plot.
    - output_dir: Directory where the plot will be saved.
    """
    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Group by 'position' and 'accumulated_evidence' and calculate mean activation for the neuron
    neuron_col = 'neuron_' + str(neuron_column_index)
    df['accumulated_evidence'] = df['accumulated_evidence'].astype(int)
    averaged_activations = df.groupby(['position', 'accumulated_evidence'])[neuron_col].mean().reset_index()
    
    if normalize:
        # Normalize activations before pivoting
        max_act = averaged_activations[neuron_col].max()
        min_act = averaged_activations[neuron_col].min()
        averaged_activations[neuron_col] = (averaged_activations[neuron_col] - min_act) / (max_act - min_act)

    # Pivot the DataFrame for the heatmap
    activation_pivot = averaged_activations.pivot(index='position', columns='accumulated_evidence', values=neuron_col)
    
    # Sort the columns (accumulated evidence values) in increasing order for the x-axis
    activation_pivot = activation_pivot.reindex(sorted(activation_pivot.columns), axis=1)
    
    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(activation_pivot, cmap='viridis', cbar_kws={'label': 'Average Activation'})
    ax.invert_yaxis()  # Optional: Invert the y-axis so that position 0 is at the bottom of the heatmap
    plt.title(f'Heatmap of Average Activation for Neuron {neuron_column_index}')
    plt.xlabel('Accumulated Evidence')
    plt.ylabel('Position')
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'2d_neurons/heatmap_neuron_{neuron_column_index}_nodivision.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()  # Close the figure to prevent it from displaying in interactive environments
    # print(f'Saved heatmap for neuron {neuron_column_index} to {plot_path}')

def plot_all_neurons_2d_heatmap_with_smoothing(df, 
                                output_dir='', 
                                normalize_stage='after', 
                                method='mean',
                                first_smooth_sigma=1,
                                second_smooth_sigma=2,
                                threshold_factor=2,
                                apply_first_smoothing=True,
                                apply_second_smoothing=True):
    """
    Plots a heatmap for every neuron column showing activation as a function
    of position (y-axis) and accumulated evidence (x-axis), divided by the 
    average occupancy of the state across episodes.
    
    method can be 
    - 'sum'
    - 'mean'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    # Calculate the occupancy of each (position, evidence) state 
    neuron_cols = [col for col in df.columns if col.startswith('neuron_')]
    
    smoothed_cols_dict = {}
    # 1) (Optional) First smoothing and thresholding across time
    if apply_first_smoothing:
        for neuron_col in neuron_cols:
            smoothed_vals = smooth_and_threshold_in_time(
                df[neuron_col].values,
                sigma=first_smooth_sigma,
                threshold_factor=threshold_factor
            )
            smoothed_cols_dict[neuron_col + '_smooth'] = smoothed_vals
    else:
        # If not applying smoothing, just copy the original signals
        for neuron_col in neuron_cols:
            smoothed_cols_dict[neuron_col + '_smooth'] = df[neuron_col].values
    df = pd.concat([df, pd.DataFrame(smoothed_cols_dict, index=df.index)], axis=1)
    df = df.copy()  # defragment to be extra safe

    # For each neuron, calculate mean activation over average occupancy
    for i, neuron_col in enumerate(neuron_cols):
        smoothed_col = neuron_col + '_smooth'
        # Merge neuron activation with occupancy
        if method == 'mean':
            # define each combo of E x Y as a group, then compute the mean specified by neuron_col within the group
            neuron_data = df.groupby(['position', 'accumulated_evidence'])[smoothed_col].mean().reset_index()
            neuron_data['weighted_activation'] = neuron_data[smoothed_col]
        elif method == 'sum':
            assert 'not implemented'

        activation_pivot = neuron_data.pivot_table(index='position',
                                             columns='accumulated_evidence',
                                             values=smoothed_col,
                                             fill_value=0.0)
        if apply_second_smoothing:
            pivot_array = activation_pivot.values
            pivot_smoothed = gaussian_filter(pivot_array, sigma=second_smooth_sigma)
            # Convert back to DataFrame
            activation_pivot = pd.DataFrame(pivot_smoothed,
                                            index=activation_pivot.index,
                                            columns=activation_pivot.columns)

        pivot_min = activation_pivot.values.min()
        pivot_max = activation_pivot.values.max()
        activation_pivot = (activation_pivot - pivot_min) / (pivot_max - pivot_min + 1e-9)

        # Plotting the heatmap
        plt.figure()
        ax = sns.heatmap(activation_pivot, cmap='jet', square=True, cbar_kws={'label': 'Average Activation'})
        ax.invert_yaxis()  # Invert the y-axis
        plt.title(f'{neuron_col} (smoothed)' if apply_first_smoothing else neuron_col)
        ax.set_yticks([0, 19])
        ax.set_yticklabels([0, 19]) 
        ax.set_xticks([0, 5, 10])            # indices for columns of interest
        ax.set_xticklabels([-5, 0, 5])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)

        # Save the plot
        plot_path = os.path.join(output_dir, f'{neuron_col}_heatmap.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close the figure
        # print(f'Saved heatmap for {neuron_col} to {plot_path}')

def plot_all_neurons_2d_heatmap(df, output_dir='', normalize_stage='after', method='sum'):
    """
    Plots a heatmap for every neuron column showing activation as a function
    of position (y-axis) and accumulated evidence (x-axis), divided by the 
    average occupancy of the state across episodes.
    
    method can be 
    - 'sum'
    - 'mean'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate the occupancy of each (position, evidence) state 
    total_episodes = df['episode'].nunique()
    occupancy = df.groupby(['position', 'accumulated_evidence']).size().reset_index(name='occupancy')

    # For each neuron, calculate mean activation over average occupancy
    for neuron_col in tqdm([col for col in df.columns if col.startswith('neuron_')], 
                        desc="Plotting heatmap for individual neurons..."):
        # Merge neuron activation with occupancy
        if method == 'mean':
            # define each combo of E x Y as a group, then compute the mean specified by neuron_col within the group
            neuron_data = df.groupby(['position', 'accumulated_evidence'])[neuron_col].mean().reset_index()
            neuron_data['weighted_activation'] = neuron_data[neuron_col]
        elif method == 'sum':
            assert 'not implemented'
        neuron_data[neuron_col] = (neuron_data[neuron_col] - neuron_data[neuron_col].min()) / (neuron_data[neuron_col].max() - neuron_data[neuron_col].min())

        activation_pivot = neuron_data.pivot(index='position', columns='accumulated_evidence', values=neuron_col)
        plt.figure()
        ax = sns.heatmap(activation_pivot, cmap='jet', square=True, cbar_kws={'label': 'Average Activation'})
        ax.invert_yaxis()  # Invert the y-axis
        plt.title(f'{neuron_col}')
        ax.set_yticks([0, 19])
        ax.set_yticklabels([0, 19]) 
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        # plt.xlabel('Accumulated Evidence')
        # plt.ylabel('Position')
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'{neuron_col}_heatmap.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        # print(f'Saved heatmap for {neuron_col} to {plot_path}')

def plot_neuron_activity_by_position_schematic(df, neuron_column_index, output_dir=''):
    """
    Plots position vs. activity for a specific neuron.

    Args:
    - df: Pandas DataFrame containing neuron activations along with 'position'.
    - neuron_column_index: Index of the neuron column to plot.
    - output_dir: Directory where the plot will be saved.
    """
    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

   # Prepare the data
    neuron_col = 'neuron_' + str(neuron_column_index)
    # Group by 'position' and calculate the average activation for each position
    average_activations = df.groupby('position')[neuron_col].mean().reset_index()

    # Extract x and y values
    x_values = average_activations['position']
    y_values = average_activations[neuron_col]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label=f'Average Neuron {neuron_column_index} Activity')
    plt.xlabel('Position')
    plt.ylabel('Average Activity')
    plt.title(f'Position vs. Average Activity Plot for Neuron {neuron_column_index}')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(output_dir, f'average_neuron_activity_{neuron_column_index}.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to prevent it from displaying in interactive environments
    print(f'Saved plot for average neuron activity {neuron_column_index} to {plot_path}')

def plot_all_neuron_activity_by_position_schematic(df, output_dir=''):
    """
    Plots position vs. average activity for all neurons in the DataFrame.

    Args:
    - df: Pandas DataFrame containing neuron activations along with 'position'.
    - output_dir: Directory where the plots will be saved.
    """
    # Loop over each neuron column
    for neuron_col in [col for col in df.columns if col.startswith('neuron_')]:  # Assuming there are 400 neurons indexed from 0 to 399
        neuron_index = int(neuron_col.split('_')[1])
        plot_neuron_activity_by_position_schematic(df, neuron_index, output_dir)

def plot_peak_activity_heatmap(df, output_dir=''):
    """
    Plots a heatmap showing the number of neurons with peak activity at each
    (position, accumulated evidence) pair.

    Args:
    - df: Pandas DataFrame with neuron activations, 'position', 'episode', and 'accumulated_evidence'.
    - output_dir: Directory where the plot will be saved.
    """
    # Ensure the output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find peak activity for each neuron
    neuron_columns = [col for col in df.columns if col.startswith('neuron_')]
    peak_activities_list = []
    
    for neuron_col in neuron_columns:
        averaged_activations = df.groupby(['position', 'accumulated_evidence'])[neuron_col].mean().reset_index()
        peak_activity = averaged_activations.loc[averaged_activations[neuron_col].idxmax()]
        peak_activities_list.append({'position': peak_activity['position'], 'accumulated_evidence': peak_activity['accumulated_evidence'], 'count': 1})

    peak_activities = pd.DataFrame(peak_activities_list)
    # Aggregate counts of peak activities
    peak_counts = peak_activities.groupby(['position', 'accumulated_evidence']).count().reset_index()
    peak_counts_pivot = peak_counts.pivot(index='position', columns='accumulated_evidence', values='count').fillna(0)
    
    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(peak_counts_pivot, cmap='viridis', cbar_kws={'label': 'Number of Neurons with Peak Activity'}, annot=True, fmt=".0f")
    ax.invert_yaxis()  # Invert the y-axis to have position 0 at the bottom
    plt.title('Heatmap of Neuron Peak Activity Across Position and Accumulated Evidence')
    plt.xlabel('Accumulated Evidence')
    plt.ylabel('Position')
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'neuron_peak_activity_heatmap.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Saved neuron peak activity heatmap to {plot_path}')

def analyze_neural_evidence_encoding(df, path, load_shuffle=True, evidence_levels=range(-5, 6)):
    neuron_columns = [col for col in df.columns if col.startswith('neuron_')]
    
    # Load shuffled metrics if available
    if load_shuffle:
        shuffled_metrics_path = os.path.join(path, 'shuffled_metrics_evidence.pkl')
        with open(shuffled_metrics_path, 'rb') as f:
            mi_mean_sd_evi = pickle.load(f)
    else:
        mi_mean_sd_evi = compute_mean_mutual_information_evidence_shuffled(df, neuron_columns, num_shuffles=10)
        with open(os.path.join(path, 'shuffled_metrics_evidence.pkl'), 'wb') as f:
            pickle.dump(mi_mean_sd_evi, f)
        print('Computed and saved shuffled mi wrt evidence...')

    # Calculate mutual information for each evidence level
    mi_evidence = {}
    for level in evidence_levels:
        df_level = df[df['accumulated_evidence'] == level]
        mi_evidence[level] = compute_mutual_information(df_level, variable='accumulated_evidence')
    
    # Analyze evidence encoding based on mutual information significance
    neuron_evidence_encoding = {}
    for neuron in neuron_columns:
        significant_levels = []
        for level in evidence_levels:
            mi_value = mi_evidence[level].get(neuron, 0)
            mean, sd = mi_mean_sd_evi[neuron]
            if mi_value > mean + 2 * sd:
                significant_levels.append(level)
        neuron_evidence_encoding[neuron] = significant_levels
    
    # Save the analysis results
    encoding_path = os.path.join(path, 'neuron_evidence_encoding.pkl')
    with open(encoding_path, 'wb') as f:
        pickle.dump(neuron_evidence_encoding, f)
    
    print(f'Saved neuron evidence encoding analysis to {encoding_path}')

def analyze_neural_preference_and_encoding(df, path, load_shuffle=True, num_shuffles=3):
    neuron_columns = [col for col in df.columns if col.startswith('neuron_')]
    if path and not os.path.exists(path):
        os.makedirs(path)
        
    # 1. Shuffle dataset and compute mean and sd of mutual information w.r.t. 'position', or jointly
    # Adjusting for computing mean and sd across shuffles for each neuron
    if not load_shuffle:
        mi_mean_sd_pos, mi_mean_sd_evi, mi_mean_sd_joint = compute_mean_mutual_information_shuffled(df, neuron_columns, num_shuffles=num_shuffles)
        with open(os.path.join(path, 'shuffled_metrics_evidence.pkl'), 'wb') as f:
            pickle.dump(mi_mean_sd_evi, f)
        with open(os.path.join(path, 'shuffled_metrics_position.pkl'), 'wb') as f:
            pickle.dump(mi_mean_sd_pos, f)
        with open(os.path.join(path, 'shuffled_metrics_joint.pkl'), 'wb') as f:
            pickle.dump(mi_mean_sd_joint, f)
        print('Computed and saved shuffled mi wrt position and joint encoding...')
    else:
        position_file_path = os.path.join(path, 'shuffled_metrics_position.pkl')
        evidence_file_path = os.path.join(path, 'shuffled_metrics_evidence.pkl')
        joint_file_path = os.path.join(path, 'shuffled_metrics_joint.pkl')

        # Load mi_mean_sd_position
        with open(evidence_file_path, 'rb') as f:
            mi_mean_sd_evi = pickle.load(f)

        with open(position_file_path, 'rb') as f:
            mi_mean_sd_pos = pickle.load(f)

        # Load mi_mean_sd_joint
        with open(joint_file_path, 'rb') as f:
            mi_mean_sd_joint = pickle.load(f)
    
    # 2. Split df into left and right choice trials
    df_left = df[df['L_R_label'] == 0]
    df_right = df[df['L_R_label'] == 1]
    
    # 3. Compute mutual information for left and right choice trials
    mi_left = compute_mutual_information(df_left, 'position')
    mi_right = compute_mutual_information(df_right, 'position')
    pd.DataFrame([mi_left], index=['mi_left']).T.to_csv(os.path.join(path, 'mutual_info_position_left.csv'))
    pd.DataFrame([mi_right], index=['mi_right']).T.to_csv(os.path.join(path, 'mutual_info_position_right.csv'))
    print('Saved mutual info wrt position...')
    # 4. Determine neuron preference based on mutual information significance
    neuron_preference = {}
    for neuron in neuron_columns:
        if mi_left[neuron] > mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1] and mi_right[neuron] <= mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1]:
            neuron_preference[neuron] = 'left'
        elif mi_right[neuron] > mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1] and mi_left[neuron] <= mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1]:
            neuron_preference[neuron] = 'right'
        elif mi_left[neuron] > mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1] and mi_right[neuron] > mi_mean_sd_pos[neuron][0] + 2 * mi_mean_sd_pos[neuron][1]:
            neuron_preference[neuron] = 'non-preferring'
        else:
            neuron_preference[neuron] = 'insignificant'
    with open(os.path.join(path, 'neuron_preference.pkl'), 'wb') as f:
        pickle.dump(neuron_preference, f)
    print('Analyzed neuron preference...')
    
    # 5. Randomize evidence and position for mutual information computation
    df_random_evidence = randomize_dimension(df, 'evidence')
    df_random_position = randomize_dimension(df, 'position')
    
    # 6. Compute mutual information for joint encoding
    mi_joint = compute_mutual_information(df, ('position', 'evidence'))
    mi_random_evidence = compute_mutual_information(df_random_evidence, ('position', 'evidence'))
    mi_random_position = compute_mutual_information(df_random_position, ('position', 'evidence'))
    # Save results
    pd.DataFrame.from_dict(mi_joint, orient='index', columns=['MI']).to_csv(os.path.join(path, 'mutual_info_joint.csv'))
    pd.DataFrame.from_dict(mi_random_evidence, orient='index', columns=['MI']).to_csv(os.path.join(path, 'mutual_info_joint_random_evidence.csv'))
    pd.DataFrame.from_dict(mi_random_position, orient='index', columns=['MI']).to_csv(os.path.join(path, 'mutual_info_joint_random_position.csv'))

    print('Saved mutual info wrt joint position and evidence...')
    joint_values = [v for k, v in mi_joint.items()]
    random_evidence_values = [mi_random_evidence[k] for k in mi_joint.keys()]
    random_position_values = [mi_random_position[k] for k in mi_joint.keys()]

    # Create scatter plot for mi_joint vs. mi_random_evidence
    def round_to_nice(value, base=0.5):
        return base * round(value / base)

    # First scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(joint_values, random_evidence_values, alpha=0.5, color='#502F74', edgecolors='none')

    # Calculate the nice upper bound for ticks (no limiting the data)
    upperbound = max(max(joint_values), max(random_evidence_values))
    nice_upperbound = round_to_nice(upperbound)  # Round to the nearest multiple of 0.5

    # Plot y=x reference line
    lowerbound = 0
    plt.plot([lowerbound, nice_upperbound], [lowerbound, nice_upperbound], '--', color='grey')

    plt.xlabel('E x Y', fontsize=24)  # Large label font size
    plt.ylabel('$R_E$ x Y', fontsize=24)  # Large label font size

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set xticks and yticks to show only 0 and nice upper bound, without cutting data
    ax.set_xticks([0, nice_upperbound])
    ax.set_yticks([0, nice_upperbound])

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set aspect ratio and save
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plot_path = os.path.join(path, 'joint_by_random_evidence.pdf')
    plt.savefig(plot_path, bbox_inches='tight')

    # Second scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(joint_values, random_position_values, alpha=0.5, color='#502F74', edgecolors='none')

    # Calculate the nice upper bound for ticks (no limiting the data)
    upperbound = max(max(joint_values), max(random_position_values))
    nice_upperbound = round_to_nice(upperbound)

    # Plot y=x reference line
    lowerbound = 0
    plt.plot([lowerbound, nice_upperbound], [lowerbound, nice_upperbound], '--', color='grey')

    plt.xlabel('E x Y', fontsize=24)  # Large label font size
    plt.ylabel('E x $R_Y$', fontsize=24)  # Large label font size

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set xticks and yticks to show only 0 and nice upper bound, without cutting data
    ax.set_xticks([0, nice_upperbound])
    ax.set_yticks([0, nice_upperbound])

    # Set tick label font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set aspect ratio and save
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plot_path = os.path.join(path, 'joint_by_random_position.pdf')
    plt.savefig(plot_path, bbox_inches='tight')
    # 7. Determine encoding based on mutual information significance
    neuron_encoding = {}
    figf = {'joint':[], 'position':[], 'evidence':[], 'none':[]}
    for neuron in neuron_columns:
        if mi_joint[neuron] > mi_mean_sd_joint[neuron][0] + 2 * mi_mean_sd_joint[neuron][1]:
            neuron_encoding[neuron] = 'joint'
            figf['joint'].append(neuron)
        elif mi_random_evidence[neuron] > mi_mean_sd_joint[neuron][0] + 2 * mi_mean_sd_joint[neuron][1]:
            neuron_encoding[neuron] = 'position'
            figf['position'].append(neuron)
        elif mi_random_position[neuron] > mi_mean_sd_joint[neuron][0] + 2 * mi_mean_sd_joint[neuron][1]:
            neuron_encoding[neuron] = 'evidence'
            figf['evidence'].append(neuron)
        else:
            neuron_encoding[neuron] = 'none'
            figf['none'].append(neuron)
    total_neurons = len(neuron_columns)
    print('p(E x Y) =', len(figf['joint'])/total_neurons)
    print('p(R_E x Y) =', len(figf['position'])/total_neurons)
    print('p(E x R_Y) =', len(figf['evidence'])/total_neurons)
    print('p(none) =', len(figf['none'])/total_neurons)
        
    with open(os.path.join(path, 'neuron_encoding.pkl'), 'wb') as f:
        pickle.dump(neuron_encoding, f)
    print('Analyzed neuron encoding...')
    
    print('Exiting...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tower Task Testing with GridWrapper')
    parser.add_argument('--model_type', type=str, required=False, help='choose from M1-M5')
    parser.add_argument('--num_episodes', type=int, required=False, default=1000, help='Num of episodes saved')
    args = parser.parse_args()

    """Data processing"""
    if args.model_type == 'M1':
        format_path = '...' # TODO: user enters
    elif args.model_type == 'M2':
        format_path = '...' # TODO: user enters
    elif args.model_type == 'M3':
        format_path = '...' # TODO: user enters
    elif args.model_type == 'M4':
        format_path = '...' # TODO: user enters
    elif args.model_type == 'M5':
        # for example...
        format_path = 'M5/mlp32/p/no_sensory/HaSH_star/seq20/maxTower5/RNN32/7_8_11/0.0005/trial_debug/800/' 

    save_path = os.path.join(DATA_DIR, format_path)
    plot_path = os.path.join(FIGURE_DIR, format_path)
    load_file = os.path.join(save_path, f'{args.num_episodes}trials.mat')
    
    os.makedirs(plot_path, exist_ok=True)
    print('Loading from path', load_file)
    print('Saving plots to', plot_path)
    
    # load the .mat data file
    mat = loadmat(load_file)
    column_labels = ['episode', 'step', 'position',
                     'current_evidence_axis', 'accumulated_evidence',
                     'total_evidence', 'current_action', 
                     'ground_truth_action', 'L_R_label', 'current_spatial_axis',
                     'p_L', 'p_R', 'p_F', # you can ignore this row; we don't use them in this script
                     'final_success']

    info_df = pd.DataFrame(mat['info'], columns=column_labels)
    place_cell_df = pd.DataFrame(mat['ps'], columns=[f'neuron_{i}' for i in range(mat['ps'].shape[1])])
    df = pd.concat([info_df, place_cell_df], axis=1)
    print('Loaded df of shape', df.shape)

    # Exclude rows where final_success is 0
    df = df[df['final_success'] != 0]
    print('Filtered into df of shape', df.shape)
    df = average_readouts(df)
    print('Averaged out same (episode, position)', df.shape)
    
    """ 1) plot, for each neuron, normalized average activity in E x Y space """
    plot_all_neurons_2d_heatmap(df, output_dir=os.path.join(plot_path, '2d_neurons'), method='mean')

    # OPTIONAL: YOU MAY ALSO SMOOTH ACTIVATION HEATMAP USING (appeared in ICML rebuttal stage and added to appendices)
    ## `plot_all_neurons_2d_heatmap_with_smoothing`
    
    """ 2) plot, with all neuron, normalized activity grouped by evidence value """
    plot_neuron_evidence_activation_heatmap(df, output_dir=os.path.join(plot_path, "evidence_field"))

    """ 3) compute M.I. wrt evi, pos, and joint evi + pos; the shuffled result is saved to .pkl for reuse
    ## Determine neuron preference based on mutual information significance
    ## Plot joint wrt one variable being randomnized """
    analyze_neural_preference_and_encoding(df, os.path.join(plot_path, 'mutual_information'), load_shuffle=False, num_shuffles=10)
    
    """3.5) Using computed M.I. wrt evi from above, check each neuron's evidence preference"""
    analyze_neural_evidence_encoding(df, os.path.join(plot_path, 'mutual_information'), load_shuffle=True)

    """Load neuron_encoding assignment"""
    encoding_file_path = os.path.join(plot_path, "mutual_information", "neuron_encoding.pkl")

    # Load neuron_preference
    mi_dir = os.path.join(plot_path, "mutual_information")
    preference_file_path = os.path.join(mi_dir, "neuron_preference.pkl")
    evidence_file_path   = os.path.join(mi_dir, "neuron_evidence_encoding.pkl")
    create_position_preference_heatmaps(df, preference_file_path, mi_dir)

    # OPTIONAL: Plot activation heatmap individually in a loop; this overlaps plot_all_neurons_2d_heatmap with more customization for selected neurons.
    # for i in range(400):
        # plot_2d_neuron_heatmap(df, i, plot_path)
        # plot_3d_neuron_activation(df, i, plot_path)

    