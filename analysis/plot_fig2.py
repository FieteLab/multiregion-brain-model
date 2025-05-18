import pickle
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import sem
import torch
import os
import sys
from pathlib import Path
import matplotlib.lines as mlines
import seaborn as sns
sns.set_palette("colorblind")

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from towertask.config import DATA_DIR, get_figure_path, FIGURE_DIR
from towertask.utils import calculate_ema, calculate_cumulative_success, calculate_moving_window_success_rate, calculate_grouped_success_rate, variance_in_success_rate, early_vs_late_success

def moving_average(data, window_size=50):
    """Applies a moving average to smooth the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def calculate_windowed_exploration(exploration_times, window_size=10000):
    windowed_exploration = []
    cumulative_sum = 0
    for i, time in enumerate(exploration_times):
        cumulative_sum += time
        if i >= window_size:
            cumulative_sum -= exploration_times[i - window_size]
        windowed_exploration.append(cumulative_sum / min(i + 1, window_size))
    return windowed_exploration

# Directory dictionary for different models with multiple runs
lr = 5e-05
directory_dic = {
    'M5: joint g, mix p': [
        # TODO: trial 1 directory, for example: 
        'M5/mlp32/p/no_sensory/HaSH_star/seq20/maxTower5/RNN32/7_8_11/0.0005/trial_debug/800',
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M0: RNN w/ sensory':[
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M0+: RNN w/ sensory,\npos. vel. & evi. vel.': [
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M1: pos. only g, nonmix p': [
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M2: pos. only g, mix p': [
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M3: joint g, nonmix p': [
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
    'M4: disjoint g, mix p': [
        '', # TODO: trial 1 directory
        '', # TODO: trial 2 directory
        '' # TODO: trial 3 directory
    ],
}

default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Define custom colors
custom_colors = {
    'M0: RNN w/ sensory': None,  # Use default color
    'mini-M0: RNN w/ sensory': None, #'#1f77b4',
    'M0+: RNN w/ sensory,\npos. vel. & evi. vel.': None, #'black',  # Black for this key
    'mini-M0+: RNN w/ sensory':None, # 'black',
    'M1: pos. only g, nonmix p': None,  # Use default color
    'M2: pos. only g, mix p': None,  # Use default color
    'M3: joint g, nonmix p': None,  # Use default color
    'M4: disjoint g, mix p': None,  # Use default color
    'M5: joint g, mix p': None # '#8c564b',  # Use default color
}

custom_linestyles = {
    'M0: RNN w/ sensory':'solid',
    'mini-M0: RNN w/ sensory': 'solid',
    'M0+: RNN w/ sensory,\npos. vel. & evi. vel.': 'solid',
    'mini-M0+: RNN w/ sensory': 'solid',
    'M1: pos. only g, nonmix p': 'solid',
    'M2: pos. only g, mix p': 'solid',
    'M3: joint g, nonmix p': 'solid',
    'M4: disjoint g, mix p': 'solid',
    'M5: joint g, mix p': 'solid',
    'Random chance in A\nExpected #steps in B':'dashed'
}

keys = list(custom_colors.keys())
cycle_colors = iter(default_color_cycle)
for key in keys:
    if custom_colors[key] is None:
        custom_colors[key] = next(cycle_colors)

all_episode_successes = {}
all_exploration_times = {}
all_reward = {}
initial_cutoff = 100
# Iterate over each model and load data from multiple runs
for name, directories in directory_dic.items():
    model_successes = []
    model_exploration_times = []
    model_reward = []
    grand_path = FIGURE_DIR
    for directory in directories:
        # print('directory',directory)
        checkpoint = torch.load(os.path.join(grand_path, directory, 'checkpoint.pth'), map_location=torch.device('cpu'))

        with open(os.path.join(grand_path, directory, 'final_decision.pkl'), 'rb') as file:
            final_decision_data = pickle.load(file)
     
        model_exploration_times.append(final_decision_data['exploration_times'][initial_cutoff:])
        model_successes.append(checkpoint['episode_successes'][initial_cutoff:])
        
    # Store the episode successes and exploration times for each model
    all_episode_successes[name] = model_successes
    all_exploration_times[name] = model_exploration_times

# Calculate minimum length across all successes and exploration times
min_length_success = min(min(len(s) for s in successes) for successes in all_episode_successes.values())
min_length_exploration = min(min(len(e) for e in exploration_times) for exploration_times in all_exploration_times.values())

# Define the figure and axes for the two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

# Plot for cumulative success rates on the first axis (axs[0])
for name, successes in all_episode_successes.items():
    truncated_successes = [s[:min_length_success] for s in successes]
    truncated_successes = np.array(truncated_successes)

    cumulative_successes = [calculate_windowed_exploration(s, window_size=5000) for s in truncated_successes]
    
    mean_successes = np.mean(cumulative_successes, axis=0)
    std_successes = np.std(cumulative_successes, axis=0)
    episodes = np.arange(initial_cutoff, len(mean_successes) + initial_cutoff)
    
    color = custom_colors[name]
    linestyle = custom_linestyles[name]
    axs[0].plot(episodes, mean_successes * 100, label=name, color=color, linestyle=linestyle,)
    axs[0].fill_between(episodes, (mean_successes - std_successes) * 100, 
                        (mean_successes + std_successes) * 100, alpha=0.2, color=color,
                        edgecolor=None)

axs[0].set_xlabel('Episode', fontsize=20)
axs[0].set_ylabel('Cumulative Success Rate (%)', fontsize=20)
axs[0].axhline(y=50, color='gray', linestyle='--', label='Random chance in A\nExpected #steps in B')
axs[0].set_ylim(45, 100)
axs[0].set_xlim(0, 17500)
axs[0].set_xticks([100, 17400])
axs[0].tick_params(labelsize=18)
axs[0].set_title('Cum. success rate in training', fontsize=18, fontweight='bold')

# Add subplot label A outside the figure frame, larger font size
axs[0].annotate('A', xy=(-0.2, 1.05), xycoords='axes fraction', fontsize=40, fontweight='bold')

# Plot for smoothed exploration times on the second axis (axs[1])
for name, exploration_times in all_exploration_times.items():
    truncated_exploration_times = [e[:min_length_exploration] for e in exploration_times]
    
    # smoothed_exploration_times = [moving_average(e, window_size=10000) for e in truncated_exploration_times]
    smoothed_exploration_times = [calculate_windowed_exploration(e) for e in truncated_exploration_times]
    smoothed_exploration_times = np.array(smoothed_exploration_times)

    mean_exploration = np.mean(smoothed_exploration_times, axis=0)
    std_exploration = np.std(smoothed_exploration_times, axis=0)
    episodes = np.arange(initial_cutoff, len(mean_exploration) + initial_cutoff)
    
    color = custom_colors[name]
    linestyle = custom_linestyles[name]
    axs[1].plot(episodes, mean_exploration, label=name, color=color, linestyle=linestyle)
    axs[1].fill_between(episodes, mean_exploration - std_exploration, mean_exploration + std_exploration, alpha=0.2, color=color,
                        edgecolor=None,
                        )

axs[1].set_xlabel('Episode', fontsize=20)
axs[1].set_ylabel('Exploration Time (steps/episode)', fontsize=18)
axs[1].axhline(y=19, color='gray', linestyle='--', label='Expected steps (y=19)')
axs[1].set_xticks([100, 17400])
axs[1].tick_params(labelsize=18)
axs[1].set_title('Steps spent per episode in training', fontsize=18, fontweight='bold')

axs[1].set_ylim(17, 45)
axs[1].set_xlim(0, 17500)

# Add subplot label B outside the figure frame, larger font size
axs[1].annotate('B', xy=(-0.2, 1.05), xycoords='axes fraction', fontsize=40, fontweight='bold')

# Move the legend closer to the second plot and remove the "Legend" title
handles, labels = axs[0].get_legend_handles_labels()
# legend_handles = [mlines.Line2D([], [], color=h.get_color(), linewidth=4) for h in handles]
legend_handles = [
    mlines.Line2D([], [], color=h.get_color(), linewidth=4, linestyle=custom_linestyles.get(label, 'solid'))
    for h, label in zip(handles, labels)
]

fig.legend(legend_handles, labels, loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=18)

# Save the figure
plt.savefig(f'{FIGURE_DIR}/new_rnn_baseline_{lr}.pdf', bbox_inches='tight', dpi=1000)
plt.savefig(f'{FIGURE_DIR}/new_rnn_baseline_{lr}.png', bbox_inches='tight', dpi=1000)