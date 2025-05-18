import torch
import numpy as np
import torch.nn as nn
import random

def generate_data_with_max_towers(num_samples, sequence_length, fov, 
                                p_select_left=0.5, 
                                noise_level=0, extend_fov=True,
                                max_towers=None, seed=42):
    """
    The T-maze is being partitioned into various regions. 
    We model outside of maze as '-1' when `extend_fov = True`.
    For each of `num_samples` of samples we generate, we pick L/R uniformly at random to be the reward side 
    (the uniform chance can be changed using `p_select_left`). 

    We represent the presence of tower as `1` and `0` otherwise. 
    We further partition the data by `fov` (agent's field of view).
    """
    # np.random.seed(seed)
    data = []
    labels = []
    
    # Define the regions of the maze (assuming sequence_length is the total length)
    start_region_length = int(0.09 * sequence_length)  # 30cm out of 3.3m
    cue_region_length = int(0.61 * sequence_length)  # 2m out of 3.3m

    if extend_fov:
        extension_length = fov  # Number of additional steps needed # IMPORTANT modification (changed from fov-1)
        left_extension = np.full(extension_length, -1)
        right_extension = np.full(extension_length, -1)
    
    for _ in range(num_samples):
        rewarded_side = np.random.choice([0, 1], p=[p_select_left,1-p_select_left])  # 0 for left, 1 for right
        side_map = {1:'right', 0:'left'}
        # Determine the number of towers for each side based on the reward side and max_towers
        if max_towers is not None:
            reward_side_towers = np.random.randint(1, max_towers + 1)
            non_reward_side_towers = np.random.randint(0, reward_side_towers)  # Ensure the non-rewarded side has fewer towers

            if rewarded_side == 0:  # Left is the rewarded side
                left_towers_count = reward_side_towers
                right_towers_count = non_reward_side_towers
            else:  # Right is the rewarded side
                left_towers_count = non_reward_side_towers
                right_towers_count = reward_side_towers
        else:
            # Handle the case where max_towers is not defined
            # e.g., one can choose to simply not limit the towers
            assert f'No valid max towers {max_towers}'  # Placeholder 

        # Initialize the sequences
        left = np.zeros(sequence_length)
        right = np.zeros(sequence_length)
        
        # Ensure towers are distributed within the cue region
        if left_towers_count > 0:
            left_positions = np.random.permutation(range(start_region_length, start_region_length + cue_region_length))[:left_towers_count]
            left[left_positions] = 1
        if right_towers_count > 0:
            right_positions = np.random.permutation(range(start_region_length, start_region_length + cue_region_length))[:right_towers_count]
            right[right_positions] = 1
        
        ################################################
        # Determine the label based on the sum of towers
        sanity_check = 0 if np.sum(left) > np.sum(right) else 1
        
        # Add Gaussian noise to `left` and `right`
        if noise_level > 0:
            left += np.random.normal(0, noise_level, sequence_length)
            right += np.random.normal(0, noise_level, sequence_length)

        label = rewarded_side
        assert label == sanity_check
        
        # Prepare the data for the field of view (fov)
        sequence_data = []
        if extend_fov:
             # Append the 'outside maze' or 'wall' data to the sequences
            left = np.concatenate((left, left_extension))
            right = np.concatenate((right, right_extension))
            for i in range(sequence_length + 1): 
                left_fov = left[i:i+fov]
                right_fov = right[i:i+fov]
                sequence_data.append(np.hstack((left_fov, right_fov)))
        else:
            for i in range(sequence_length - fov + 1):
                left_fov = left[i:i+fov]
                right_fov = right[i:i+fov]
                sequence_data.append(np.hstack((left_fov, right_fov)))

        true_local_evidence = right - left
        data.append(sequence_data)
        labels.append(label)

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long), right_towers_count-left_towers_count, true_local_evidence