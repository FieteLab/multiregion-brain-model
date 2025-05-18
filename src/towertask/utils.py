import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.fftpack
import torch
import json
import argparse
import os
import random
import numpy as np
import torch

def set_seed(seed=1):
    """Set the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def identify_model_variant(args):
    if args.rnn_only:
        if args.with_mlp:
            if hasattr(args, 'hidden_size') and args.hidden_size == 157:
                return "M0plus_matchNumParams"
            else:
                return "M0plus"
        else:
            if hasattr(args, 'hidden_size') and args.hidden_size == 158:
                return "M0_matchNumParams"
            else:
                return "M0"
    elif args.grid_assignment == ["position", "position", "position"]:
        if args.with_mlp and args.mlp_input_type == "sensory" and args.new_model:
            return "M5"
        elif args.with_mlp and args.mlp_input_type == "sensory":
            return "M3"
        elif args.new_model:
            return "M2"
        else:
            return "M1"
    elif args.grid_assignment == ["position", "position", "evidence"] and args.with_mlp and args.mlp_input_type == "sensory" and args.new_model:
        return "M4"
    else:
        return "custom"

def build_model_path(
    args,
    root_dir: str = "",
    *,
    compute_np: bool = True
) -> tuple[str, list[str], str]:
    """
    Build the relative path (and the parts list) that uniquely identifies a
    model run, matching the scheme used in 'train.py'.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (must include the same fields used in train.py).
    root_dir : str, optional
        If given, this directory is prepended to the joined path.
        Pass FIGURE_DIR, DATA_DIR, or model_path_header as needed.
    compute_np : bool, optional
        If True and args.Np is None, compute Np = min(400, prod(lambdas)).

    Returns
    -------
    full_path : str
        os.path.join(root_dir, *base_parts)
    base_parts : List[str]
        The individual path components (useful for other logic).
    model_variant : str
        The detected model variant (M0–M5, etc.).
    """
    # --- replicate Np logic from train.py ---------------------------------
    if compute_np:
        Np = args.Np if getattr(args, "Np", None) else min(400, np.prod(args.lambdas))
    else:
        Np = args.Np

    model_variant = identify_model_variant(args)

    base_parts = [
        model_variant,
        f"mlp{args.mlp_hidden_size}" if args.with_mlp else "no_mlp",
        "rnn_only" if args.rnn_only else args.gcpc,
        "with_sensory" if args.with_sensory else "no_sensory",
        "HaSH_star" if args.new_model else "HaSH_original",
        f"seq{args.sequence_length}",
        f"maxTower{args.max_towers}",
        f"RNN{args.hidden_size}",
    ]

    if args.rnn_only:
        base_parts += [str(args.learning_rate), args.trial_name,           ]
    else:
        lambda_str = "_".join(map(str, args.lambdas))
        base_parts += [lambda_str, str(args.learning_rate), args.trial_name, str(Np)]

    full_path = os.path.join(root_dir, *base_parts) if root_dir else os.path.join(*base_parts)
    return full_path, base_parts, model_variant

def calculate_cumulative_success(episode_successes, window_size=5000):
    cumulative_success = []
    cumulative_sum = 0
    for i, success in enumerate(episode_successes):
        cumulative_sum += success
        if i >= window_size:
            cumulative_sum -= episode_successes[i - window_size]
        cumulative_success.append(cumulative_sum / min(i + 1, window_size))
    return cumulative_success

def plot_success_rates_over_time(episode_successes, args, path, dump=False):
    cumulative_success = calculate_cumulative_success(episode_successes)
    ema_rates = calculate_ema(episode_successes[:], alpha=0.01)
    plt.figure(figsize=(10, 5))
    plt.plot(ema_rates, label='EMA Success Rate (%)')
    plt.plot(cumulative_success, label='Cumulative Success Rate (%)', linestyle='--')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rates Over Time (seq={args.sequence_length}, fov={args.fov})')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{path}/train_trajectory.pdf')

def get_class_weights(counter):
    maxw = max(counter.values())
    minw = min(counter.values())
    class_weights = {cls: minw / count for cls, count in counter.items()}
    return torch.tensor([class_weights.get(i, 1.0) for i in range(3)], dtype=torch.float32).to('cuda')
      
def save_args(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return None

    try:
        with open(filename, 'r') as f:
            args_dict = json.load(f)
        return argparse.Namespace(**args_dict)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def split_tensor_by_lambdas(tensor, lambdas, n):
    """
    Splits a given tensor into three parts based on the powers of provided lambdas.

    Args:
    tensor (Tensor): The tensor to split. Expected size [1, lambda_1^n + lambda_2^n + lambda_3^n].
    lambdas (list): A list of three lambda values [lambda1, lambda2, lambda3].
    n (int): The power to raise each lambda value to.

    Returns:
    tuple: A tuple of three tensors, split according to lambda_i^n.
    """
    # Calculate the sizes for each split
    sizes = [lambdas[0] ** n, lambdas[1] ** n, lambdas[2] ** n]

    # Ensure the input tensor is of the correct size
    if tensor.size(1) != sum(sizes):
        raise ValueError("Input tensor size does not match the expected size based on lambdas and n.")

    # Split the tensor
    split_tensors = torch.split(tensor, sizes, dim=1)

    return split_tensors
 
def reshape_to_square_tensors(split_tensors, lambdas):
    """
    Reshapes each tensor in split_tensors into a square tensor.

    Args:
    split_tensors (tuple): A tuple of three tensors, split according to lambda_i^n.
    lambdas (list): A list of three lambda values [lambda1, lambda2, lambda3].

    Returns:
    tuple: A tuple of three square tensors, each reshaped to lambda_i x lambda_i.
    """
    square_tensors = []

    for tensor, lambda_val in zip(split_tensors, lambdas):
        # Reshape each tensor to a square shape of lambda_i x lambda_i
        square_tensor = tensor.view(1, lambda_val, lambda_val)
        square_tensors.append(square_tensor)

    return tuple(square_tensors)

def elementwise_subtraction(tuple1, tuple2):
    """
    Performs element-wise subtraction between two tuples of tensors.

    Args:
    tuple1 (tuple): A tuple of tensors.
    tuple2 (tuple): Another tuple of tensors, must be the same length as tuple1.

    Returns:
    tuple: A tuple containing the element-wise subtraction of tuple1 and tuple2.
    """
    if len(tuple1) != len(tuple2):
        raise ValueError("Tuples must be of the same length for element-wise subtraction.")

    # Element-wise subtraction
    result = tuple(tensor1 - tensor2 for tensor1, tensor2 in zip(tuple1, tuple2))

    return result

def reshape_to_square_tensors(split_tensors, lambdas, dimension=2):
    """
    Reshapes each tensor in split_tensors into a square tensor.

    Args:
    split_tensors (tuple): A tuple of three tensors, split according to lambda_i^n.
    lambdas (list): A list of three lambda values [lambda1, lambda2, lambda3].

    Returns:
    tuple: A tuple of three square tensors, each reshaped to lambda_i x lambda_i.
    """
    square_tensors = []

    for tensor, lambda_val in zip(split_tensors, lambdas):
        # Reshape each tensor to a square shape of lambda_i x lambda_i
        if dimension == 2:
            square_tensor = tensor.view(1, lambda_val, lambda_val)
        elif dimension == 1:
            square_tensor = tensor.view(1, lambda_val)
        else:
            assert 'dimension not implemented'
        square_tensors.append(square_tensor)

    return tuple(square_tensors)

def flatten_square_tensors(square_tensors, lambdas):
    """
    Flattens each square tensor into a one-dimensional tensor with size lambda_val*lambda_val.

    Args:
    square_tensors (tuple): A tuple of square tensors.
    lambdas (list): A list of lambda values corresponding to the square tensors.

    Returns:
    tuple: A tuple of flattened tensors, each with size lambda_val*lambda_val.
    """
    flat_tensors = []

    for square_tensor, lambda_val in zip(square_tensors, lambdas):
        # Flatten each square tensor to a one-dimensional tensor with size lambda_val*lambda_val
        flat_tensor = square_tensor.view(-1)
        flat_tensors.append(flat_tensor)

    return tuple(flat_tensors)

def compute_autocorrelation(hidden_states, max_lag=None):
    # Number of time steps and number of hidden units
    T, H = hidden_states.shape
    
    if max_lag is None:
        max_lag = T - 1
    
    # Calculate mean for each hidden unit
    mean_h = np.mean(hidden_states, axis=0)
    # print(mean_h.shape)
    
    # Initialize autocorrelation values
    autocorrelation_values = np.zeros((max_lag, H))
    
    # Compute autocorrelation for each lag and each hidden unit
    for l in range(max_lag):
        num = np.sum((hidden_states[l+1:] - mean_h) * (hidden_states[:T-l-1] - mean_h), axis=0)
        den = np.sum((hidden_states - mean_h) ** 2, axis=0)
        autocorrelation_values[l] = num / den
    
    return autocorrelation_values

def transform_1d_coordinate(current_tensor, first_tensor, lambda_val):
    current_coords = torch.nonzero(current_tensor.squeeze()).squeeze()
    first_coords = torch.nonzero(first_tensor.squeeze()).squeeze()
    # do state = curernt_state - first_state
    diff = (current_coords - first_coords + lambda_val) % lambda_val
    
    result_tensor = torch.zeros_like(current_tensor)
    try:
        result_tensor[0, diff] = 1
    except:
        raise Exception('out of bound')
    # breakpoint()
    return result_tensor

def transform_2d_coordinate(current_tensor, first_tensor, lambda_val):
    current_coords = torch.nonzero(current_tensor.squeeze()).squeeze()
    first_coords = torch.nonzero(first_tensor.squeeze()).squeeze()
    # do state = curernt_state - first_state
    diff = (current_coords - first_coords + lambda_val) % lambda_val
    
    result_tensor = torch.zeros_like(current_tensor)
    # breakpoint()
    try:
        result_tensor[0, diff[0], diff[1]] = 1
    except:
        raise Exception('out of bound')
    # breakpoint()
    return result_tensor

def remove_consecutive_duplicates(arr):
    # Create an array of True/False indicating whether a row is different from the next one
    # np.roll shifts the array, so we compare each element with its next one
    # We also ensure that the last element is always kept by setting the last comparison to True
    mask = np.roll(arr, -1, axis=0) != arr
    mask[-1, :] = True
    # We only keep a row if any of its elements is different from the next one
    return arr[mask.any(1)]    

def plot_autocorrelation(autocorrelation_values):
    H = autocorrelation_values.shape[1]
    plt.figure(figsize=(15, 5))
    for h in range(H):
        plt.plot(autocorrelation_values[:, h], label=f'Hidden Unit {h+1}')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Hidden States')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/my_plt.png')
    plt.show()

def analyze_periodicity(peak_indices, name):
    intervals = []
    for peak_idx in peak_indices:
        interval = [peak_idx[i+1] - peak_idx[i] for i in range(len(peak_idx)-1)]

        interval = interval[:1]
        if len(interval) == 0:
            interval = [0]
        intervals.append(interval)
    
    # average intervals
    avg_intervals = np.asarray([np.mean(interval) for interval in intervals])
    print(np.mean(avg_intervals))
    print(np.unique(avg_intervals, return_counts=True))
    fig = plt.figure(figsize=(6,4),dpi=200)
    print(plt.hist(avg_intervals, bins=np.arange(avg_intervals.max()+1), density=True))
    fig.savefig(name)
    plt.xlabel('Perioid')
    plt.ylabel('Frequency')
    return avg_intervals

def run_autocorr(hiddens, save_dir, vis=True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hidden_size = hiddens[0].shape[0]
    
    if vis:
        XX = int(hidden_size ** 0.5)
        YY = int(np.ceil(hidden_size / XX))
        fig = plt.figure(figsize=(YY * 3, XX * 3))
        gs = gridspec.GridSpec(XX, YY, figure=fig)
    
    corrs = []
    for hid_id in tqdm(range(hidden_size)):
        corr = []
        if vis:
            i = hid_id // YY
            j = hid_id % YY
            ax = fig.add_subplot(gs[i, j])

        for traj_id in range(len(hiddens)):
            hid = hiddens[traj_id][hid_id]
            _corr = np.correlate(hid, hid, mode='full')
            corr.append(_corr)
            xs = np.arange(len(_corr)) - len(_corr) // 2
            if vis:
                ax.plot(xs, _corr, alpha=0.1, color='grey')
                print
        corrs.append(corr)

    auto_corr_dir = os.path.join(save_dir, 'auto_corr')
    if not os.path.exists(auto_corr_dir):
        os.makedirs(auto_corr_dir)
    fig.savefig(os.path.join(auto_corr_dir, 'auto_corr_all.pdf'))

def calculate_moving_window_success_rate(successes, window_size=10):
    moving_window_success_rates = []
    for i in range(len(successes) - window_size + 1):
        window = successes[i:i + window_size]
        # success_rate = sum(window) / window_size
        weights = list(range(1, window_size + 1))
        success_rate = sum(w * s for w, s in zip(weights, window)) / sum(weights)
        
        moving_window_success_rates.append(success_rate)
    return moving_window_success_rates

def calculate_ema(successes, alpha):
    ema = [successes[0]]  # Start with the first success rate
    for success in successes[1:]:
        ema.append(alpha * success + (1 - alpha) * ema[-1])
    return ema

def calculate_cumulative_success(episode_successes):
    num_episodes = len(episode_successes)
    success_rates = np.cumsum(episode_successes) / (np.arange(num_episodes) + 1)
    return success_rates

def calculate_moving_average(successes, window_size=10):
    """Calculate the moving average of success rates over a specified window size."""
    if len(successes) < window_size:
        return np.mean(successes)  # Not enough data, use simple average
    else:
        return np.mean(successes[-window_size:])  # Moving average over the last window_size episodes

def calculate_grouped_success_rate(episode_successes, group_size):
    num_groups = len(episode_successes) // group_size
    return [np.mean(episode_successes[i*group_size:(i+1)*group_size]) for i in range(num_groups)]

def variance_in_success_rate(episode_successes, group_size):
    num_groups = len(episode_successes) // group_size
    return [np.var(episode_successes[i*group_size:(i+1)*group_size]) for i in range(num_groups)]

def early_vs_late_success(episode_successes, split_ratio=0.5):
    split_index = int(len(episode_successes) * split_ratio)
    early_success_rate = np.mean(episode_successes[:split_index])
    late_success_rate = np.mean(episode_successes[split_index:])
    return early_success_rate, late_success_rate