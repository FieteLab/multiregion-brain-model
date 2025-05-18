import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold as skmanifold
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import torch
from scipy.io import loadmat
import glob
from tqdm import tqdm
import argparse
import matplotlib.cm as cm

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from towertask.config import DATA_DIR, FIGURE_DIR, get_figure_path

def logistic_separability_test(X, labels, label_name, key, use_pca=True, n_components=3, test_size=0.2, random_state=42, C=1e6):
    """
    Test linear separability using Logistic Regression, with optional PCA.

    Parameters:
    - X (ndarray): Input features.
    - labels (ndarray): Class labels.
    - label_name (str): Name of the label.
    - key (str): Data type.
    - use_pca (bool): Whether to apply PCA for dimensionality reduction.
    - n_components (int): Number of PCA components for dimensionality reduction.
    - test_size (float): Proportion of the data to be used for testing.
    - random_state (int): Seed for reproducibility.
    - C (float): Inverse regularization strength for Logistic Regression.

    Returns:
    - train_accuracy (float): Accuracy of logistic regression on the training set.
    - test_accuracy (float): Accuracy of logistic regression on the test set.
    - balanced_test_accuracy (float): Balanced accuracy on the test set.
    - stratified_baseline (float): Stratified baseline accuracy.
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"Skipping {key} data with label '{label_name}' because it has fewer than 2 classes.")
        return None, None, None, None, None

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA if needed
    if use_pca:
        pca = PCA(n_components=n_components, random_state=random_state)
        X_transformed = pca.fit_transform(X_scaled)
    else:
        X_transformed = X_scaled

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, labels, test_size=test_size, random_state=random_state)

    # Logistic Regression
    logreg = LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=random_state)
    logreg.fit(X_train, y_train)

    # Predictions
    y_train_pred = logreg.predict(X_train)
    y_test_pred = logreg.predict(X_test)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    balanced_test_accuracy = balanced_accuracy_score(y_test, y_test_pred)

    # Compute baselines
    majority_class = np.argmax(np.bincount(y_train))
    majority_baseline = np.mean(y_test == majority_class)
    random_baseline = 1 / len(unique_classes)

    # Compute stratified baseline
    class_counts = np.bincount(y_train)
    class_proportions = class_counts / len(y_train)
    stratified_baseline = sum(class_proportions ** 2)

    print(f"{label_name}: Train Acc: {train_accuracy:.2f}, Test Acc: {test_accuracy:.2f}, "
          f"Balanced Test Acc: {balanced_test_accuracy:.2f}, "
          f"Random Chance: {random_baseline:.2f}, Majority Baseline: {majority_baseline:.2f}, "
          f"Stratified Baseline: {stratified_baseline:.2f}")

    return train_accuracy, test_accuracy, balanced_test_accuracy, random_baseline, stratified_baseline


def check_perceptron_separability(X, labels, label_name, key, test_size=0.2, random_state=42, max_iter=1000):
    """
    Test linear separability using the Perceptron algorithm and evaluate test accuracy.

    Parameters:
    - X (ndarray): Input features (e.g., 'hidden' or 'ps').
    - labels (ndarray): Class labels for the data.
    - label_name (str): Name of the label (e.g., 'action', 'curr_pos').
    - key (str): Data type ('hidden' or 'ps').
    - test_size (float): Proportion of the data to be used for testing.
    - random_state (int): Seed for reproducibility.
    - max_iter (int): Maximum number of iterations for the Perceptron.

    Returns:
    - converged (bool): Whether the Perceptron converged.
    - test_accuracy (float): Accuracy of the Perceptron on the test set.
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"Skipping {key} data with label '{label_name}' because it has fewer than 2 classes.")
        return None, None

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=test_size, random_state=random_state
    )

    # Initialize the Perceptron
    perceptron = Perceptron(max_iter=max_iter, tol=1e-4, random_state=random_state)

    # Fit the Perceptron
    perceptron.fit(X_train, y_train)

    # Check convergence
    converged = perceptron.n_iter_ < max_iter
    if converged:
        print(f"The Perceptron converged for {key} data with label '{label_name}'.")
    else:
        print(f"The Perceptron did NOT converge for {key} data with label '{label_name}'.")

    # Evaluate test accuracy
    y_test_pred = perceptron.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Determine random chance accuracy
    random_chance = 1 / len(unique_classes)
    print(f"Test Accuracy: {test_accuracy:.2f}, Random Chance: {random_chance:.2f}")

    # Check if test accuracy is significantly better than random chance
    if test_accuracy < random_chance + 0.1:
        print(f"Warning: Test accuracy is close to random chance. Data may not be well-separated.")

    return converged, test_accuracy
    
def check_data_separability(X, labels, label_name, key, plot_path, test_size=0.2, random_state=42):
    """
    Fit an SVM to check data separability for a given label using train-test split.

    Parameters:
    - X (ndarray): Input features (e.g., 'hidden' or 'ps').
    - labels (ndarray): Class labels for the data.
    - label_name (str): Name of the label (e.g., 'action', 'curr_pos').
    - key (str): Data type ('hidden' or 'ps').
    - plot_path (str): Path to save the plots.
    - test_size (float): Proportion of the data to be used for testing.
    - random_state (int): Seed for reproducibility.

    Returns:
    - train_accuracy (float): Accuracy of the SVM on the training set.
    - test_accuracy (float): Accuracy of the SVM on the test set.
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"Skipping {key} data with label '{label_name}' because it has fewer than 2 classes.")
        return None, None
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=test_size, random_state=random_state
    )

    # Initialize the SVM with a linear kernel
    svm = SVC(kernel='linear', max_iter=1000, random_state=random_state)

    # Fit the SVM on the training data
    svm.fit(X_train, y_train)

    # Compute accuracy on training and test data
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    support_vector_ratio = len(svm.support_) / len(X_train)


    # print(f"SVM Train Accuracy for {key} data with label '{label_name}': {train_accuracy:.2f}")
    print(f"SVM Test Accuracy for {key} data with label '{label_name}': {test_accuracy:.2f}, random chance: {1/unique_classes}")
    
    # Plot decision boundary if the data is 2D
    if X.shape[1] == 2:
        plt.figure()
        h = .02  # Step size in the mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')
        plt.title(f"SVM Decision Boundary for {key} data ({label_name})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig(f"{plot_path}/{key}_{label_name}_svm_decision_boundary.png", dpi=300)

    return train_accuracy, test_accuracy


def plot_cumulative_explained_variance(pca_dict, plot_path, prefix, model_type):
    """
    Plots the cumulative explained variance by principal components for multiple PCA analyses.

    Parameters:
    - pca_dict (dict): A dictionary where keys are labels and values are PCA objects.
    - plot_path (str): Path where the plot will be saved.
    - prefix (str): Prefix for the saved plot file name.
    """
    plt.figure()
    label_dict = {'hidden': 'RNN states', 'ps': 'Place cells'}
    for i, (label, pca) in enumerate(pca_dict.items()):
        # Calculate cumulative explained variance and convert to percentage
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
        plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2,
                 label=label_dict.get(label, label))

    # Add a dashed line at 80%
    plt.axhline(y=80, color='gray', linestyle='--', linewidth=1.5)

    # Set y-axis limits from 0 to 100
    plt.ylim(0, 100)

    plt.title(f'Cumulative Variance Explained by #PCs in {model_type}', fontsize=16)
    plt.xlabel('Number of Principal Components', fontsize=16)
    plt.ylabel('Cumulative Variance Explained (%)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    # Save the figure
    plt.savefig(os.path.join(plot_path, f'{prefix}_cumulative_explained_variance.pdf'), dpi=1000)
    
def generate_filter_path(filters):
    # Create a string representation of the filters
    filter_str = "_".join([f"{var}={val}" for var, val in filters])
    return filter_str

def print_filter(filters):
    # Print the filtering criteria
    if filters:
        filter_descriptions = [f"{var}={val}" for var, val in filters]
        print(f"Applying filters: {', '.join(filter_descriptions)}")
    else:
        print("No filters applied.")

def filter_and_reduce_matrices(mat, traj_limit=100, filters=None):
    # Extract the info matrix
    info = mat['info'].T
    try:
        traj_ids, time, curr_pos, curr_diff, accum_diff, total_evi, action, gt_action, task, curr_spatial_axis, suc = info
    except:
        traj_ids, time, curr_pos, curr_diff, accum_diff, total_evi, action, gt_action, task, curr_spatial_axis, p0, p1, p2, suc = info
    # Create a dictionary to map variable names to their corresponding arrays
    variables = {
        'traj_ids': traj_ids,
        'time': time,
        'curr_pos': curr_pos,
        'curr_diff': curr_diff,
        'accum_diff': accum_diff,
        'total_evi': total_evi,
        'action': action,
        'gt_action': gt_action,
        'task': task,
        'curr_spatial_axis': curr_spatial_axis,
        'suc': suc
    }

    # Start with rows where suc == 1
    valid_rows = np.where(suc == 1)[0]
    # valid_rows = np.arange(len(suc))
    
     # Apply additional filtering if filters are specified
    if filters:
        for variable_name, filter_value in filters:
            if variable_name in variables:
                valid_rows = valid_rows[np.isin(valid_rows, np.where(variables[variable_name] != filter_value)[0])]
    
    # Further filter rows where action == gt_action (i.e., correct actions)
    # valid_rows = valid_rows[np.where(action[valid_rows] == gt_action[valid_rows])[0]]

    # Extract all successful traj_ids 
    valid_traj_ids = traj_ids[valid_rows]

    # Limit the traj_ids to the first 100 unique IDs
    # random_traj_ids = np.unique(valid_traj_ids)[:traj_limit]
    # Get the unique traj_ids
    unique_traj_ids = np.unique(valid_traj_ids)

    # Randomly select traj_limit number of unique traj_ids
    print('number of valid traj is', len(unique_traj_ids))
    # breakpoint()
    if len(unique_traj_ids) > traj_limit:
        random_traj_ids = np.random.choice(unique_traj_ids, traj_limit, replace=False)
    else:
        random_traj_ids = unique_traj_ids  # If there are less than traj_limit unique IDs, just use them all

    # Create a mask for rows with these traj_ids
    traj_mask = np.isin(valid_traj_ids, random_traj_ids)

    # Apply the mask to valid_rows to get final valid rows
    final_valid_rows = valid_rows[traj_mask]

    # Apply the filter to all relevant matrices in the mat dictionary
    filtered_mat = {key: mat[key][final_valid_rows, :] for key in mat.keys() if key in ['info', 'hidden', 'gs', 'ps']}
    
    return filtered_mat

def make_plot(Y, C, traj_ids, name, is_pca=False):
    """
    Create and save a scatter plot or a 3D plot based on the given data.

    Parameters:
    - Y (ndarray): Transformed data points.
    - C (ndarray): Colors for the data points.
    - traj_ids (ndarray): Trajectory identifiers.
    - name (str): Name for the saved plot file.
    """    
    axis_labels = ['PC1', 'PC2', 'PC3'] if is_pca else ['Dim1', 'Dim2', 'Dim3']
    
    # Create the 2D and 3D plots
    if Y.shape[1] > 2:
        # Combined 2D and 3D plot
        fig = plt.figure(figsize=(12, 6))
        ax_3d = fig.add_subplot(121, projection='3d')
        if traj_ids is None:
            ax_3d.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, alpha=0.5, cmap='viridis')
        else:
            for i in np.unique(traj_ids):
                mask = traj_ids == i
                ax_3d.plot3D(Y[mask, 0], Y[mask, 1], Y[mask, 2], alpha=0.05, color='grey', linewidth=0.5)
            ax_3d.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, alpha=0.5, cmap='viridis')
        
        ax_3d.set_xlabel(axis_labels[0])
        ax_3d.set_ylabel(axis_labels[1])
        ax_3d.set_zlabel(axis_labels[2])
        # ax_3d.set_title('3D Projection')
        
        ax_2d = fig.add_subplot(122)
        if traj_ids is None:
            result = ax_2d.scatter(Y[:, 0], Y[:, 1], c=C, alpha=0.5)
        else:
            for i in np.unique(traj_ids):
                mask = traj_ids == i
                ax_2d.plot(Y[mask, 0], Y[mask, 1], alpha=0.05, c='grey', linewidth=0.5)
            result = ax_2d.scatter(Y[:, 0], Y[:, 1], c=C, alpha=0.75)
        
        ax_2d.set_xlabel(axis_labels[0])
        ax_2d.set_ylabel(axis_labels[1])
        # ax_2d.set_title('2D Projection')
        
        fig.colorbar(result, ax=ax_2d)
        plt.savefig(f'{name}_combined.pdf', dpi=1000)
        
        # Separate 3D plot
        fig_3d = plt.figure(figsize=(8, 6))
        ax_3d_sep = fig_3d.add_subplot(111, projection='3d')
        if traj_ids is None:
            ax_3d_sep.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, alpha=0.5, cmap='viridis')
        else:
            for i in np.unique(traj_ids):
                mask = traj_ids == i
                ax_3d_sep.plot3D(Y[mask, 0], Y[mask, 1], Y[mask, 2], alpha=0.05, color='grey', linewidth=0.5)
            ax_3d_sep.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=C, alpha=0.5, cmap='viridis')
        
        ax_3d_sep.set_xlabel(axis_labels[0])
        ax_3d_sep.set_ylabel(axis_labels[1])
        ax_3d_sep.set_zlabel(axis_labels[2])
        # ax_3d_sep.set_title('3D Projection')
        # plt.savefig(f'{name}_3d.pdf')
    
    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if traj_ids is None:
            result = ax.scatter(Y[:, 0], Y[:, 1], c=C, alpha=0.5)
        else:
            for i in np.unique(traj_ids):
                mask = traj_ids == i
                ax.plot(Y[mask, 0], Y[mask, 1], alpha=0.05, c='grey', linewidth=0.5)
            result = ax.scatter(Y[:, 0], Y[:, 1], c=C, alpha=0.75)
        
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title('2D Projection')
        
        fig.colorbar(result, ax=ax)
        plt.savefig(f'{name}_2d.png', dpi=1000)

def run_non_linear_reduction(X, labels, hidden_name, dirname='jaedong', traj_ids=None, prefix='', do_PCA=False):
    """
    Perform non-linear dimensionality reduction on the given data and create plots.

    Parameters:
    - X (ndarray or Tensor): Input data.
    - labels (list or ndarray): Labels for the data.
    - hidden_name (str or list): Name(s) for the hidden states.
    - dirname (str): Directory name for saving plots.
    - traj_ids (ndarray): Trajectory identifiers.
    - prefix (str): Prefix for the saved plot files.
    """
    if len(X) == 0:
        print("X is empty")
        return

    os.makedirs(dirname, exist_ok=True)
    hidden_names = [hidden_name] if isinstance(hidden_name, str) else hidden_name
    label_list = [labels] if isinstance(labels, np.ndarray) else labels

    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    n_neighbors = min(100, X.shape[0] - 1)
    n_components = 10
    pca_dict = {}

    try:
        if not do_PCA:
            Ylle, squared_error = skmanifold.locally_linear_embedding(X, n_neighbors=n_neighbors, n_components=n_components)
        else:
            pca = PCA(n_components=n_components)
            Ylle = pca.fit_transform(X)
            # Plot the explained variance by each PC
            # plot_cumulative_explained_variance(pca, dirname, prefix)
            pca_dict[prefix.strip('_')] = pca
    except Exception as e:
        print(f"Error in LLE: {e}")
        return

    for hidden_name, labels in zip(hidden_names, label_list):
        print(hidden_name)
        if do_PCA:
            tail_str = '_pca'
        else:
            tail_str = 'locally_linear'
        make_plot(Ylle, labels, traj_ids, f'{dirname}/{prefix}{hidden_name}_{tail_str}', is_pca=do_PCA)
        
    return pca_dict 

def reverse_time(time, end_points):
    """
    Convert time to reverse time based on the end points.

    Parameters:
    - time (ndarray): Original time points.
    - end_points (ndarray): End points for reversing time.

    Returns:
    - reverse_time (ndarray): Reversed time points.
    """
    reverse_time = np.zeros_like(time)
    reverse_time[:end_points[0] + 1] = time[:end_points[0] + 1] - time[end_points[0]]
    for i in range(len(end_points) - 1):
        reverse_time[end_points[i] + 1:end_points[i + 1] + 1] = time[end_points[i] + 1:end_points[i + 1] + 1] - time[end_points[i + 1]]

    return reverse_time

def run_with_mask(mask, X, start_img, targ_img, dist, time, curr_state_img, curr_state_dist, traj_ids, _dirname):
    """
    Apply a mask to the data and run non-linear reduction on the masked data.

    Parameters:
    - mask (ndarray): Mask to apply to the data.
    - X (ndarray): Input data.
    - start_img (ndarray): Start images.
    - targ_img (ndarray): Target images.
    - dist (ndarray): Distances.
    - time (ndarray): Time points.
    - curr_state_img (ndarray): Current state images.
    - curr_state_dist (ndarray): Current state distances.
    - traj_ids (ndarray): Trajectory identifiers.
    - _dirname (str): Directory name for saving plots.
    """
    _time = time[mask]
    _start_img = start_img[mask]
    _targ_img = targ_img[mask]
    _dist = dist[mask]
    _curr_state_img = curr_state_img[mask]
    _curr_state_dist = curr_state_dist[mask]
    _traj_ids = traj_ids[mask]
    _X = X[mask]

    print(_X.shape)
    print(np.unique(_dist), np.unique(dist))

    hidden_names = ['dist', 'curr_state_img', 'curr_state_dist', 'time', 'start_img', 'targ_img']
    labels = [_dist, _curr_state_img, _curr_state_dist, _time, _start_img, _targ_img]
    run_non_linear_reduction(_X, labels, hidden_names, _dirname, _traj_ids)

def gen_confusion_matrix(X, Y, name, labels=None, xlabel=None):
    """
    Generate and save a confusion matrix plot based on the given data.

    Parameters:
    - X (ndarray): Predicted labels.
    - Y (ndarray): True labels.
    - name (str): Name for the saved plot file.
    - labels (list): List of labels to include in the matrix.
    - xlabel (str): Label for the x-axis.
    """
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    confusion = np.zeros((len(x_unique), len(y_unique)))

    for i, x in enumerate(x_unique):
        for j, y in enumerate(y_unique):
            confusion[i, j] = np.sum((X == x) & (Y == y))

    confusion = confusion / (1e-12 + confusion.sum(0))

    fig = plt.figure()
    plt.matshow(confusion)
    plt.xticks(np.arange(len(y_unique)), y_unique)
    plt.ylabel('K-Means Class')
    if xlabel is not None:
        plt.xlabel(xlabel)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, f'{confusion[i, j]:.2f}', ha='center', va='center', color='white')

    plt.colorbar()
    plt.savefig(name, dpi=1000)
    print(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tower Task Testing with GridWrapper')
    parser.add_argument('--with_sensory', action='store_true', help='if true, concat grid vector with sensory info')
    parser.add_argument('--model_type', type=str, required=True, help='choose from M1, ...M5')
    
    args = parser.parse_args()
    np.random.seed(42)
    
    num_episodes = 1000

    """Load saved hippocampal and RNN vectors from `format_path`"""
    # m1
    if args.model_type == 'M1':
        # for exaample:
        format_path = '...' # TODO for user: either enter directory manually OR utilize the parser above wisely to reconstruct the directory of .mat file
    elif args.model_type == 'M2':
        format_path = '...' # TODO for user
    elif args.model_type == 'M3':
        format_path = '...' # TODO for user
    elif args.model_type == 'M4':
        format_path = '...' # TODO for user
    elif args.model_type == 'M5':
        # for example...
        format_path = 'M5/mlp32/p/no_sensory/HaSH_star/seq20/maxTower5/RNN32/7_8_11/0.0005/trial_debug/800/'
    
    vector_save_path = os.path.join(DATA_DIR, format_path)
    load_file = os.path.join(vector_save_path, f'{num_episodes}trials.mat')
    plot_path = os.path.join(FIGURE_DIR, format_path, "nonlinear_reduction") # TODO: user can change the directory for saving plots
    
    os.makedirs(plot_path, exist_ok=True)
    print('Loading vectors from path:', load_file)
    print('Saving plots to:', plot_path)
    
    # load the .mat data file
    mat = loadmat(load_file)
    # `traj_limit=N` will randomly select N trials (if not enough trials are present, it uses all trials.)
    mat = filter_and_reduce_matrices(mat, traj_limit=100)

    traj_ids, time, curr_pos, curr_diff, accum_diff, total_evi, action, gt_action, task, curr_spatial_axis, _, _, _, suc = mat['info'].T

    hidden_names = ['time', 'curr_pos', 'curr_diff', 'accum_diff', 'total_evi', 'action', 'gt_action', 'task', 'curr_spatial_axis', 'suc']
    labels = [time, curr_pos, curr_diff, accum_diff, total_evi, action, gt_action, task, curr_spatial_axis, suc]

    pca_dict = {}
    for key in tqdm(['hidden', 'ps']):
        ## This block of code runs PCA, as shown in the paper:
        pca_dict.update(run_non_linear_reduction(mat[key], labels, hidden_names, plot_path, traj_ids, prefix=key+'_', do_PCA=True))

        ## This block of code runs nonlinear reduction using `skmanifold.locally_linear_embedding` (not in the paper):
        # run_non_linear_reduction(mat[key], labels, hidden_names, plot_path, traj_ids, prefix=key+'_', do_PCA=False)

    # Appendix FIG10 (TOP): Plot cumulative explained variance for both 'hidden' and 'ps'
    plot_cumulative_explained_variance(pca_dict, plot_path=plot_path, prefix='hidden_and_ps', model_type=args.model_type)


