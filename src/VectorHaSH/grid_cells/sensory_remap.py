import numpy as np
import matplotlib.pyplot as plt
import timeit
import torch
import random

from VectorHaSH.grid_cells.assoc_utils_np import (gen_gbook, train_gcpc, pseudotrain_3d, pseudotrain_2d, module_wise_NN,
                             pseudotrain_3d_iterative, pseudotrain_2d_iterative_step, pseudotrain_3d_iterative_step)
from VectorHaSH.grid_cells.assoc_utils_np_2D import gen_gbook_2d, module_wise_NN_2d, path_integration_Wgg_2d
from VectorHaSH.grid_cells.surprisal import SurprisalLinear, SurprisalExponential, SurprisalExponentialBVC, SurprisalRegion, SurprisalRegionCheat

def random_binary(size):
    """
    Generate a random binary vector (1/-1) with given *size*
    """
    
    return np.random.choice([1, -1], size=size)

def flip_vector_random(v, nflips):
    """
    Randomly flip (change sign) *nflips* units in input vector
    Returns new vector
    """
    
    v = v.copy()
    indices = np.random.choice(len(v), size=nflips, replace=False)
    v[indices] = -v[indices]
    return v

def flip_vector_first(v, nflips):
    """
    Flip (change sign) first *nflips* units of input vector
    Returns new vector
    """
    
    v = v.copy()
    v[:nflips] = -v[:nflips]
    return v

def flip_vector(v, flip_indices):
    """
    Flip (change sign) input vector at given indices
    Mutates vector in-place
    """
    
     # in-place
    v[flip_indices] = -v[flip_indices]

def diff_bits(v1, v2):
    """
    Return indices where the two input 1D vectors differ
    """
    
    return np.argwhere(v1 != v2).flatten()

def sbook_1D_corridor(s1, s2, Ns, k_room, k_door, n_room, n_door, random=True):
    """
    Return sensory codebook (sensory vector for each position) for
    1D corridor with two rooms connected by a doorway
    Gradually flips bits to transition from s1 into s2
    Rate of change differs between room and doorway

    Inputs:
        s1 - np.array, sensory vector at the start (in room 1)
        s2 - np.array, sensory vector at the end (in room 2)
        Ns - int, number of sensory cells
        k_room - int, number of bits/cells to change per step in a room
        k_door - int, number of bits/cells to change per step at doorway
        n_room - int, number of steps in a room
        n_door - int, number of steps at doorway
        random - bool, whether to randomize which bits are flipped each time

    Outputs:
        sbook - np.array, size (Ns, 2*(n_room)+n_door+1)
    """
    
    diff = diff_bits(s1, s2) # indices of bits to flip
    if random:
        np.random.shuffle(diff) # randomize order of bit flipping
    flip_index, sbook_index = 0, 0
    sbook = np.zeros((Ns, 2*(n_room)+n_door+1))

    # add s1
    s = s1.copy()
    sbook[:, sbook_index] = s
    sbook_index += 1

    # add room1, doorway, room2: flip bits sequentially
    for n, k in zip([n_room, n_door, n_room], 
                   [k_room, k_door, k_room]):
        for i in range(n):
            flip_vector(s, diff[flip_index:flip_index+k])
            sbook[:, sbook_index] = s
            flip_index += k
            sbook_index += 1

    # assert np.all(s == s2) # check we have arrived at s2
    return sbook

def sbook_1D_corridor_random(Ns, k_room, k_door, n_room, n_door):
    """
    Wrapper around sbook_1D_corridor()
    Generates random start and end sensory vectors
    """
    
    N = 2*k_room*n_room + k_door*n_door # total number of locations in corridor
    s1 = random_binary(Ns)
    s2 = flip_vector_random(s1, N)
    return sbook_1D_corridor(s1, s2, Ns, k_room, k_door, n_room, n_door)


def gbook_1D_corridor(lambdas, Ng, Npos, n_room, n_door, gap=None):
    """
    Return grid codebook (grid vector for each position) for
    1D corridor with two rooms connected by a doorway
    Room1 and room2 take two disconnected sections of overall grid codebook
    NOTE: only implemented for n_door=1 right now

    Inputs:
        lambdas - list[int], periods of grid modules
        Ng - int, number of grid cells
        Npos - int, number of positions that can be represented
        n_room - int, number of steps in a room
        n_door - int, number of steps at doorway
        gap - int/None, number of steps between grid patterns for the end of room1
            and the doorway (start of room2)
            if None, uses gap that maximally separates the two sections of overall grid codebook used

    Outputs:
        sbook - np.array, size (Ns, 2*(n_room)+n_door+1)
    """
    
    if n_door != 1:
        raise NotImplementedError()
    gbook_all = gen_gbook(lambdas, Ng, Npos) # get overall grid codebook
    gbook = np.zeros((Ng, 2*n_room+1+1)) # initialize corridor grid codebook
    # room 1 (including s1): start at beginning of overall gridbook 
    gbook[:, :n_room+1] = gbook_all[:, :n_room+1]
    # room 2 (including doorway, s2): jump to different section of overall grid codebook
    if gap is None:
        gap = (Npos-(n_room+1)*2)//2
    start = n_room+1 + gap
    gbook[:, n_room+1:] = gbook_all[:, start:start+n_room+1]
    return gbook

def grid_to_pos(g, gbook):
    """
    Return the encoded location of grid pattern g
    using the given grid codebook
    """
    
    indices = np.where(np.all(gbook == g[:, None], axis=0))[0]
    if len(indices) == 0: # if not found, return None
        return None
    return indices[0]

def path_integration_Wgg(lambdas, Ng):
    """
    Return Wgg that maps any grid pattern g
    to the grid pattern for the next position
    """
    
    Wgg = np.zeros((Ng, Ng))
    lambda_cum = np.cumsum([0] + lambdas)
    # for each module, make permutation matrix
    for i, lam in enumerate(lambdas):
        mat = np.eye(lam)
        mat = np.concatenate([mat[-1:], mat[:lam-1]], axis=0)
        Wgg[lambda_cum[i]:lambda_cum[i+1],
           lambda_cum[i]:lambda_cum[i+1]] = mat
    return Wgg

def dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                gbooks, module_sizes,
                                ginit, sinit, surprisal, Niter, 
                                continuous=False, _2d=False,
                                binary=True, alpha=0, beta=0, use_torch=False,
                                with_recurrence=False, Wpp=None, theta_pp=None, prev_p=None):
    """
    Runs network of grid/place/sensory (g, p, s) layers
    Dynamics for remapping model:
        1. path integration advances g
        2. surprisal inhibits Wpg
        3. compute p based on s and g
        4. compute reconstructed g, s from p
        (repeat 3-4 for *Niter* times)
    Handles multiple runs simultaneously

    Inputs:
        Wgp, Wpg, Wgg, Wsp, Wps - np.array, weight matrices (Wyx is weights from x to y)
            contains matrices for multiples runs (except for Wgg, which is shared acrosss runs)
        gbooks, module_sizes - grid codebook & grid periods for computing g from raw activitions
            if 2D, gbooks is a list of flattened gbooks for each module
            if 1D, gbooks is a single gbook containing all modules
            (see module_wise_NN(), module_wise_NN_2d())
        ginit - np.array, grid vector before current input, size (Ng, 1)
        sinit - np.array, current sensory input, size (nruns, Ns, 1)
        surprisal - np.array, surprisal of current sensory input, size (nruns, 1, 1)
        Niter - int, number of times to run s/p -> g -> s/p loop
        continuous - bool, whether to output continuous sensory vectors 
            or binary (1/-1, from sign function)
        _2d - bool, whether env is 2D (or 1D)
        binary, alpha, beta - how to weigh inputs from s, g to p
            binary - bool, if True then p either gets only input from p (if surprisal is low)
                or only input from s (if surprisal is high)
            alpha, beta - float, weights on input from p, g to p
                not used if binary=True

    Outputs:
        p, g, s - activity vectors for p, g, s layers, 
            size (nruns, # cells in layer, 1)
    """
    
    s = sinit
    g = ginit
    # breakpoint()
    g = Wgg@g # 1. path integrate advances g
    if use_torch:
        sign_fn = torch.sign
    else:
        sign_fn = np.sign
    fn = torch.relu
    for i in range(Niter):
        # 2. surprisal inhibits Wpg? 3. compute p based on s and g
        if binary:
            if with_recurrence:
                p = fn(surprisal * Wps@s + (1-surprisal) * Wpg@g)
            else:
                p = fn(surprisal * Wps@s + (1-surprisal) * Wpg@g)
            # breakpoint()
        else:
            p = fn(alpha * Wps@s + beta*(1-surprisal) * Wpg@g)
        gin = Wgp@p
        # 4. compute reconstructed g, s from p
        if _2d:
            g = module_wise_NN_2d(gin, gbooks, module_sizes, use_torch=use_torch)
        else:
            g = module_wise_NN(gin, gbooks[:,:module_sizes[-1]], module_sizes, use_torch=use_torch)
        s = Wsp@p
        if not continuous:
            s = sign_fn(s)
    # breakpoint()
    return p, g, s

def run_surprisal_remap_1d(nruns=20, Npatts_train=25, Npatts_test=102,
    lambdas=(7,8,9), Np=200, 
    sbook_func="corridor", Ns=1000, k_room=5, k_door=250, n_room=50, n_door=1,
    surprisal_method='exponential', tau=0.1, threshold=-950, 
    Niter=1, alpha=0.05, beta=1, 
    incremental=True, incremental_s='real', Npatts_train_gp='all',
    return_info=False):
    """
    Simulates network as agent traverses an environment
    Returns p, g, s activities at each location (and optionally other info)

    Inputs:
        nruns - int, number of runs
            each run has its own sbook, random Wpg weights; shares gbook
        Npatts_train - int, the first *Npatts_train* locations are used
            to train Wsp, Wps, and optionally Wgp (see *Npatts_train_gp*)
        Npatts_test - int or None, the first *Npatts_test* locations in the environment 
            are traversed in order
            if None, then traverse all locations in the environment (sbook)
        lambdas - tuple(int) or list[int], grid periods
        Np - int, number of place cells
        sbook_func - str or function, used to generate sbook for each run
            "corridor": calls sbook_1D_corridor_random()
            function: will be called to generate sbook
        Ns, k_room, k_door, n_room, n_door: only used if sbook_func="corridor",
            see sbook_1D_corridor_random()
        surprisal_method - str
            "linear" or "exponential": which SurprisalEngine to use
            "cheat": will use surprisal=1 at doorway, =0 elsewhere
                only works for sbook_func="corridor"
        tau, threshold - params for SurprisalEngine
        Niter, alpha, beta - params for model dynamics, see dynamics_surprisal_remap()
        incremental - bool, whether to update weights (between s & p) 
            when model traverses new locations not seen in training 
        incremental_s - str, when updating s & p weights
            "real": use the real sensory input
            "model": use reconstructed activity from model dynamics
            only used if incremental=True
        Npatts_train_gp - str or int, which locations to use for training Wgp
            "all": all locations in codebooks
            "train": the first *Npatts_train* locations
            int: the first *Npatts_train_gp* locations
        return_info - bool, whether to return additional info (see Outputs)

    Outputs:
        if return_info=False: outputs
        if return_info=True: outputs, [pbook, gbook, sbook], surprisal_history
            outputs - list[np.array], activities in p/g/s layers at each location
                each has size (nruns, # cells, Npatts_test)
            pbook - np.array, size (nruns, Np, Npos)
                (Npos = product of lambdas, total number of positions that can be represented
                in grid place)
            gbook - np.array, size (Ng, Npos) (Ng = sum of lambdas)
            sbook - np.array, size (nruns, Ns, # positions in environment)
            surprisal_history - np.array, size (nruns, Npatts_test), raw surprisal values
        Note the first location is not actually traversed, and activities there should be ignored
    """
    
    # --- Set up
    # params
    lambdas = list(lambdas)
    Ng = np.sum(lambdas)
    Npos = np.prod(lambdas); 
    gbook = gen_gbook(lambdas, Ng, Npos)
    
    # sbook
    if sbook_func == "corridor":
        sbook_func = lambda: sbook_1D_corridor_random(Ns, k_room, k_door, n_room, n_door)
    sbook = np.array([sbook_func() for i in range(nruns)])
    Ns = sbook.shape[1] # number of sensory cells
    if Npatts_test is None:
        Npatts_test = sbook.shape[2]

    # weights
    Wpg = np.random.randn(nruns, Np, Ng) #/ (np.sqrt(M));       # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)
    if Npatts_train_gp == 'all':
        Wgp = train_gcpc(pbook, gbook, Npos)
    elif Npatts_train_gp == 'normal':
        Wgp = train_gcpc(pbook, gbook, Npatts_train)
    else:
        Wgp = train_gcpc(pbook, gbook, Npatts_train_gp)
    Wsp = pseudotrain_3d(sbook, pbook, Npatts_train)
    Wps = pseudotrain_3d(pbook, sbook, Npatts_train)
    Wgg = path_integration_Wgg(lambdas, Ng)

    # surprisal
    if surprisal_method == 'linear':
        surprisal_engine = SurprisalLinear(tau, threshold, nruns)
    elif surprisal_method == 'exponential':
        surprisal_engine = SurprisalExponential(tau, threshold, nruns)

    outputs = [np.zeros((nruns, dim, Npatts_test)) for dim in [Np, Ng, Ns]] # store outputs (p, g, s)
    surprisal_history = np.zeros((nruns, Npatts_test)) # store raw surprisal

    # --- Traverse environment and run model
    # the first location is not explicitly traversed bc it doesn't have a previous grid state
    # (for path integration)
    for x in range(1, Npatts_test):

        if x == 1:
            ginit = gbook[:,0,None]
        strue = sbook[:,:,x,None]       # true (noiseless) sensory pattern

        # compute surprisal
        if surprisal_method == 'cheat':
            if x == n_room + 1:
                surprisal = np.ones((nruns, 1, 1))
            else:
                surprisal = np.zeros((nruns, 1, 1))
            surprisal_history[:, x] = surprisal.squeeze()
        else:
            surprisal, raw_surprisal = surprisal_engine.get_surprisal(sbook[:, :, x], return_raw=True)
            surprisal_history[:, x] = raw_surprisal.squeeze()
        if np.any(surprisal > 0):
            print(x, np.mean(surprisal))

        # run model
        sinit = strue
        results = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps, alpha, beta,
                                    gbook, lambdas,
                                    ginit, sinit, surprisal, Niter) 
        p, g, s = results
        ginit = results[1]
        for i in range(3):
            outputs[i][:, :, x] = results[i].squeeze()

        # update Wsp, Wps
        if incremental and x >= Npatts_train:
            if x == Npatts_train:
                sbook_obs = sbook[:, :, :x].copy()
                pbook_obs = pbook[:, :, :x].copy()
            if incremental_s == 'real':
                sbook_obs = np.concatenate((sbook_obs, strue), axis=2)     
            else:
                sbook_obs = np.concatenate((sbook_obs, s), axis=2)     
            pbook_obs = np.concatenate((pbook_obs, p), axis=2)
            n = x + 1 # since x is 0-indexed
            Wsp = pseudotrain_3d(sbook_obs, pbook_obs, n)
            Wps = pseudotrain_3d(pbook_obs, sbook_obs, n)
    
    if return_info:
        return outputs, [pbook, gbook, sbook], surprisal_history
    return outputs

def plot_grid_results(g, gbook):
    
    grid_pos_new_runs = []
    for run in range(g.shape[0]):
        grid_pos_new = []
        for i in range(g.shape[2]):
            grid_pos_new.append(grid_to_pos(g[run, :, i], gbook))
        grid_pos_new_runs.append(grid_pos_new)

    for grid_pos_new in grid_pos_new_runs:
        plt.plot(grid_pos_new)
    return grid_pos_new_runs

# --- 2D
def sbook_2D_corridor(s1, s2, Ns, Npos, k_room, k_door, n_room, n_door, env_positions,
    random=True):
    """
    See sbook_1D_corridor(), but returns 2D sensory codebook 
    (sensory vector for each location (x, y)), 
    and user specifies the location of each step in the corridor
    
    Locations not in the corridor have 0 as sensory vector

    Inputs:
        s1, s2, Ns, k_room, k_door, n_room, n_door, random - see sbook_1D_corridor()
        Npos - int, total number of positions in each spatial axis
            (for setting size of sbook)
        env_positions - list[(int, int)], coords of each location in the corridor
        
    Outputs:
        sbook - np.array, size (Ns, Npos, Npos)
    """
    
    diff = diff_bits(s1, s2) # indices of bits to flip
    if random:
        np.random.shuffle(diff) # randomize order of bit flipping
    flip_index, position_index = 0, 0
    sbook = np.zeros((Ns, Npos, Npos)) # initialize sbook

    # store s1 for the first location in the corridor
    s = s1.copy()
    sbook[(slice(None), *env_positions[position_index])] = s
    position_index += 1

    # room1, doorway, room2: flip bits sequentially
    for n, k in zip([n_room, n_door, n_room], 
                   [k_room, k_door, k_room]):
        for i in range(n):
            flip_vector(s, diff[flip_index:flip_index+k])
            sbook[(slice(None), *env_positions[position_index])] = s
            flip_index += k
            position_index += 1

    # assert np.all(s == s2) # check we have arrived at s2
    return sbook

def grid_to_pos_2d(g, gbook):
    """
    Return the encoded location of grid pattern g
    using the given grid codebook
    """
    
    indices = np.where(np.all(gbook == g[:, None, None], axis=0))
    if len(indices[0]) == 0: # if not found, return None
        return None
    return tuple(arr[0] for arr in indices)

def decode_grid(g, gbook):
    
    grid_pos_new_runs_x = []
    grid_pos_new_runs_y = []
    for run in range(len(g)):
        grid_pos_new_x = [None] # 1st location is not visited, g is invalid
        grid_pos_new_y = [None]
        for i in range(1, g.shape[2]):
            x, y = grid_to_pos_2d(g[run, :, i], gbook)
            grid_pos_new_x.append(x)
            grid_pos_new_y.append(y)
        grid_pos_new_runs_x.append(grid_pos_new_x)
        grid_pos_new_runs_y.append(grid_pos_new_y)
    # encoded_pos shape (2, nruns, Npatts)
    encoded_pos = np.array([grid_pos_new_runs_x, grid_pos_new_runs_y])
    return encoded_pos

def run_surprisal_remap_2d(path_locations, sbook_func,
    nruns=20, Npatts_train=25, Npatts_test=None,
    lambdas=(5,6,7), Np=200, 
    surprisal_method='exponential', tau=0.1, threshold=-950, surprisal_indices=None,
    Niter=1, alpha=0.05, beta=1, continuous=False, 
    incremental=True, incremental_s='real', Npatts_train_gp='all',
    return_info=False):
    """
    Simulates network as agent traverses an 2D environment
    Returns p, g, s activities at each location (and optionally other info)
    2D version of run_surprisal_remap_1d()

    Inputs:
        path_locations - list[(int, int)], location of each step in the path
        sbook_func - function, called to generate sbook
            (sbook should be np.array, size (Ns, # positions in path),
            maps position on path to sensory input)
        nruns - int, number of runs
            each run has its own sbook, random Wpg weights; shares gbook
        Npatts_train - int, the first *Npatts_train* locations in *path_locations* are
            used to train Wsp, Wps, and optionally Wgp (see *Npatts_train_gp*)
        Npatts_test - int or None, the first *Npatts_test* locations in 
            *path_locations*  are traversed in order
            if None, then traverse all locations in *path_locations*
        lambdas - tuple(int) or list[int], grid periods
        Np - int, number of place cells
        surprisal_method - str
            "linear" or "exponential": which SurprisalEngine to use
            "cheat": will use surprisal=1 at *surprisal_indices*, =0 elsewhere
        tau, threshold - params for SurprisalEngine
        surprisal_indices - None or list[int], indices of locations on the path 
            that have surprisal=1
            only used if surprisal_method="cheat"
        Niter, alpha, beta, continuous - params for model dynamics, see dynamics_surprisal_remap()
            continuous - whether sensory inputs are continuous or binary (1/-1)
        incremental - bool, whether to update weights (between s & p) 
            when model traverses new locations not seen in training 
        incremental_s - str, when updating s & p weights
            "real": use the real sensory input
            "model": use reconstructed activity from model dynamics
            only used if incremental=True
        Npatts_train_gp - str or int, which locations to use for training Wgp
            "all": all locations in codebooks
            "train": the first *Npatts_train* locations in the path
            int: the first *Npatts_train_gp* locations in the path
        return_info - bool, whether to return additional info (see Outputs)

    Outputs:
        if return_info=False: outputs
        if return_info=True: outputs, [pbook, gbook, sbook], surprisal_history
            outputs - list[np.array], activities in p/g/s layers at each location
                each has size (nruns, # cells, Npatts_test)
            pbook - np.array, size (nruns, Np, Npos, Npos)
                (Npos = product of lambdas, total number of positions that can be represented 
                in grid space in one axis)
            gbook - np.array, size (Ng, Npos, Npos) (Ng = sum of lambda^2)
            sbook - np.array, size (nruns, Ns, # positions in path)
            surprisal_history - np.array, size (nruns, Npatts_test), raw surprisal values
        Note the first location in the path is not actually traversed, 
            and activities there should be ignored
    """
    
    # --- Set up
    # params
    lambdas = list(lambdas)
    Ng = np.sum(np.dot(lambdas, lambdas))
    Npos = np.prod(lambdas)
    module_sizes = [i**2 for i in lambdas]
    module_gbooks = [np.eye(i) for i in module_sizes]
    if Npatts_test is None:
        Npatts_test = len(path_locations)
    # for indexing flattened codebooks
    path_locations_flattened = [x*Npos+y for x, y in path_locations]

    # codebooks
    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    sbook = np.array([sbook_func() for i in range(nruns)])
    Ns = sbook.shape[1] # number of sensory cells

    Wpg = np.random.randn(nruns, Np, Ng) #/ (np.sqrt(M));   # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,klm->ijlm', Wpg, gbook))  # (nruns, Np, Npos, Npos)
    # flattened: compress the two axes (Npos, Npos) into single dimension
    gbook_flattened = gbook.reshape(Ng, Npos*Npos)
    pbook_flattened = pbook.reshape(nruns, Np, Npos*Npos)
    
    # weights
    if Npatts_train_gp == 'all':
        Wgp = train_gcpc(pbook_flattened, gbook_flattened)
    elif Npatts_train_gp == 'normal':
        Wgp = train_gcpc(pbook_flattened, gbook_flattened, patts_indices=path_locations_flattened[:Npatts_train])
    else:
        Wgp = train_gcpc(pbook, gbook_flattened, patts_indices=path_locations_flattened[:Npatts_train_gp])
    Wsp = pseudotrain_3d(sbook[:,:,:Npatts_train], 
                         pbook_flattened[:,:,path_locations_flattened[:Npatts_train]])
    Wps = pseudotrain_3d(pbook_flattened[:,:,path_locations_flattened[:Npatts_train]],
                         sbook[:,:,:Npatts_train])
    Wggs = {}
    for axis in [0, 1]:
        for direction in [1, -1]:
            shift = [0, 0]
            shift[axis] = direction
            Wggs[tuple(shift)] = path_integration_Wgg_2d(lambdas, Ng, axis, direction)

    # surprisal
    if surprisal_method == 'linear':
        surprisal_engine = SurprisalLinear(tau, threshold, nruns)
    elif surprisal_method == 'exponential':
        surprisal_engine = SurprisalExponential(tau, threshold, nruns)
    
    outputs = [np.zeros((nruns, dim, Npatts_test)) for dim in [Np, Ng, Ns]] # store outputs (p, g, s)
    surprisal_history = np.zeros((nruns, Npatts_test)) # store raw surprisal

    # --- Traverse environment and run model
    # the first location is not explicitly traversed bc it doesn't have a previous grid state
    # (for path integration)
    for x in range(1, Npatts_test): 

        if x == 1:
            ginit = gbook[(slice(None), *path_locations[0], None)]
        # true (noiseless) sensory pattern
        strue = sbook[(slice(None), slice(None), x, None)]  # true (noiseless) sensory pattern
        
        # compute surprisal
        if surprisal_method == 'cheat':
            if x in surprisal_indices:
                surprisal = np.ones((nruns, 1, 1))
            else:
                surprisal = np.zeros((nruns, 1, 1))
            surprisal_history[:, x] = surprisal.squeeze()
        else:
            if x == 1:
                strue_start = sbook[(slice(None), slice(None), 0, None)] 
                _ = surprisal_engine.get_surprisal(strue_start.squeeze(axis=-1), return_raw=True)
            surprisal, raw_surprisal = surprisal_engine.get_surprisal(strue.squeeze(axis=-1), return_raw=True)
            surprisal_history[:, x] = raw_surprisal.squeeze()
        if np.any(surprisal > 0):
            print(x, np.mean(surprisal))

        # run model
        sinit = strue
        Wgg = Wggs[tuple(np.array(path_locations[x]) - np.array(path_locations[x-1]))]
        results = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps, alpha, beta,
                                    module_gbooks, module_sizes,
                                    ginit, sinit, surprisal, Niter, continuous, _2d=True) 
        p, g, s = results
        ginit = results[1]
        for i in range(3):
            outputs[i][:, :, x] = results[i].squeeze()

        # update Wsp, Wps
        if incremental and x >= Npatts_train:
            if x == Npatts_train:
                sbook_obs = np.zeros((nruns, Ns, x+1))
                pbook_obs = np.zeros((nruns, Np, x+1))
                for i, path_location in enumerate(path_locations[:x+1]):
                    sbook_obs[:, :, i] = sbook[(slice(None), slice(None), i)]
                    pbook_obs[:, :, i] = pbook[(slice(None), slice(None), *path_location)]
            if incremental_s == 'real':
                sbook_obs = np.concatenate((sbook_obs, strue), axis=2)     
            else:
                sbook_obs = np.concatenate((sbook_obs, s), axis=2)     
            pbook_obs = np.concatenate((pbook_obs, p), axis=2)
            # n = x + 1 # since x is 0-indexed
            Wsp = pseudotrain_3d(sbook_obs, pbook_obs)
            Wps = pseudotrain_3d(pbook_obs, sbook_obs)
    
    if return_info:
        return outputs, [pbook, gbook, sbook], surprisal_history
    return outputs

def run_surprisal_remap_2d_surprisal_region(path_locations, sbook_func,
    nruns=20, Npatts_train=25, Npatts_test=None,
    lambdas=(5,6,7), Np=200, 
    surprisal_method='exponential', tau=0.1, threshold=-950, return_threshold=None, normalized=False, flipped=False,
    surprisal_enter_indices=None, surprisal_exit_indices=None, b=0, predict=False, scale_factor=1,
    Niter=1, binary=True, alpha=0, beta=0, continuous=False, 
    incremental=True, incremental_s='real', Npatts_train_gp='all',
    return_info=False, pseudoinverse_iterative=False, epsilon=0.01, lite=False, verbose=True):
    
    # --- Set up
    
    # params
    lambdas = list(lambdas)
    Ng = np.sum(np.dot(lambdas, lambdas))
    Npos = np.prod(lambdas)
    module_sizes = [i**2 for i in lambdas]
    module_gbooks = [np.eye(i) for i in module_sizes]
    if Npatts_test is None:
        Npatts_test = len(path_locations)
    # for indexing flattened codebooks
    path_locations_flattened = [x*Npos+y for x, y in path_locations]

    # codebooks
    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    sbook = np.array([sbook_func() for i in range(nruns)])
    Ns = sbook.shape[1] # number of sensory cells
    if normalized == 'sensory_only':
        sbook = (sbook / np.linalg.norm(sbook, axis=1)[:, None, :])

    Wpg = np.random.randn(nruns, Np, Ng) #/ (np.sqrt(M));   # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,klm->ijlm', Wpg, gbook))  # (nruns, Np, Npos, Npos)
    # flattened: compress the two axes (Npos, Npos) into single dimension
    gbook_flattened = gbook.reshape(Ng, Npos*Npos)
    pbook_flattened = pbook.reshape(nruns, Np, Npos*Npos)
    
    # weights
    if Npatts_train_gp == 'all':
        Wgp = train_gcpc(pbook_flattened, gbook_flattened)
    elif Npatts_train_gp == 'normal':
        Wgp = train_gcpc(pbook_flattened, gbook_flattened, patts_indices=path_locations_flattened[:Npatts_train])
    else:
        Wgp = train_gcpc(pbook, gbook_flattened, patts_indices=path_locations_flattened[:Npatts_train_gp])
    sbook_train = sbook[:,:,:Npatts_train]
    pbook_train = pbook_flattened[:,:,path_locations_flattened[:Npatts_train]]
    if pseudoinverse_iterative:
        Wsp, theta_sp = pseudotrain_3d_iterative(pbook_train, sbook_train, epsilon=epsilon)
        Wps, theta_ps = pseudotrain_3d_iterative(sbook_train, pbook_train, epsilon=epsilon)
    else:
        Wsp = pseudotrain_3d(sbook_train, pbook_train)
        Wps = pseudotrain_3d(pbook_train, sbook_train)
    Wggs = {}
    for axis in [0, 1]:
        for direction in [1, -1]:
            shift = [0, 0]
            shift[axis] = direction
            Wggs[tuple(shift)] = path_integration_Wgg_2d(lambdas, Ng, axis, direction)

    # surprisal
    if normalized == 'sensory_only':
        normalized = False
    if surprisal_method in ['linear', 'exponential', 'exponential_bvc']:
        if surprisal_method == 'linear':
            surprisal_engine = SurprisalLinear(tau, threshold, nruns, normalized, flipped)
        elif surprisal_method == 'exponential':
            surprisal_engine = SurprisalExponential(tau, threshold, nruns, normalized, flipped)
        elif surprisal_method == 'exponential_bvc':
            surprisal_engine = SurprisalExponentialBVC(b, tau, threshold, nruns, normalized, flipped)
        surprisal_engine = SurprisalRegion(surprisal_engine, threshold, return_threshold)
    elif surprisal_method == 'cheat':
        surprisal_engine = SurprisalRegionCheat(nruns, surprisal_enter_indices, surprisal_exit_indices)
    if lite:
        outputs = np.zeros((nruns, 2, Npatts_test)) # store position encoded by g
        outputs[:, :, 0] = None # first position is never seen by model
    else:
        outputs = [np.zeros((nruns, dim, Npatts_test)) for dim in [Np, Ng, Ns]] # store outputs (p, g, s)
    surprisal_history = np.zeros((nruns, Npatts_test)) # store raw surprisal

    # Testing
    # checkpoints = np.arange(Npatts_test, Npatts_test//10)
    positions = range(1, Npatts_test)
    # if verbose:
    #     iterable = tqdm(positions)
    start_time = timeit.default_timer()
    for x in positions: 
        if verbose:
            print(x, f'time so far: {(timeit.default_timer() - start_time) / 60:.1f} min', flush=True)
        # if x in checkpoints:
        #     print(x)
        if x == 1:
            ginit = gbook[(slice(None), *path_locations[0], None)]
        # true (noiseless) sensory pattern
        strue = sbook[(slice(None), slice(None), x, None)]
        disp = tuple(np.array(path_locations[x]) - np.array(path_locations[x-1]))


        # ---- compute surprisal
        start_surprisal = timeit.default_timer()
        if x == 1:
            strue_start = sbook[(slice(None), slice(None), 0, None)] 
            if surprisal_method == 'cheat':
                surprisal_engine.visit(x, update_state=False);
            else:
                surprisal_engine.visit(strue_start.squeeze(axis=-1), update_state=False);
        if surprisal_method == 'cheat':
            surprisal_input = x
        else:
            surprisal_input = strue.squeeze(axis=-1)
        if predict:
            v = (0,) + tuple(-scale_factor * np.array(disp))
            end_current_map, start_new_map, raw_surprisal = surprisal_engine.visit(surprisal_input, v=v)
        else:
            end_current_map, start_new_map, raw_surprisal = surprisal_engine.visit(surprisal_input)
        surprisal_history[:, x] = raw_surprisal.squeeze()
        if np.any(end_current_map) | np.any(start_new_map):
            print(x, np.mean(end_current_map), np.mean(start_new_map))
        end_surprisal = timeit.default_timer()
        if verbose:
            print(x, f'time to compute surprisal: {(end_surprisal - start_surprisal) :.1f} sec', flush=True)


        # ---- run dynamics
        start_dynamics = timeit.default_timer()
        sinit = strue
        Wgg = Wggs[disp]
        results = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                    module_gbooks, module_sizes,
                                    ginit, sinit, start_new_map.reshape(-1, 1, 1), Niter, 
                                    continuous, _2d=True, binary=binary, alpha=alpha, beta=beta) 
        p, g, s = results
        ginit = results[1]
        if lite:
            for run in range(nruns):
                outputs[run, :, x] = grid_to_pos_2d(g[run].squeeze(-1), gbook)
        else:
            for i in range(3):
                outputs[i][:, :, x] = results[i].squeeze()
        end_dynamics = timeit.default_timer()
        if verbose:
            print(x, f'time to run dynamics: {(end_dynamics - start_dynamics) :.1f} sec', flush=True)


        if incremental and x >= Npatts_train:
            start_pseudo = timeit.default_timer()
            if pseudoinverse_iterative and lite:
                if not surprisal_engine.high_surprisal[0]:
                    if incremental_s == 'real':
                        s_obs = strue.squeeze(axis=-1)
                    else:
                        s_obs = s.squeeze(axis=-1)
                    p_obs = p.squeeze(axis=-1)
                    Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, 
                                                            p_obs, s_obs)
                    Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps,
                                                            s_obs, p_obs)
                continue
            if x == Npatts_train:
                # these arrays may have different size later on if surprisals vary between runs
                sbook_obs = [np.zeros((Ns, x)) for i in range(nruns)]
                pbook_obs = [np.zeros((Np, x)) for i in range(nruns)]
                for run in range(nruns):
                    for i, path_location in enumerate(path_locations[:x]):
                        sbook_obs[run][:, i] = sbook[(run, slice(None), i)]
                        pbook_obs[run][:, i] = pbook[(run, slice(None), *path_location)]
            for i in range(nruns):
                if not surprisal_engine.high_surprisal[i]:
                    if incremental_s == 'real':
                        sbook_obs[i] = np.concatenate((sbook_obs[i], strue[i]), axis=1)     
                    else:
                        sbook_obs[i] = np.concatenate((sbook_obs[i], s[i]), axis=1)     
                    pbook_obs[i] = np.concatenate((pbook_obs[i], p[i]), axis=1)
                    if pseudoinverse_iterative:
                        Wsp[i], theta_sp[i] = pseudotrain_2d_iterative_step(Wsp[i], theta_sp[i],
                                                               pbook_obs[i][:, -1],
                                                               sbook_obs[i][:, -1])
                        Wps[i], theta_ps[i] = pseudotrain_2d_iterative_step(Wps[i], theta_ps[i],
                                                               sbook_obs[i][:, -1],
                                                               pbook_obs[i][:, -1])
                    else:
                        Wsp[i] = pseudotrain_2d(sbook_obs[i], pbook_obs[i])
                        Wps[i] = pseudotrain_2d(pbook_obs[i], sbook_obs[i])
            end_pseudo = timeit.default_timer()
            if verbose:
                print(x, f'time to run pseudoinverse learning: {(end_pseudo - start_pseudo) :.1f} sec', flush=True)
    if return_info and not lite:
        encoded_pos = decode_grid(outputs[1], gbook)
        return outputs, Wpg, surprisal_history, [sbook_obs, pbook_obs], encoded_pos
    return outputs