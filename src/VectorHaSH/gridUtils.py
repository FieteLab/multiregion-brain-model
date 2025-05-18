# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:50:59 2022

@author: murra
"""
import numpy as np
import torch
import itertools
import random

from VectorHaSH.grid_cells.assoc_utils_np import train_gcpc, pseudotrain_3d_iterative, pseudotrain_3d_iterative_step, module_wise_NN, pseudotrain_Wps, pseudotrain_Wsp, gen_gbook, pseudotrain_3d_iterative_initialize
from VectorHaSH.grid_cells.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, configurated_path_integration_Wgg_2d, module_wise_NN_2d, path_integration_Wgg_1d
from VectorHaSH.grid_cells.sensory_remap import dynamics_surprisal_remap

epsilon=0.01

def create_binary_mask(Wps, percentage):
    """
    Create a binary mask of the same shape as Wps with a specified percentage of entries set to 1.
    
    Parameters:
    Wps (numpy.ndarray): Input matrix.
    percentage (float): Percentage of entries to be set to 1 (between 0 and 100).
    
    Returns:
    numpy.ndarray: Binary mask with the same shape as Wps.
    """
    # Ensure the percentage is a valid value
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100.")
    
    # Calculate the number of elements to be set to 1
    shape = Wps.shape
    total_elements = np.prod(shape)
    num_ones = int((percentage / 100) * total_elements)
    
    # Create a flat mask with the correct number of ones
    flat_mask = np.zeros(total_elements, dtype=int)
    flat_mask[:num_ones] = 1
    
    # Shuffle the mask to distribute the ones randomly
    np.random.shuffle(flat_mask)
    
    # Reshape the flat mask to the shape of Wps
    binary_mask = flat_mask.reshape(shape)
    
    return binary_mask

def generate_bernoulli_tensor(Ns, p, device=None):
    """
    Generate a tensor with each entry being a Bernoulli random variable with probability p.

    Parameters:
    Ns (int): Size of the tensor (shape will be [1, Ns, Ns]).
    p (float): Probability of success for the Bernoulli distribution.
    device (torch.device, optional): Device to create the tensor on. Default is None, which means the tensor will be created on the CPU.

    Returns:
    torch.Tensor: Tensor of shape [1, Ns, Ns] with Bernoulli random variables.
    """
    if device is None:
        device = torch.device('cpu')  # Default to CPU if no device is specified

    # Create a tensor of probabilities with the same shape
    prob_tensor = torch.full((1, Ns, Ns), p, device=device, dtype=torch.float32)

    # Generate the Bernoulli tensor
    bernoulli_tensor = torch.bernoulli(prob_tensor)

    return bernoulli_tensor

def initialize_p_masks(p_size, proportion_g, proportion_s, device, permutation=False):
    assert proportion_g + proportion_s == 1, "The proportions must sum to 1."
    mask_pg = torch.zeros(p_size, 1, dtype=torch.float32, device=device)
    mask_ps = torch.zeros(p_size, 1, dtype=torch.float32, device=device)
    
    num_g = int(proportion_g * p_size)
    num_s = int(proportion_s * p_size)
    
    # Randomly select indices for g and s
    if permutation:
        indices = torch.randperm(p_size)
        g_indices = indices[:num_g]
        s_indices = indices[num_g:num_g + num_s]
    
        for idx in g_indices:
            mask_pg[idx, 0] = 1
        
        for idx in s_indices:
            mask_ps[idx, 0] = 1
    else:
        mask_pg[:num_g, 0] = 1
        mask_ps[num_g:, 0] = 1
    return mask_pg, mask_ps

def generate_permutated_directions(lambdas):
    """
    generate a list of tuples of permutation among [-1, 0, 1].
    """
    # Generate all combinations of [-1, 0, 1] for the current length
    return list(itertools.product([-1, 0, 1], repeat=len(lambdas)))

def generate_all_grid_configuration(grid_assignment, dirs_combo):
    """
    Generate configuration according to grid_assignment.
    Configuration is of format [(module 1 axis, module 1 direction), (), ()...(module n axis, module n direction)]
    """
    axis_map = {'position': 0, 'evidence': 1, 
                'context': 1,
                'both': 1,}
    configuration = [(axis_map[assignment], direction) for assignment, direction in zip(grid_assignment, dirs_combo)]
    return configuration

def grid_cell_initial(Np, Ns, lambdas, cuda=False, dimension=2, grid_step_size=1):
    nruns = 1
    
    if dimension== 2:
        Ng = np.sum(np.dot(lambdas, lambdas))
    else:
        Ng = np.sum(lambdas)
    Npos = np.prod(lambdas)
    module_sizes = [i**dimension for i in lambdas]

    if cuda:
        # a list of identity matrices, each with size size_i by size_i
        module_gbooks = [torch.eye(i).cuda() for i in module_sizes]
    else:
        module_gbooks = [np.eye(i) for i in module_sizes]
    
    print("Gen Gbook 2D")
    if dimension == 2:
        gbook = gen_gbook_2d(lambdas, Ng, Npos)
    else:
        gbook = gen_gbook(lambdas, Ng, Npos)

    Wggs = {}
    axis = 0
    print("Path Integration")
    for axis in range(dimension):
        for direction in [-1, 0, 1]:
            if dimension == 2:
                Wggs[direction] = path_integration_Wgg_2d(lambdas, Ng, axis, direction * grid_step_size)
            else:
                Wggs[direction] = path_integration_Wgg_1d(lambdas, Ng, axis, direction * grid_step_size)

    Wpg = np.random.randn(nruns, Np, Ng)      
    print("Get PBook")
    if cuda:
        gbook = torch.Tensor(gbook).cuda()
        Wpg = torch.Tensor(Wpg).cuda()
        if dimension == 2:
            pbook = torch.einsum('ijk,klm->ijlm', Wpg, gbook)
        else:
            pbook = torch.einsum('ijk,km->ijm', Wpg, gbook)

        pbook[pbook>0] = 1
        pbook[pbook<0] = -1
        torch.cuda.empty_cache()
    else:
        if dimension == 2:
            pbook = np.sign(np.einsum('ijk,klm->ijlm', Wpg, gbook))  # (nruns, Np, Npos, Npos)
        else:
            pbook = np.sign(np.einsum('ijk,km->ijm', Wpg, gbook))  # (nruns, Np, Npos, Npos)

    gbook_flattened = gbook.reshape(Ng, int(Npos**dimension))
    pbook_flattened = pbook.reshape(nruns, Np, int(Npos**dimension))
    print("Train GCPC")
    Wgp = train_gcpc(pbook_flattened, gbook_flattened, use_torch=cuda)
    # breakpoint()
    
    if cuda:
        if dimension == 2:
            ginit = torch.unsqueeze(torch.unsqueeze(gbook[:,0,0], 0), -1)
        else:
            ginit = torch.unsqueeze(torch.unsqueeze(gbook[:,0], 0), -1)
        sbook_obs = torch.zeros((Ns, 1)).cuda()
        pbook_obs = torch.zeros((Np, 1)).cuda()
        torch.cuda.empty_cache()
    else:
        if dimension == 2:
            ginit = np.expand_dims(np.expand_dims(gbook[:, 0, 0], 0), -1)
        else:
            ginit = np.expand_dims(np.expand_dims(gbook[:, 0], 0), -1)
        sbook_obs = np.zeros((Ns, 1))
        pbook_obs = np.zeros((Np, 1))
        
    pbook_train = pbook_flattened[:,:,0:1]
    pinit = pbook_train
    if cuda:
        pbook_train = pbook_train.cpu().numpy()

    sbook_train = np.zeros((1, Ns, 1)) # dummy
    
    print("PseudoTrain 3D Iterative")
    Wsp, theta_sp = pseudotrain_3d_iterative(pbook_train, sbook_train, epsilon=epsilon, init_only=True)
    Wps, theta_ps = pseudotrain_3d_iterative(sbook_train, pbook_train, epsilon=epsilon, init_only=True)
    
    # Wps, Wsp = np.zeros((nruns, Np, Ns)), np.zeros((nruns, Ns, Np))
    if cuda:
        Wps = torch.Tensor(Wps).cuda()
        Wsp = torch.Tensor(Wsp).cuda()
        Wggs = {k: torch.Tensor(v).cuda() for k, v in Wggs.items()}
        theta_sp = torch.Tensor(theta_sp).cuda()
        theta_ps = torch.Tensor(theta_ps).cuda()

        torch.cuda.empty_cache()
    return Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, ginit, sbook_obs, pbook_obs, theta_sp, theta_ps

def run_grid_cells(sensory, action, Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes,  
                   prev_g, sbook_obs, pbook_obs, theta_sp, theta_ps, cuda=False, is_eval=False, force_prev=False, verbose=False, allocated_grids=None, gap=0, recon_only=False):
    if action == 0 or action == 1:
        action = 0
    elif action == 2: # go straight
        action = 1
    Wgg = Wggs[action]

    p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                       module_gbooks, module_sizes,
    #                                       Wgg@prev_g, sensory, 1, Niter=1, 
                                       prev_g, sensory, 1, Niter=1, 
                                       continuous=True, _2d=True,
                                       binary=True, use_torch=cuda)
    if recon_only:
        return p_recon, g_recon, s

    if verbose:
        cos_sim = (sensory * s).sum() / (s.norm() * sensory.norm() + 1e-12)
        cos_dist = 1 - cos_sim


    if force_prev:
        p = p_recon
        g = g_recon
        
        # To check whether a new sensory input is referred to already associated input.
        if allocated_grids is not None and not is_eval:
            compare = g_recon == allocated_grids
            compare = compare.all(1).nonzero()
            while len(compare) > 0:
                breakpoint()
                print(compare)
                idx = int(compare[-1,0])
                step = gap + len(allocated_grids) - idx
                Wgg = Wggs[1]
                new_g = prev_g
                step = 1 #Niter
                for _ in range(step):
                    new_g = Wgg @ new_g
                print(f"Duplicate grid code, Shift {step} step")
                p_recon, g_recon, s = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                                   module_gbooks, module_sizes,
                                                   new_g, sensory, 0, Niter=1, 
                                                   continuous=True, _2d=True,
                                                   binary=True, use_torch=cuda)
                compare = g_recon == allocated_grids
                compare = compare.all(1).nonzero()

            p = p_recon
            g = g_recon

        p = Wpg @ g
        if cuda:
            p = p.sign()
        else:
            p = np.sign(p)


        if verbose:
            print(force_prev)
            print("USE_PREV", cos_dist, sensory.view(-1), s.view(-1), (sensory-s).max(), g.view(-1).nonzero().cpu().tolist())

    else:
        surprisal = 0
        # when init
        p, g, s = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                           module_gbooks, module_sizes,
                                           prev_g, sensory, surprisal, Niter=1, 
                                           continuous=True, _2d=True,
                                           binary=True, use_torch=cuda)
    
    if not is_eval and not torch.all(sensory == 0):
        Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0], cuda)
        Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0], p[:,:,0], cuda)

    if not force_prev:
        p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                           module_gbooks, module_sizes,
                                           g, sensory, 0, Niter=1, 
                                           continuous=True, _2d=True,
                                           binary=True, use_torch=cuda)
        if verbose:
            print("USE NEW", cos_dist, sensory.view(-1), s.view(-1), (sensory-s).max(), g.view(-1).nonzero().cpu().tolist())

    if cuda:
        sbook_obs = torch.cat((sbook_obs, sensory[0]), 1)
        pbook_obs = torch.cat((pbook_obs, p[0]), 1)
    else:
        sbook_obs = np.concatenate((sbook_obs, sensory[0]), axis=1)
        pbook_obs = np.concatenate((pbook_obs, p[0]), axis=1)

        
    return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps



def reconstruct(Wgp, Wpg, Wsp, Wps,
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
    if use_torch:
        sign_fn = torch.sign
    else:
        sign_fn = np.sign
    fn = torch.relu
    for i in range(Niter):
        if binary:
            if with_recurrence:
                p = fn(surprisal * Wps@s + (1-surprisal) * Wpg@g + Wpp@prev_p)
            else:
                # print(f'surprisal {surprisal}, Wps {Wps.shape}, s {s.shape}, Wpg {Wpg.shape}, g {g.shape}')
                p = fn(surprisal * Wps@s + (1-surprisal) * Wpg@g)
        else:
            p = fn(alpha * Wps@s + beta*(1-surprisal) * Wpg@g)
        gin = Wgp@p
        if _2d:
            g = module_wise_NN_2d(gin, gbooks, module_sizes, use_torch=use_torch)
        else:
            g = module_wise_NN(gin, gbooks[:,:module_sizes[-1]], module_sizes, use_torch=use_torch)
        s = Wsp@p
        if not continuous:
            s = sign_fn(s)
    return p, g, s

def grid_cell_initial_2d(Np, Ns, lambdas, cuda=False, dimension=2, 
                         grid_assignment=None, new_model=False, with_mlp=False, with_recurrence=False):
    print('calling init')
    nruns = 1
    
    if dimension== 2:
        Ng = np.sum(np.dot(lambdas, lambdas))
    else:
        Ng = np.sum(lambdas)
    Npos = np.prod(lambdas)
    module_sizes = [i**dimension for i in lambdas]

    if cuda:
        module_gbooks = [torch.eye(i).cuda() for i in module_sizes]
    else:
        module_gbooks = [np.eye(i) for i in module_sizes]
    
    print("Gen Gbook 2D")
    if dimension == 2:
        gbook = gen_gbook_2d(lambdas, Ng, Npos)
    else:
        gbook = gen_gbook(lambdas, Ng, Npos)
    Wggs = {}
    axis = 0
    print("Path Integration")
    # if default or assignment all `both`
    if grid_assignment is None or all(x == 'both' for x in grid_assignment):
        for axis in range(dimension):
            for direction in [-1, 0, 1]:
                if dimension == 2:
                    Wggs[(axis, direction)] = path_integration_Wgg_2d(lambdas, Ng, axis, direction)
                else:
                    Wggs[(axis, direction)] = path_integration_Wgg_1d(lambdas, Ng, axis, direction)
    else:
        all_dirs_combo = generate_permutated_directions(lambdas)
        for direction in all_dirs_combo:
            # Generate module configuration for this direction tuple, 
            ## grid_assignment fixes axis as axis_map = {'position': 0, 'evidence': 1, 'context':1}
            if dimension == 2:
                module_configs = generate_all_grid_configuration(grid_assignment, direction)
                # # if it's a new model using MLP prediction) and this module encodes evidence, we will not be updating this module manually (thus an identity mtx)
                Wggs[direction] = configurated_path_integration_Wgg_2d(lambdas, Ng, module_configs, no_evidence_update=with_mlp)
            else:
                assert 'need to implement'
        special_position_update_config = []
        if 'both' in grid_assignment:
            for x in grid_assignment:
                if x == 'both':
                    # axis, dir
                    special_position_update_config.append((0,1))
                else:
                    special_position_update_config.append((0,0))
            Wggs['position_forward_only'] = configurated_path_integration_Wgg_2d(lambdas, 
                                                                                Ng, 
                                                                                special_position_update_config)
    
    Wpg = torch.randn(nruns, Np, Ng)      
    print("Get PBook")
    if cuda:
        gbook = torch.Tensor(gbook).cuda()
        Wpg = torch.Tensor(Wpg).cuda()
        if dimension == 2:
            pbook = torch.einsum('ijk,klm->ijlm', Wpg, gbook)
        else:
            pbook = torch.einsum('ijk,km->ijm', Wpg, gbook)

        pbook[pbook>0] = 1
        pbook[pbook<0] = -1
        torch.cuda.empty_cache()
    else:
        if dimension == 2:
            pbook = np.sign(np.einsum('ijk,klm->ijlm', Wpg, gbook))  # (nruns, Np, Npos, Npos)
        else:
            pbook = np.sign(np.einsum('ijk,km->ijm', Wpg, gbook))  # (nruns, Np, Npos, Npos)

    gbook_flattened = gbook.reshape(Ng, int(Npos**dimension))
    pbook_flattened = pbook.reshape(nruns, Np, int(Npos**dimension))
    print("Train GCPC")
    Wgp = train_gcpc(pbook_flattened, gbook_flattened, use_torch=cuda)
    # breakpoint()
    if cuda:
        if dimension == 2:
            ginit = torch.unsqueeze(torch.unsqueeze(gbook[:,0,0], 0), -1)
        else:
            ginit = torch.unsqueeze(torch.unsqueeze(gbook[:,0], 0), -1)
        sbook_obs = torch.zeros((Ns, 1)).cuda()
        pbook_obs = torch.zeros((Np, 1)).cuda()
        torch.cuda.empty_cache()
    else:
        if dimension == 2:
            ginit = np.expand_dims(np.expand_dims(gbook[:, 0, 0], 0), -1)
        else:
            ginit = np.expand_dims(np.expand_dims(gbook[:, 0], 0), -1)
        sbook_obs = np.zeros((Ns, 1))
        pbook_obs = np.zeros((Np, 1))

    pbook_train = pbook_flattened[:,:,0:1]
    pinit = pbook_train
    # if cuda:
    #     pbook_train = pbook_train.cpu().numpy()

    sbook_train = np.zeros((1, Ns, 1)) # dummy
    # sbook_train = torch.randn(1, Ns, 1)  # drawing from normal distribution
    
    print("PseudoTrain 3D Iterative")
    Wsp, theta_sp = pseudotrain_3d_iterative(pbook_train, sbook_train, epsilon=epsilon, init_only=True)
    Wps, theta_ps = pseudotrain_3d_iterative(sbook_train, pbook_train, epsilon=epsilon, init_only=True)
    # breakpoint()
    # Wsp, theta_sp = pseudotrain_3d_iterative(pbook_train, sbook_train, epsilon=epsilon)
    # Wps, theta_ps = pseudotrain_3d_iterative(sbook_train, pbook_train, epsilon=epsilon)
    # breakpoint()
    # Wps, Wsp = np.zeros((nruns, Np, Ns)), np.zeros((nruns, Ns, Np))
    
    if with_recurrence:
        Wpp, theta_pp = pseudotrain_3d_iterative_initialize(1, Np, Np, epsilon=epsilon)
    
    if cuda:
        Wps = torch.Tensor(Wps).cuda()
        Wsp = torch.Tensor(Wsp).cuda()
        Wggs = {k: torch.Tensor(v).cuda() for k, v in Wggs.items()}
        theta_sp = torch.Tensor(theta_sp).cuda()
        theta_ps = torch.Tensor(theta_ps).cuda()

        torch.cuda.empty_cache()
    if with_recurrence:
        return Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, ginit, sbook_obs, pbook_obs, theta_sp, theta_ps, Wpp, theta_pp
    else:
        return Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, ginit, sbook_obs, pbook_obs, theta_sp, theta_ps

def run_grid_cells_2d(sensory, action, Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, 
                   prev_g, sbook_obs, pbook_obs, theta_sp, theta_ps, cuda=False, 
                   is_eval=False, force_prev=False, verbose=False, allocated_grids=None, 
                   gap=0, recon_only=False,
                   with_recurrence=False, Wpp=None, theta_pp=None, prev_p=None):
    """
    With force_prev == False:
        1. Path integration g (self-loop) -> p -> g and s
        2. g -> p -> g and s
    """
    if type(action) is tuple:
        Wgg = Wggs[action]
    elif type(action) is str and action == 'trigger':
        Wgg = Wggs['position_forward_only']
    
    # (not used) with surprisal==1, `reconstruct()` does s -> p -> g and s
    if recon_only or force_prev:
        p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                        module_gbooks, module_sizes,
                                        ginit=prev_g, sinit=sensory, surprisal=1, Niter=1, 
                                        continuous=True, _2d=True,
                                        binary=True, use_torch=cuda,
                                        with_recurrence=with_recurrence, Wpp=Wpp, theta_pp=theta_pp, prev_p=prev_p)
        if verbose:
            cos_sim = (sensory * s).sum() / (s.norm() * sensory.norm() + 1e-12)
            cos_dist = 1 - cos_sim

    if recon_only:
        return p_recon, g_recon, s

    # with surprisal==0, `d_s_r()` does g (self-loop) -> p -> g and s
    # p, g, s = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
    #                                     module_gbooks, module_sizes,
    #                                     ginit=prev_g, sinit=sensory, surprisal=0, Niter=1, 
    #                                     continuous=True, _2d=True,
    #                                     binary=True, use_torch=cuda)
    g = Wgg@prev_g # path integrate advances g
    if with_recurrence:
        p = torch.relu(Wpg@g + Wpp@prev_p)
    else:
        p = torch.relu(Wpg@g)
    gin = Wgp@p
    g = module_wise_NN_2d(gin, module_gbooks, module_sizes, use_torch=True)
    s = Wsp@p

    # update Wsp, Wps during training  
    if not is_eval and not (sensory == 0).all():
        Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0], cuda)
        Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0], p[:,:,0], cuda)
    
    if with_recurrence and not is_eval and not (prev_p == 0).all() and not (p == 0).all():
        Wpp, theta_pp = pseudotrain_3d_iterative_step(Wpp, theta_pp, prev_p[:,:,0], p[:,:,0], cuda, max_norm=50)    

    """comment back depn Jaedong's reply"""
    if not force_prev:
        # with surprisal==0, `reconstruct()` does g -> p -> g and s
        p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                           module_gbooks, module_sizes,
                                           g, sensory, surprisal=0, Niter=1, 
                                           continuous=True, _2d=True,
                                           binary=True, use_torch=cuda,
                                           with_recurrence=with_recurrence, Wpp=Wpp, theta_pp=theta_pp, prev_p=prev_p)
        if verbose:
            print("USE NEW", cos_dist, sensory.view(-1), s.view(-1), (sensory-s).max(), g.view(-1).nonzero().cpu().tolist())

    if cuda:
        sbook_obs = torch.cat((sbook_obs, sensory[0]), 1)
        pbook_obs = torch.cat((pbook_obs, p[0]), 1)
    else:
        sbook_obs = np.concatenate((sbook_obs, sensory[0]), axis=1)
        pbook_obs = np.concatenate((pbook_obs, p[0]), axis=1)
        
    if with_recurrence:
        return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps, Wpp, theta_pp
    else:
        return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps

def run_grid_cells_2d_new_model(sensory, action, Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, 
                   prev_g, sbook_obs, pbook_obs, theta_sp, theta_ps, cuda=False, 
                   is_eval=False, force_prev=False, verbose=False, allocated_grids=None, 
                   gap=0, recon_only=False, W_L=None, W_R=None, mask_pg=None, mask_ps=None, new_model=True, with_mlp=False,
                   with_recurrence=False, Wpp=None, theta_pp=None, prev_p=None):
    """
    With force_prev == False:
        1. s = W_L @ s_R + W_R @ s_L
        2. \hat{p} = \hat{Wps} @ s
        3. Path integration g = Wgg @ g (position only)
        4. p = ReLU(Wpg @ g + \hat{p}) (all place cells receive MEC signals, selective subpop retains sensory signals as well)
        5. learn W_sp, W_ps using s, p; s = Wsp @ p 
        6. g = Wgp @ p (should this be here? then info from (4) is only present in p)
        
    * New Version * Some portion of p receives info from s, and the other portion of p receives info from g, exclusively. 
    It's essentially a concatnation at step 4.
        1. s = W_L @ s_R + W_R @ s_L
        2. p_s = Wps @ s
        3. Path integration g = Wgg @ g (position only)
        3.5 p_g = Wpg @ g
        4. p = ReLU(mask_pg*p_g + mask_ps*p_s) (all place cells receive MEC signals, selective subpop retains sensory signals as well)
        5. learn W_sp, W_ps using s, p; s = Wsp @ p 
        6. g = Wgp @ p (should this be here? then info from (4) is only present in p)
    """
    obs = sensory.squeeze()
    length = len(obs) // 2

    # UPDATE: change to concat s_r and s_l, and make a block matrxix
    # Create block matrix W
    W = torch.zeros((2 * length, 2 * length), device=W_L.device)
    W[:length, :length] = W_R
    W[length:, length:] = W_L
    obs = obs.view(1, 2 * length, 1)
    sensory = W@obs

    # 1. s = W_L @ s_R + W_R @ s_L
    # left_fov = obs[:length].view(1, length, 1)
    # right_fov = obs[length:].view(1, length, 1)
    # sensory =  W_L@right_fov + W_R@left_fov 
    
    if type(action) is tuple:
        Wgg = Wggs[action]
    elif type(action) is str and action == 'trigger':
        Wgg = Wggs['position_forward_only']
     
        
    # 2. p_s = Wps @ s
    p_s = Wps @ sensory

    # 3. Path integration g = Wgg @ g (position only)
    g = Wgg@prev_g
    
    # 3.5 p_g = Wpg @ g
    p_g = Wpg @ g
    
    # 4. p = ReLU(mask_pg*p_g + mask_ps*p_s) (all place cells receive MEC signals, selective subpop retains sensory signals as well)
    # p = torch.relu(Wpg@g) + torch.relu(hat_p)
    # p = torch.relu(mask_pg*p_g) + torch.relu(mask_ps*p_s)
    # p = torch.relu(torch.relu(p_g) * torch.relu(p_s))
    # p = torch.relu(p_g + p_s)
    # p = torch.relu(mask_pg*p_g+mask_ps*p_s)
    if new_model:
        if with_recurrence:
            recurrence_term = Wpp @ prev_p
            p = torch.relu(p_g+p_s+recurrence_term)
        else:
            p = torch.relu(p_g+p_s)
    else:
        if with_recurrence:
            recurrence_term = Wpp @ prev_p
            p = torch.relu(p_g+recurrence_term)
        else:
            p = torch.relu(p_g)
    # p = torch.clamp(p, max=1e3)
    
    # 5. learn W_sp, W_ps using s, p; s = Wsp @ p 
    if not is_eval and not (sensory == 0).all() and not with_mlp:
        Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0], cuda)
        Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0], p[:,:,0], cuda)
    if with_recurrence and not is_eval and not (prev_p == 0).all() and not (p == 0).all():
        Wpp, theta_pp = pseudotrain_3d_iterative_step(Wpp, theta_pp, prev_p[:,:,0], p[:,:,0], cuda, max_norm=50)    
    
    s = Wsp @ p
    
    # 6. g = Wgp @ p (should this be here? then info from (4) is only present in p)
    # g = module_wise_NN_2d(Wgp @ p, module_gbooks, module_sizes, use_torch=cuda)
    
    if cuda:
        sbook_obs = torch.cat((sbook_obs, sensory[0]), 1)
        pbook_obs = torch.cat((pbook_obs, p[0]), 1)
    else:
        sbook_obs = np.concatenate((sbook_obs, sensory[0]), axis=1)
        pbook_obs = np.concatenate((pbook_obs, p[0]), axis=1)
    
    if torch.isnan(p).any() or torch.isnan(g).any():  
        breakpoint()  
    
    if with_recurrence:
        return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps, p_s, Wpp, theta_pp
    else:
        return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps, p_s

def run_grid_cells_2d_context(sensory, action, Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, 
                   prev_g, sbook_obs, pbook_obs, theta_sp, theta_ps, cuda=False, 
                   is_eval=False, W_L=None, W_R=None, path_integration=True):
    """
        1. s = W_L @ s_R + W_R @ s_L
        2. p_s = Wps @ s
        3. g_in = Wgp @ p
           g = module_wise_NN_2d(g_in, gbooks, module_sizes, use_torch=use_torch)
        OR 
        3. Path integration g = Wgg @ g (position only)
        4. p_g = Wpg @ g
        5. learn W_sp, W_ps using sensory, p_g; 
    """
    obs = sensory.squeeze()
    length = len(obs) // 2

    # UPDATE: change to concat s_r and s_l, and make a block matrxix
    # Create block matrix W
    W = torch.zeros((2 * length, 2 * length), device=W_L.device)
    W[:length, :length] = W_R
    W[length:, length:] = W_L
    obs = obs.view(1, 2 * length, 1)
    sensory = W@obs

    # 1. s = W_L @ s_R + W_R @ s_L
    # left_fov = obs[:length].view(1, length, 1)
    # right_fov = obs[length:].view(1, length, 1)
    # sensory =  W_L@right_fov + W_R@left_fov 
    
    if type(action) is tuple:
        Wgg = Wggs[action]
    elif type(action) is str and action == 'trigger':
        Wgg = Wggs['position_forward_only']
     
    # 2. p_s = Wps @ s
    p_s = torch.relu(Wps @ sensory)
    
    # 3. update grid code
    if path_integration:
        g = Wgg @ prev_g # (position only)
    else:
        g_in = Wgp @ p_s
        g = module_wise_NN_2d(g_in, module_gbooks, module_sizes, use_torch=True)

    # 4. p_g = Wpg @ g
    p = torch.relu(Wpg @ g)
    
    # 5. learn W_sp, W_ps using s, p_g; s = Wsp @ p_g
    if not is_eval and not (sensory == 0).all():
        Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0], cuda)
        Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0], p[:,:,0], cuda)
    
    s = Wsp @ p
    # breakpoint()

    if cuda:
        sbook_obs = torch.cat((sbook_obs, sensory[0]), 1)
        pbook_obs = torch.cat((pbook_obs, p[0]), 1)
    else:
        sbook_obs = np.concatenate((sbook_obs, sensory[0]), axis=1)
        pbook_obs = np.concatenate((pbook_obs, p[0]), axis=1)
    
    if torch.isnan(p).any() or torch.isnan(g).any():  
        breakpoint()  
        
    return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps, p_s

