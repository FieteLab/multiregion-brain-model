"""
All functions are for 2D environments
"""
import numpy as np
import torch
import random
from VectorHaSH.grid_cells import assoc_utils_np as assoc_utils

# --- grid cells
def gen_gbook_2d(lambdas, Ng, Npos):
    """
    Return grid codebook (grid activity vector for each position)

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        Npos - int, number of spatial positions in each axis
    
    Outputs:
        gbook - np.array, size (Ng, Npos, Npos)
            gbook[:, a, b] = grid vector at position (a, b)
    """
    
    # Ng = np.sum(np.dot(lambdas, lambdas))
    # Npos = np.prod(lambdas)
    gbook = np.zeros((Ng, Npos, Npos))
    for x in range(Npos):
        for y in range(Npos):
            index = 0
            for period in lambdas:
                phi1, phi2 = x % period, y % period
                gpattern = np.zeros((period, period))
                gpattern[phi1, phi2] = 1
                gpattern = gpattern.flatten()
                gbook[index:index+len(gpattern), x, y] = gpattern
                index += len(gpattern)
    return gbook

def shift_matrix(size, shift_amount):
    """
    Return permutation matrix that shifts entries in a vector

    Inputs:
        size - int, size of permutation matrix
        shift_amount - int, number of positions to shift entries
            Example: for size=3, shift_amount=1,
                vector [0, 1, 0] will be permuted to [0, 0, 1]
                for shift_amount=-1, 
                vector [0, 1, 0] will be permuted to [1, 0, 0]
    
    Outputs:
        mat - np.array, size (size, size)
    """
    
    mat = np.eye(size)
    if shift_amount > 0:
        mat = np.concatenate([mat[-shift_amount:], mat[:size-shift_amount]], axis=0)
    else:
        shift_amount = -shift_amount
        mat = np.concatenate([mat[shift_amount:], mat[:shift_amount]], axis=0)
    return mat

def module_Wgg_axis1(period, shift_amount):
    """
    Helper function for path_integration_Wgg()
    Return matrix for single grid module, moving in axis=1

    Example: for period=2, shift_amount=1, returns
        array([[0., 1., 0., 0.],
               [1., 0., 0., 0.],
               [0., 0., 0., 1.],
               [0., 0., 1., 0.]])
    """
    
    n = period**2
    W = np.zeros((n, n))
    for i in range(period):
        W[i*period:(i+1)*period, 
         i*period:(i+1)*period] = shift_matrix(period, shift_amount)
    return W

def path_integration_Wgg_1d(lambdas, Ng, axis, direction):
    """
    Return weight matrix for recurrent connections in grid cells
    that perform path integration

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        axis - int (0 or 1), axis to move along
        direction - int, amount to move
    
    Outputs:
        Wgg - np.array, size (Ng, Ng)
    """
    
    Wgg = np.zeros((Ng, Ng))
    idx = 0
    for period in lambdas:
        n = period
        mat = shift_matrix(n, direction)
        Wgg[idx:idx+n, idx:idx+n] = mat
        idx += n
    assert idx == Ng
    return Wgg

def path_integration_Wgg_2d(lambdas, Ng, axis, direction):
    """
    Return weight matrix for recurrent connections in grid cells
    that perform path integration

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        axis - int (0 or 1), axis to move along
        direction - int, amount to move
    
    Outputs:
        Wgg - np.array, size (Ng, Ng)
    """
    
    Wgg = np.zeros((Ng, Ng))
    idx = 0
    for period in lambdas:
        n = period**2
        if axis == 1:
            mat = module_Wgg_axis1(period, direction)
        else:
            mat = shift_matrix(n, direction*period)
        Wgg[idx:idx+n, idx:idx+n] = mat
        idx += n
    assert idx == Ng
    return Wgg

def configurated_path_integration_Wgg_2d(lambdas, Ng, module_configs, no_evidence_update=False):
    """
    Return weight matrix for recurrent connections in grid cells
    that perform path integration, according to `module_configs`.

    Inputs:
        lambdas - list[int], grid periods
        Ng - int, number of grid cells
            should equal to sum of period squared
        module_configs - list[tuple], each tuple contains (axis, direction) for the cooresponding module, where
            axis - int (0 or 1), axis to move along
            direction - int, amount to move
    
    Outputs:
        Wgg - np.array, size (Ng, Ng)
    """
    
    assert len(lambdas) == len(module_configs), "Each grid module must have a corresponding axis and direction"
    
    Wgg = np.zeros((Ng, Ng))
    idx = 0
    for period, (axis, direction) in zip(lambdas, module_configs):
        n = period**2
        # Generate Wgg for each module based on its specific axis and direction
        if axis == 1: # usually this is for evidence axis
            # if it's a new model (using MLP prediction) and this module encodes evidence, we will not be updating this module manually (thus an identity mtx)
            mat = module_Wgg_axis1(period, direction) if not no_evidence_update else np.identity(period**2)
        else:
            mat = shift_matrix(n, direction*period)
        Wgg[idx:idx+n, idx:idx+n] = mat
        idx += n
    assert idx == Ng, "Combined matrix dimensions do not match the total number of grid cells"
    return Wgg


def module_wise_NN_2d(gin, module_gbooks, module_sizes, use_torch=False):
    """
    Given raw grid activity vector, return the nearest grid vector in grid codebook
    Nearest neighbor is done module by module

    Inputs:
        gin - np.array, raw grid activity vector, size (Ng, 1)
        module_gbooks - list[np.array], 
            each element is the collection of possible grid vectors for a module, 
            has size (module_size, number of possible grid vectors)
        module_sizes - list[int], size of each grid module
            note this size is not the grid period, but the number of cells (=period^2)

    Outputs:
        g - np.array, grid vector after taking nearest neighbor
    """
    
    size = gin.shape
    if use_torch:
        g = torch.zeros(size, device=gin.device)
    else:
        g = np.zeros(size)               #size is (Ng, 1)
    i = 0
    for j, gbook_mod in zip(module_sizes, module_gbooks):
        gin_mod = gin[:, i:i+j]           # module subset of gin
        if use_torch:
            g_mod = assoc_utils.torch_nearest_neighbor(gin_mod, gbook_mod)
        else:
            g_mod = assoc_utils.nearest_neighbor(gin_mod, gbook_mod)
        g[:, i:i+j, 0] = g_mod
        i = i+j
    return g  

# only diff with 1d: need to pass in full grid codebook to module_wise_NN()
def gcpc_2d(pinit, ptrue, Niter, Wgp, Wpg, module_gbooks, lambdas, Np, modular=True):
    
    m = len(lambdas)
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        if modular:
            g = module_wise_NN_2d(gin, module_gbooks, lambdas)  # modular net
        else:
            g = assoc_utils.topk_binary(gin, m)         # non modular net
        p = np.sign(Wpg@g); 
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))  #(2*Np)

# note pflip must be above 0, otherwise gcpc() will divide by 0
def gcpc_capacity_2d(Npatts_lst, nruns=5, Np=350,
             lambdas=None, gbook=None, pflip=0.2, Niter=15, modular=True):
    
    if gbook is None:
        Ng = np.sum(np.dot(lambdas, lambdas))
        Npos = np.prod(lambdas)
        gbook = gen_gbook_2d(lambdas, Ng, Npos).reshape(Ng, Npos*Npos)
    Ng = gbook.shape[0]
    module_sizes = [i**2 for i in lambdas]

    err_hop = -1*np.ones((len(Npatts_lst), nruns))
    err_gcpc = -1*np.ones((len(Npatts_lst), nruns))

    Wpg = np.random.randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    module_gbooks = [np.eye(i) for i in module_sizes] # TODO: make this more general

    k=0
    for Npatts in Npatts_lst:
        print(Npatts)
        W = np.zeros((nruns, Np, Np));      # plastic pc-pc weights
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights

        # Learning patterns 
        W = assoc_utils.train_hopfield(pbook, Npatts)
        Wgp = assoc_utils.train_gcpc(pbook, gbook, Npatts)

        # Testing
        sum_hop = 0
        sum_gcpc = 0 
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
            pinit = assoc_utils.corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            cleanup = assoc_utils.hopfield(pinit, ptrue, Niter, W)      # pc-pc autoassociative cleanup  
            sum_hop += cleanup
            cleanup = gcpc_2d(pinit, ptrue, Niter, Wgp, Wpg, module_gbooks, module_sizes, Np, modular)   # pc-gc autoassociative cleanup
            sum_gcpc += cleanup
        err_hop[k] = sum_hop/Npatts
        err_gcpc[k] = sum_gcpc/Npatts
        k += 1
    return err_hop, err_gcpc