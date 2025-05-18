# corrupt_p01() and topk() from assoc_utils.py
# haven't been converted to numpy so they are missing 

import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter1d
from numpy.random import rand
from numpy.random import randn 

from VectorHaSH.grid_cells.theory_utils import *

# compute pattern correlations
# codebook = gbook/pbook/sbook
def correlation(codebook):
    
    return np.corrcoef(codebook, rowvar=False)


def extend_gbook(gbook, discretize):
    
    return np.repeat(gbook,discretize,axis=1)


def colvolve_1d(codebook, std):
    
    return gaussian_filter1d(codebook, std, mode="constant")


def cont_gbook(gbook, discretize=10, std=1):
    
    gbook = extend_gbook(gbook, discretize)
    gbook = colvolve_1d(gbook, std)
    return gbook


# generate modular grid code
def gen_gbook(lambdas, Ng, Npos):
    
    ginds = np.concatenate([[0], np.cumsum(lambdas[:-1])]) 
    gbook=np.zeros((Ng,Npos))
    for x in range(Npos):
        phis = np.mod(x,lambdas) 
        gbook[phis+ginds,x]=1    
    return gbook


def train_hopfield(pbook, Npatts):
    
    return (1/Npatts)*np.einsum('ijk, ilk->ijl', pbook[:,:,:Npatts], pbook[:,:,:Npatts])


def train_gcpc(pbook, gbook, Npatts=None, patts_indices=None, use_torch=True):
    
    if Npatts:
        gbook, pbook = gbook[:,:Npatts], pbook[:,:,:Npatts]
    elif patts_indices:
        gbook, pbook = gbook[:,patts_indices], pbook[:,:,patts_indices]
    Npatts = pbook.shape[-1]
    if use_torch:
        einsum = torch.einsum
    else:
        einsum = np.einsum
    return (1/Npatts)*einsum('ij, klj -> kil', gbook, pbook)  
    #return np.einsum('ij, klj -> kil', gbook[:,:Npatts], pbook[:,:,:Npatts]) 

def train_gcpc_3d(pbook, gbook, Npatts):
    
    return (1/Npatts)*(gbook[:,:,:Npatts]@np.swapaxes(pbook[:,:,:Npatts],2,1))

def pseudotrain_Wsp(sbook, ca1book, Npatts=None):
    
    if Npatts:
        sbook, ca1book = sbook[:,:Npatts], ca1book[:,:Npatts]
    Npatts = sbook.shape[-1]
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ij, kjl -> kil', sbook, ca1inv) 

def pseudotrain_Wps(ca1book, sbook, Npatts):
    
    sbookinv = np.linalg.pinv(sbook[:, :Npatts])
    return np.einsum('ij, kli -> klj', sbookinv[:Npatts,:], ca1book[:,:,:Npatts])

def pseudotrain_Wpp(ca1book, Npatts):
    
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return (1/Npatts)*np.einsum('ijk, ikl -> ijl', ca1book[:,:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wgp(ca1book, gbook, Npatts):
    
    ca1inv = np.linalg.pinv(ca1book[:, :, :Npatts])
    return np.einsum('ij, ljk -> lik', gbook[:,:Npatts], ca1inv[:,:Npatts,:]) 

def pseudotrain_Wgg(gbook, Npatts):
    
    gbookinv = np.linalg.pinv(gbook)
    return np.einsum('ij, jk -> ik', gbook[:,:Npatts], gbookinv[:Npatts,:])   


def pseudotrain_2d(book1, book2, Npatts=None):
    
    if Npatts:
        book1, book2 = book1[:,:Npatts], book2[:,:Npatts]
    book2inv = np.linalg.pinv(book2)
    return book1 @ book2inv

# book1 is output, book2 is input
def pseudotrain_3d(book1, book2, Npatts=None, patts_indices=None):
    
    if Npatts:
        book1, book2 = book1[:,:,:Npatts], book2[:,:,:Npatts]
    elif patts_indices:
        book1, book2 = book1[:,:,patts_indices], book2[:,:,patts_indices]
    Npatts = book1.shape[-1]
    book2inv = np.linalg.pinv(book2)
    return book1 @ book2inv


def pseudotrain_2d_iterative_step(W, theta, ak, yk):
    
    ak = ak[:, None]
    yk = yk[:, None]
    bk = ((theta@ak) /(1+ak.T@theta@ak)).T
    theta = theta - theta@ak@bk
    W = W + (yk - W@ak)@bk
    return W, theta

def pseudotrain_2d_iterative_initialize(N, M, epsilon):
    
    theta = (1/epsilon**2)*np.eye(M, M)
    W = np.zeros((N, M))
    return W, theta

def pseudotrain_3d_iterative_step(W, theta, ak, yk, use_torch=True, max_norm=100, use_double=True):
    """
    Say input space has dimension Ni, output space has dimension No
    W - weights (Woi), shape (nruns, No, Ni)
    theta - shape (nruns, Ni, Ni)
    ak - new input, shape (nruns, Ni)
    yk - new output, shape (nruns, No)
    """
    original_dtype = W.dtype  # Save original dtype for conversion later if needed

    # Convert to double precision if specified
    if use_double:
        W, theta, ak, yk = W.double(), theta.double(), ak.double(), yk.double()

    ak = ak[:, :, None]
    yk = yk[:, :, None]

    if use_torch:
        ak_norm = torch.norm(ak)
        if ak_norm > 1e3:
            ak = (ak / ak_norm) * max_norm
        bk_norm = torch.norm(ak)
        if bk_norm > 1e3:
            bk = (bk / bk_norm) * max_norm
    
        W_bp, theta_bp = W.clone(), theta.clone()
        # Compute bk with numerical stability
        denom = 1 + (ak.permute(0, 2, 1)) @ theta @ ak
        # denom = torch.where(denom == 0, denom + epsilon, denom)  # Add epsilon to zero denominators
        bk = ((theta @ ak) / denom).permute(0, 2, 1)
            
        # bk = ((theta@ak) / (1+(ak.permute(0, 2, 1))@theta@ak)).permute(0, 2, 1)
    else:
        bk = ((theta@ak) / (1+(ak.transpose(0, 2, 1))@theta@ak)).transpose(0, 2, 1)
    theta = theta - theta@ak@bk
    W = W + (yk - W@ak)@bk
    norm = torch.norm(W)

    if norm > max_norm:
        W = (max_norm / norm) * W
        # print(norm, 'normalized')
        
    if torch.isnan(W).any() or torch.isnan(theta).any() or torch.isnan(ak).any() or torch.isnan(yk).any():
        print(torch.isnan(W).any(), torch.isnan(theta).any(), torch.isnan(ak).any(), torch.isnan(yk).any())
        breakpoint()
        raise ValueError("output contains NaN values")
    if torch.all(torch.eq(W, 0)):
        print('weight all 0s')
    if yk.shape[1]==10 and not torch.all(torch.eq(yk, 0)):
        pass
    # Convert back to original dtype if use_double was True
    if use_double:
        W, theta = W.to(original_dtype), theta.to(original_dtype)

    return W, theta

# M=input dimension, N=output dimension
# resulting weights are (runs, N, M)
def pseudotrain_3d_iterative_initialize(nruns, N, M, epsilon):
    
    theta = np.zeros((nruns, M, M))
    theta[:] = (1/epsilon**2)*np.eye(M, M)
    theta = torch.tensor(theta, device='cuda', dtype=torch.float32)
    
    W = torch.zeros((nruns, N, M)).cuda()
    return W, theta

def pseudotrain_3d_iterative(inputs, outputs, epsilon=0.01, Npatts=None, patts_indices=None, init_only=False):
    # inputs - shape (nruns, Ni, Npatts)
    # outputs - shape (nruns, No, Npatts)
    
    if Npatts:
        inputs, outputs = inputs[:,:,:Npatts], outputs[:,:,:Npatts]
    elif patts_indices:
        inputs, outputs = inputs[:,:,patts_indices], outputs[:,:,patts_indices]
    nruns, M, Npatts = inputs.shape
    N = outputs.shape[1]
    W, theta = pseudotrain_3d_iterative_initialize(nruns, N, M, epsilon)
    if not init_only:
        for i in range(Npatts):
            W, theta = pseudotrain_3d_iterative_step(W, theta, inputs[:, :, i], outputs[:, :, i])
    return W, theta
# def pseudotrain_3d_iterative_initialize(nruns, N, M, epsilon, use_torch=True):
#     if use_torch:
#         theta = (1 / epsilon**2) * torch.eye(M, M).repeat(nruns, 1, 1)
#         W = torch.zeros((nruns, N, M))
#     else:
#         theta = (1 / epsilon**2) * np.eye(M, M)
#         theta = np.repeat(theta[np.newaxis, :, :], nruns, axis=0)
#         W = np.zeros((nruns, N, M))
    
#     return W, theta

# def pseudotrain_3d_iterative(inputs, outputs, epsilon=0.01, Npatts=None, patts_indices=None, init_only=False, use_cuda=True):
#     # inputs - shape (nruns, Ni, Npatts)
#     # outputs - shape (nruns, No, Npatts)
#     if use_cuda:
#         if not inputs.is_cuda:
#             inputs = inputs.cuda()
#         if not outputs.is_cuda:
#             outputs = outputs.cuda()
            
#     if Npatts:
#         inputs, outputs = inputs[:,:,:Npatts], outputs[:,:,:Npatts]
#     elif patts_indices:
#         inputs, outputs = inputs[:,:,patts_indices], outputs[:,:,patts_indices]
#     nruns, M, Npatts = inputs.shape
#     N = outputs.shape[1]
#     W, theta = pseudotrain_3d_iterative_initialize(nruns, N, M, epsilon)
    
#     if use_cuda:
#         if not W.is_cuda:
#             W = W.cuda()
#         if not theta.is_cuda:
#             theta = theta.cuda()
    
#     if not init_only:
#         for i in range(Npatts):
#             W, theta = pseudotrain_3d_iterative_step(W, theta, inputs[:, :, i], outputs[:, :, i])
#     return W, theta

# whether recon or not.
# if it is reconstructable, remove existing connection with g* and update to a new g which is adj. to the prev g.

def corrupt_pmask(Np, pflip, ptrue, nruns):
    
    #flipmask = rand(nruns, Np)>(1-pflip)
    flipmask = rand(*ptrue.shape)>(1-pflip)
    ind = np.argwhere(flipmask == True)  # find indices of non zero elements 
    pinit = np.copy(ptrue) 
    return pinit, ind


# corrupts p when its -1/1 code
def corrupt_p(Np, pflip, ptrue, nruns):
    
    if pflip == 0:
        return ptrue
    pinit, ind = corrupt_pmask(Np, pflip, ptrue, nruns)
    pinit[ind[:,0], ind[:,1]] = -1*pinit[ind[:,0], ind[:,1]] 
    return pinit


def hopfield(pinit, ptrue, Niter, W):
    
    p = pinit
    for i in range (Niter):
        p = np.sign(W@p)
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))


def gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np, modular=True):
    
    m = len(lambdas)
    p = pinit
    for i in range(Niter):
        gin = Wgp@p;
        if modular:
            g = module_wise_NN(gin, gbook[:,:lambdas[-1]], lambdas)  # modular net
        else:
            g = topk_binary(gin, m)         # non modular net
        p = np.sign(Wpg@g); 
    return np.sum(np.abs(p-ptrue), axis=(1,2))/np.sum(np.abs(pinit-ptrue), axis=(1,2))  #(2*Np)


# module wise nearest neighbor
def module_wise_NN(gin, gbook, lambdas, use_torch=False):
    
    size = gin.shape
    g = np.zeros(size)               #size is (Ng, 1)
    i = 0
    for j in lambdas:
        gin_mod = gin[:, i:i+j]           # module subset of gin
        gbook_mod = gbook[i:i+j]
        if use_torch:
            gbook_mod = torch.unique(gbook_mod, 1)
            g_mod = torch_nearest_neighbor(gin_mod, gbook_mod)
        else:
            gbook_mod = np.unique(gbook_mod, axis=1)
            g_mod = nearest_neighbor(gin_mod, gbook_mod)

        g[:, i:i+j, 0] = g_mod
        # print(i, i+j)
        i = i+j
    return g  


# global nearest neighbor
def nearest_neighbor(gin, gbook):
    
    est = np.einsum('ijk, jl -> ikl', gin, gbook)
    maxm = np.amax(est, axis=2)       #(nruns,1)
    g = np.zeros((len(maxm), len(gbook)))
    for r in range(len(maxm)):
        a = np.argwhere(est[r] == maxm[r])
        idx = np.random.choice(a[:,1])
        g[r,:] = gbook[:,idx]; 
    return g

def torch_nearest_neighbor(gin, gbook):
    """
    computes the nearest neighbors of a set of query vectors (gin) 
    from a set of reference vectors (gbook). 
    It finds the reference vector that has the highest dot product value 
    with each query vector, which indicates the closest in terms of 
    angle (or cosine similarity). 
    The function handles cases where there are multiple equally 
    close neighbors by randomly selecting one.
    """
    
    est = torch.einsum('ijk, jl -> ikl', gin, gbook)
    maxm = torch.amax(est, axis=2)       #(nruns,1)
    g = torch.zeros((len(maxm), len(gbook))).to(gin.device)
    for r in range(len(maxm)):
        a = torch.argwhere(est[r] == maxm[r])
        if len(a) == 0:
            raise RuntimeError("No maximum found; this should not happen")
        idx = a[:,1][torch.randperm(len(a[:,1]))][0]
        g[r,:] = gbook[:,idx]; 
    return g


# return topk sparse code by setting 
# topk to 1 and all other values to zero
def topk_binary(gin, k):
    
    idx = np.argsort(gin, axis=1)
    idx = idx[:,-k:]
    idx = np.squeeze(idx)   # nruns x k
    g = np.zeros_like(gin) 
    nruns = gin.shape[0]   
    if k==1:
        g[np.arange(nruns),idx] = 1 
    else:               
        for i in range(k):
            g[np.arange(nruns),idx[:,i]] = 1
    return g


def default_model(lambdas, Ng, Np, pflip, Niter, Npos, gbook, Npatts_lst, nruns):
    # avg error over Npatts
    
    err_hop = -1*np.ones((len(Npatts_lst), nruns))
    err_gcpc = -1*np.ones((len(Npatts_lst), nruns))
    
    Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
    pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)

    k=0
    for Npatts in Npatts_lst:
        W = np.zeros((nruns, Np, Np));      # plastic pc-pc weights
        Wgp = np.zeros((nruns, Ng, Np));    # plastic pc-to-gc weights

        # Learning patterns 
        W = train_hopfield(pbook, Npatts)
        Wgp = train_gcpc(pbook, gbook, Npatts)

        # Testing
        sum_hop = 0
        sum_gcpc = 0 
        for x in range(Npatts): 
            ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
            pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
            cleanup = hopfield(pinit, ptrue, Niter, W)      # pc-pc autoassociative cleanup  
            sum_hop += cleanup
            cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas)   # pc-gc autoassociative cleanup
            sum_gcpc += cleanup
        err_hop[k] = sum_hop/Npatts
        err_gcpc[k] = sum_gcpc/Npatts
        k += 1   

    return err_hop, err_gcpc    

def sparse_rand(nruns, Np, Ng, sparsity):
    
    W = -1*np.ones((nruns, Np,Ng))
    W[:, :, :sparsity] = 1
    #shuffles at each position in-place 
    # (random shuffling across neurons) 
    for j in range(nruns):
        for i in range(Np):
            np.random.shuffle(W[j,i,:])     
    return W

def capacity_gcpc(lambdas, Ng, Np_lst, pflip, Niter, Npos, gbook, nruns):
    
    # avg error over Npatts
    Npatts_lst = np.arange(1,Npos+1)
    #Npatts_lst = [21]
    err_gcpc = -1*np.ones((len(Np_lst), len(Npatts_lst), nruns))

    l=0
    for Np in Np_lst:
        print("l = ",l)

        Np = 200
        Wpg = randn(nruns, Np, Ng);           # fixed random gc-to-pc weights
        #Wpg = sparse_rand(nruns, Np, Ng, 3)
        pbook = np.sign(np.einsum('ijk,kl->ijl', Wpg, gbook))  # (nruns, Np, Npos)  

        # corr = pbook[0].T@pbook[0] / Np
        # corr = correlation(pbook[0])
        # # # print(corr)
        # plt.figure()
        # plt.imshow(corr)
        # plt.colorbar()
        # plt.show()
        # exit()

        k=0
        for Npatts in Npatts_lst:
            print("k = ",k)
            Wgp = np.zeros((nruns, Ng, Np));        # plastic pc-to-gc weights

            #Wgp = train_gcpc(pbook_grid, gbook_grid, np.prod(lambdas))
            Wgp = train_gcpc(pbook, gbook, Npatts)  # Training
            #Wgp = pseudotrain_Wgp(pbook, gbook, Npatts)

            # capacity for randomly sampled patterns
            #sampledpatt = np.random.choice(range(Npos), size=(Npatts), replace=False)

            # Testing
            sum_gcpc = 0 
            for x in range(Npatts): 
            #for x in range(Npos):
            #for x in sampledpatt:
                ptrue = pbook[:,:,x,None]                       # true (noiseless) pc pattern
                pinit = corrupt_p(Np, pflip, ptrue, nruns)      # make corrupted pc pattern
                cleanup = gcpc(pinit, ptrue, Niter, Wgp, Wpg, gbook, lambdas, Np)   # pc-gc autoassociative cleanup
                #if cleanup[0] > 0:
                    #print(cleanup[0], x)
                    #print(gbook[:,x,None].T)
                sum_gcpc += cleanup
                #print(sum_gcpc[0])
            err_gcpc[l,k] = sum_gcpc/Npatts
            #print("Error; ",err_gcpc[0,k,0])
            k += 1   
        l += 1    
    return err_gcpc                 

