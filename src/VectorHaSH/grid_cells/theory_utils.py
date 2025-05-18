import numpy as np
from numpy.random import shuffle 
import random
import torch

# computes rank of a codebook
def rank(codebook):    
    return np.linalg.matrix_rank(codebook)

# computes rank of subsets of codebook
# all consecutive subsets until idx
def rank_sub(codebook, idx=25):
    
    rank_part = np.zeros((idx))
    for i in range(idx): 
      rank_part[i] = np.linalg.matrix_rank(gbook[:,:i+1])
    return rank_part

# shuffles grid code at each position 
# in-place (random shuffling across neurons)         
def shuffle_gcode(gbook, Npos):
    for i in range(Npos):
        shuffle(gbook[:,i]) 
    return gbook

# shuffles gbook across positions
def shuffle_gbook(gbook):   
    gbook = permutation(np.transpose(gbook))   # need to transpose since permutation only shuffles along first index
    gbook = np.transpose(gbook)                # transpose back 
    return gbook

# scale matrix values to lie between 0 and 1
def normmat(W):
    normW = W - np.min(W)
    normW = normW / np.max(normW)
    return normW