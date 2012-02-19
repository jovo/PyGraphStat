"""
DPE.py

Created by Joshua Vogelstein on 2012-01-22.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""


import networkx as nx
import numpy as np
from numpy import random
from numpy import math

def add_random_edges_between(G, nodes1, p, nodes2 = None):
    """Generate random edges between specified nodes with probability p
    
    If nodes2 is None then just generate random undirected edges between nodes1
    If nodes2 is not None then generate random directed edges between nodes1 and nodes2
    
    This code is mostly taken from nx.generators.fast_gnp_random_graph. It generalizes
    that code to the situation of generating independent identically distributed 
    
    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes, 
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    directed = nodes2!=None

    v = 0  # Nodes in graph are from 0,n-1 (this is the first node index).
    w = -1
    lp = np.log(1.0 - p)
    n1 = len(nodes1)

    if directed:
        n2 = len(nodes2) 
        loop = not np.array_equal(nodes1, nodes2)
        while v < n1: 
            lr = np.log(1.0 - random.random()) 
            w = w + 1 + int(lr/lp) 
            if not loop or v == w: # avoid self loops 
                w = w + 1 
            while  w >= n2 and v < n1:
                w = w - n2
                v = v + 1
                if not loop or v == w: # avoid self loops
                    w = w + 1
            if v < n1:
                G.add_edge(nodes1[v],nodes2[w])
    else:
        while v < n1:
            lr = math.log(1.0 - random.random())
            w = w + 1 + int(lr/lp)
            while w >= v and v < n1: 
                w = w - v 
                v = v + 1
            if v < n1:
                G.add_edge(nodes1[v],nodes1[w])
    return G

def ER(n,p,seed=None,directed=False):
    """ Return a graph sampled from an Erdos-Renyi model
    
    Parameters
    ----------
    n : int
        The number of nodes.
    p : float
        Probability for edge creation.
    seed : int, optional
        Seed for random number generator (default=None). 
    directed : bool, optional (default=False)
        If True return a directed graph 
  
    """

    if p<0:
        raise ValueError('p<0')
    if p>1:
        raise ValueError('p>1')

    G=nx.gnp_random_graph(n,p)
    return G
    # print G.adjacency_matrix
    

# add some aliases to common names
erdos_renyi_graph=ER
ErdosRenyi=ER


def SBM(nvec,block_probs, directed=True, seed=None):
    """Return a graph sampled from a stochastic block model
    
    Parameters
    ----------
    nvec : array [k,1]
        The number of vertices per block; there are k blocks.
    B : array [k,k] in (0,1)^{k x k}
        Probability for edge creation for each block.
    seed : int, optional
        Seed for random number generator (default=None). 
    
      math
    Notes
    -----
    
    WILL ADD THESE FEATURES LATER
    directed : bool, optional (default=True)
        If True return a directed graph 
    
    loopy : bool, optional (default=True)
        If True return a loopy graph
    
    This algorithm iterates over blocks, samples a binomial for each block, 
    and then randomly assigns edges within the block.
    """
    
    if (block_probs<0).any():
        raise ValueError('some probability is <0')
    if (block_probs>1).any():
        raise ValueError('some probability is >1')
    
    if not seed is None:
        np.random.seed(seed)
    
    Nvertices=nvec.sum()        # total number of vertices
    Nblocks=len(nvec)             # number of groups
    if directed:
        G=nx.empty_graph(Nvertices,create_using=nx.DiGraph())
    else:
        G=nx.empty_graph(Nvertices,create_using=nx.Graph())
    block_idx = np.append(0, nvec).cumsum()
    
    
    for ii in xrange(Nblocks):
        nodes1 = np.arange(block_idx[ii],block_idx[ii+1])
        if directed:
            add_random_edges_between(G, nodes1, block_probs[ii,ii],nodes1)
        else:
            add_random_edges_between(G, nodes1, block_probs[ii,ii])
            
        for jj in xrange(ii+1,Nblocks):
            nodes2 = np.arange(block_idx[jj],block_idx[jj+1])
            if directed:
                add_random_edges_between(G, nodes1, block_probs[ii,jj],nodes2)
                add_random_edges_between(G, nodes2, block_probs[jj,ii],nodes1)
            else:
                add_random_edges_between(G, nodes1, block_probs[ii,jj],nodes2)

    return G
    
# add some aliases to common names
stochastic_block_model=SBM

def affiliation_model(n, k, p, q, seed=None):
    block_probs = q*np.ones((k,k))+(p-q)*np.eye(k);
    nvec = n*np.ones((k,1),dtype = np.int)
    
    return SBM(nvec, block_probs, seed) 
