"""
DPE.py

Created by Joshua Vogelstein on 2012-01-22.
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
"""


import networkx as nx
import numpy as np
from numpy import random
from numpy import math
from random import sample
from itertools import combinations

def add_random_edges_between(G, nodes1, p, nodes2 = None):
    """Generate random edges between specified nodes in a networkx graph w/probability p
    
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
    if p == 0:
        return G
    directed = nodes2!=None

    v = 0  # Nodes in graph are from 0,n-1 (this is the first node index).
    w = -1
    lp = np.log(1.0 - p)
    n1 = len(nodes1)

    if directed:
        n2 = len(nodes2) 
        loop = not np.array_equal(nodes1, nodes2) # avoid self loops
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
    loopy : bool, optional (default=True)
        If True return a loopy graph
    
    This algorithm iterates over pairs of blocks and then assigns edges uniformly at random
    between nodes in each block
    """
    
    if (block_probs<0).any():
        raise ValueError('some probability is <0')
    if (block_probs>1).any():
        raise ValueError('some probability is >1')
    if np.shape(block_probs)[0] != len(nvec):
        raise ValueError('nvec must be of length equal to the number of columns/rows of block_probs')
    
    if seed:
        np.random.seed(seed)
    
    Nvertices=nvec.sum()        # total number of vertices
    Nblocks=len(nvec)             # number of groups
    if directed:
        G=nx.empty_graph(Nvertices,create_using=nx.DiGraph())
    else:
        G=nx.empty_graph(Nvertices,create_using=nx.Graph())
    block_idx = np.append(0, nvec).cumsum()
    block = np.zeros(Nvertices, dtype=np.int)
    
    for ii in xrange(Nblocks):
        nodes1 = np.arange(block_idx[ii],block_idx[ii+1])
        block[block_idx[ii]:block_idx[ii+1]] = ii
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

    nx.set_node_attributes(G, 'block', dict(zip(np.arange(Nvertices), block)))
    return G
    
# add some aliases to common names
stochastic_block_model=SBM

def get_errorful_subgraph(G, z, eps):
    """Generate a graph with z edges observed errorfully from the original graph G
    
    Let z=good+bad where bad ~ Binom(z,eps)
    good eges will be randomly selected from the edges in the original graph
    bad edges will be randomlu generated uniformly at random
    """
    bad = random.binomial(z,eps)
    good = z-bad
    n = G.number_of_nodes()
    
    edges = sample(nx.edges(G), good)
    edges.extend(sample(list(combinations(xrange(n),2)), bad))
    return nx.from_edgelist(edges, create_using=nx.Graph(G))
    
    

def affiliation_model(n, k, p, q, seed=None):
    """ Generates a random undirected affiliation model graph with n*k vertices
    
    Parameters
    ----------
    n -- number of nodes in each block
    k -- number of blocks
    p -- probability of edges within a block
    q -- probability of edges between blocks
    seed -- possible seed for random numbers
    """ 
    block_probs = q*np.ones((k,k))+(p-q)*np.eye(k);
    nvec = n*np.ones((k,1),dtype = np.int)
    
    return SBM(nvec, block_probs, seed) 

class RandomGraphGenerator(object):
    """A parent class for random graph generators
    
    Children of this class must implement generate_random_graph and get_param_dict
    methods.
    """
    
    def __init__(self):
        """WARNING: Not Implemented"""
        raise NotImplementedError("Base class is not implemented serves only as a place holder")
    
    def iter_graph(self, nmc):
        """A generator that will generate nmc instances of the random graph"""
        for _ in xrange(nmc):
            yield self.generate_graph()
    
    def generate_graph(self):
        """WARNING NOT IMPLEMENTED!! Generates a random graph"""
        raise NotImplementedError("Base class is not implemented serves only as a place holder")
    
    def get_param_dict(self):
        """WARNING NOT IMPLEMENTED!! Returns a dict of the params"""
        raise NotImplementedError("Base class is not implemented serves only as a place holder")

class ERGenerator(RandomGraphGenerator):
    """Generator for Erdos-Renyi random graph"""
    p = None
    nnodes = None
    directed = None
    
    def __init__(self, p, nnodes, directed=False):
        self.p = p
        self.nnodes = nnodes
        self.directed = directed
    
    def generate_graph(self):
        """Generates an Erdos-Renyi random graph."""
        return ER(self.nnodes, self.p, directed=self.directed)
        
    
    def get_param_dict(self):
        """Returns a dict of the params for the Erdos-Renyi Graph"""
        return {'nnodes':self.nnodes, 'p':self.p, 'directed':self.directed}

class SBMGenerator(RandomGraphGenerator):
    block_prob = None
    nvec = None
    directed = None
    label = None
    
    def __init__(self, block_prob, nvec, directed=False):
        self.block_prob = block_prob
        self.nvec = nvec
        self.directed = directed
        self.label = np.concatenate([k*np.ones(nvec[k]) for k in xrange(len(nvec))])
    
    def generate_graph(self):
        """Generates an Erdos-Renyi random graph."""
        return SBM(self.nvec, self.block_prob, directed=self.directed)
    
    def get_param_dict(self):
        """Returns a dict of the params for the Erdos-Renyi Graph"""
        return {'nvec':self.nvec, 'block_prob':self.block_prob, 'directed':self.directed}

class AffilationGenerator(RandomGraphGenerator):
    p = None
    q = None
    nodes_per_block = None
    k = None
    directed = None
    
    def __init__(self,n,k,p,q, directed=False):
        self.p, self.q,self.nodes_per_block,self.k = (p,q,n,k)
        self.directed = directed
        
    def generate_graph(self):
        return affiliation_model(self.nodes_per_block, self.k, self.p, self.q)
    
    def get_param_dict(self):
        return {'nodes_per_block':self.nodes_per_block, 'k':self.k, 'p':self.p, 'q':self.q}