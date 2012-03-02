'''
Created on Feb 19, 2012

@author: dsussman
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
'''

import numpy as np
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import linalg as la
from sklearn.cluster import KMeans
from itertools import product

def adjacency_matrix(G):
    return np.array(nx.adjacency_matrix(G))
adjacency_sparse = nx.to_scipy_sparse_matrix

def laplacian_sparse(G):
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_matrix(G)
    degree =  A*np.ones(n)
    scale = sparse.lil_matrix((n,n))
    scale.setdiag([np.sqrt(1.0/deg) if deg!=0 else 0 for deg in degree])
    #scale = np.array([np.sqrt(d**-1) if d!=0 else 0 for d in degree])
    return scale*A*scale

def laplacian_matrix(G):
    n = G.number_of_nodes()
    A = np.array(nx.adjacency_matrix(G))
    degree = np.dot(A,np.ones(n))
    scale = [np.sqrt(1.0/deg) if deg!=0 else 0 for deg in degree]
    return np.dot(np.diag(scale),np.dot(A, np.diag(scale)))

def self_matrix(G):
    """A function for embedding if G is already stored in matrix form"""
    return G

class Embed(object):
    dim = None
    
    sval = None
    svec = None
    
    matrix = None
    G = None
    
    def __init__(self, dim, matrix=adjacency_matrix):
        self.dim = dim
        self.matrix = matrix
    
    
    def __check_dim(self,d):
        if d<1:
            raise ValueError('Dimension must be >=1')
        if d>self.dim:
            raise ValueError('Dimension must be <=self.dim')
        
    def embed(self, G, fast=True):
        if not fast or self.G is not G:
            self.G = G
            self.svec,self.sval,_ = la.svds(self.matrix(G), self.dim)
            self.sval = self.sval[::-1]
            self.svec = self.svec[:, ::-1]
        
    def get_embedding(self, d, scale=None):
        if scale:
            return self.get_scaled(d)
        else:
            return self.get_unscaled(d)
    
    def get_unscaled(self, d=None):
        if not d:
            d=self.dim
        
        self.__check_dim(d)
          
        return self.svec[:,np.arange(d)]
        
    def get_scaled(self, d=None):
        if not d:
            d=self.dim
        self.__check_dim(d)
        
        return np.dot(self.svec[:,np.arange(d)], 
                    np.diag(np.sqrt(self.sval[np.arange(d)])))
        
class EmbedIter(object):
    d = []
    matrix = []
    embed = None
    
    def __init__(self, d_range, matrices, scales):
        self.d = d_range
        self.matrix = matrices
        self.scale = scales
        self.embed = [Embed(np.max(self.d), m) for m in self.matrix]
        
        
    def get_embedding(self, G):
        for ( d,embed, scale) in product(self.d, self.embed, self.scale):
            embed.embed(G)
            yield embed.get_embedding(d, scale)
        
        
    



#
#class DotProductEmbed(Embed):
#    sparse = None
#    
#    def __init__(self, dim, sparse=Falscipy_sparse_matrix(se):
#        self.dim = dim
#        self.sparse = sparse
#    
#    def embed(self, G, scaled=True):
#        assert(isinstance(G, nx.Graph))
#        if self.sparse:
#            A = nx.to_scipy_sparse_matrix(G)
#        else:
#            A = nx.adjacency_matrix(G)
#        self.svecs,self.svals,_ = la.svds(A, self.dim)
#
#class LaplacianEmbed(Embed):
#    sparse = None
#    
#    def __init__(self, dim, sparse=False):
#        self.dim = dim
#        self.sparse = sparse
#    
#    def embed(self, G, scaled=True):
#        assert(isinstance(G, nx.Graph))
#        n = G.number_of_nodes()
#        if self.sparse:
#            A = nx.to_scipy_sparse_matrix(G)
#            degree =  A*np.ones(n)
#            scale = sparse.lil_matrix((n,n))
#            scale.setdiag([np.sqrt(1.0/deg) if deg!=0 else 0 for deg in degree])
#            #scale = np.array([np.sqrt(d**-1) if d!=0 else 0 for d in degree])
#            L = scale*A*scale
#        else:
#            A = nx.adjacency_matrix(G)
#            degree = np.dot(A,np.ones(n))
#            scale = [np.sqrt(1.0/deg) if deg!=0 else 0 for deg in degree]
#            L = np.dot(np.diag(scale),np.dot(A, np.diag(scale)))
#            
#        self.svecs,self.svals,_ = la.svds(L, self.dim)
#            
#            
    

def dot_product_embed(G, d, scaled=True):
    """ Generates an n by d matrix using an svd of the adjacency matrix
    
    Each row of the output corresponds to a node (ordered according to G.node)
    so that each node is assigned a vector in d-dimensional euclidean space.
    
    Parameters
    ----------
    G -- networkx graph
    d -- embedding dimension
    scaled -- whether to scaled the embedding by the square root
             of the eigenvalues (default=True) 
             
    Returns
    -------
    n times d matrix where n=G.number_of_nodes()
    """
    A = adjacency_matrix(G)
    if scaled:
        u,s,_ = la.svds(A, d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(A, d)
        return u

dot_product_embed_unscaled = lambda G,d: dot_product_embed(G,d,scaled=False)

def normalized_laplacian_embed(G,d, scaled=False):
    """ Generates an n by d matrix using an svd of the normalized laplacian
    
    Each row of the output corresponds to a node (ordered according to G.node)
    so that each node is assigned a vector in d-dimensional euclidean space.
    
    Parameters
    ----------
    G -- networkx graph
    d -- embedding dimension
    scaled -- whether to scaled the embedding by the square root
             of the eigenvalues (default=False) 
             
    Returns
    -------
    n times d matrix where n=G.number_of_nodes()
    """
#    n = G.number_of_nodes()
#    A = nx.to_scipy_sparse_matrix(G)
#    degree =  A*np.ones(n)
#    
#    scale = sparse.lil_matrix((n,n))
#    scale.setdiag([np.sqrt(1.0/deg) if deg!=0 else 0 for deg in degree])
#    
#    L = scale*A*scale
    L  = laplacian_matrix(G)
    
    if scaled:
        u,s,_ = la.svds(sparse.csr_matrix(L), d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(sparse.csr_matrix(L), d)
        return u
        
normalized_laplacian_embed_scaled = lambda G,d:normalized_laplacian_embed(G, d, scaled=True)
        
def cluster_vertices_kmeans(G, embed, d, k, name=None):
    """ Clusters vertices into k groups based on an embedding in d dimensions
    
    Parameters
    ----------
    G -- networkx graph
    embed -- embedding method takes 2 paramets, Graph and dimension
    d --  embedding dimension
    k -- number of clusters
    name -- if not None then assign labels as attribute name to each node
    """
    k_means = KMeans(init='k-means++', k=k, n_init=10)
    x = embed(G, d)
    k_means.fit(x)
    label = k_means.labels_
    if name is not None:
        nx.set_node_attributes(G, name, dict(zip(np.arange(G.number_of_nodes()),label)))

    return label

    