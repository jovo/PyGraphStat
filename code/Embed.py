'''
Created on Feb 19, 2012

@author: dsussman
Copyright (c) 2012 Johns Hopkins University. All rights reserved.
'''

import numpy as np
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import linalg as la

def dot_product_embed(G, d, scaled=True):
    """ Generates an n by d matrix using an svd of the adjacency matrix
    
    Each row 
    """
    A = nx.to_scipy_sparse_matrix(G)
    if scaled:
        u,s,_ = la.svds(A, d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(A, d)
        return u

def normalized_laplacian_embed(G,d, scaled=False):
    L = nx.normalized_laplacian(G)
    if scaled:
        u,s,_ = la.svds(sparse.csr_matrix(L), d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(sparse.csr_matrix(L), d)
    


    