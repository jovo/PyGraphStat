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
    A = nx.to_scipy_sparse_matrix(G)
    if scaled:
        u,s,_ = la.svds(A, d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(A, d)
        return u

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
    L = nx.normalized_laplacian(G)
    if scaled:
        u,s,_ = la.svds(sparse.csr_matrix(L), d)
        return np.dot(u,np.diag(np.sqrt(s)))
    else:
        u,_,_ = la.svds(sparse.csr_matrix(L), d)
        
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
    k_means = KMeans(init='k-means++', k=3, n_init=10)
    x = embed(G, d)
    k_means.fit(x)
    label = k_means.labels_
    if name is not None:
        nx.set_node_attributes(G, name, dict(zip(np.arange(G.number_of_nodes()),label)))

    return label

    