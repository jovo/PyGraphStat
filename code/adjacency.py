'''
Created on Mar 4, 2012

@author: dsussman
'''
# RandomGraph.py
"""Graphs and Random Graphs
========================

This module provides classes to store graphs and generate random graphs.
It also has methods to manipulate graphs.

Classes:
    Graph -- Parent class that stores Adjacency as well as graph properties
    BlockGraph -- Graph that contains inherent block structure
    RandomGraph -- Parent class for graphs that are generated randomly
    ERGraph -- Erdos-Renyi graphs
    SBMGraph -- Stochastic Blockmodel
    
    RareEventExcpetion"""

from scipy import sparse
import numpy as np
from scipy.sparse import linalg as la

class Graph(object):
    """Parent class with attributes common to all graphs.
    
    Attributes:
        Adj -- Adjacency matrix, stored in sparse.coo_matrix form
        n_nodes -- number of nodes in the graph
        directed -- boolean for whether the graph is directed or not
        loopy -- boolean for whether the graph contains self loops
        
    #TODO: Add stuff to deal with weighted edges"""
    Adj = []
    n_nodes = 0
    directed = True
    loopy = False
    def __init__(self, n, directed, loopy):
        """Constructor for class Graph
        
        Input:
            n -- number of nodes
            directed -- boolean for whether the graph directed or not
            loopy -- boolean for whether to allow self loops
            edge_list -- 2xE array where first row corresponds to the rows of Adj
                         and the second row corresponds to the columns of Adj."""
        self.n_nodes = n
        self.directed = directed
        self.loopy = loopy
    
    @classmethod
    def from_edge_list(cls, n, edge_list, directed, loopy):
        """Generate a graph from an edge list
        
        Input:
            n -- number of nodes
            edge_list -- 2xE array where first row corresponds to the rows of Adj
                         and the second row corresponds to the columns of Adj.
            directed
            loopy"""
        G = Graph(n,directed,loopy)
 
        G.Adj = sparse.coo_matrix((np.ones(edge_list.shape[1]),
                                    (edge_list[0,:],edge_list[1,:])),
                                    shape=(n,n))
        G.check_graph()
        return G
        
    def check_graph(self):
        """Checks that the graph adheres to directed- and loopy-ness"""
        if not self.loopy:
            assert np.sum(np.abs(self.Adj.diagonal()))==0
        if not self.directed:
            assert np.all(np.all(self.Adj.todense()
                                 ==self.Adj.todense().transpose()))
        assert self.Adj.shape[0]==self.n_nodes
        assert self.Adj.shape[1]==self.n_nodes
        assert np.all(self.Adj.data == 1)
        
    def dot_product_embed(self,dim):
        if self.directed:
            out_vec,S,in_vec = la.svds(self.Adj.tocsr(), dim)
            return np.sqrt(S)*out_vec, np.sqrt(S)*in_vec.transpose()
        else:
            evals, evecs = la.eigsh(self.Adj.tocsr(), dim)
            return np.sqrt(np.abs(evals))*evecs


class BlockGraph( Graph ):
    """Class for graphs with inherent block structure
    
    Inherits from Graph class. 
    
    Additional attributes:
        K -- number of blocks
        block_assignment -- array of length n which assigns a block to each node
        nvec -- vector of length K with the number of nodes in each block"""
    block_assignment = np.array([])
    nvec = np.array([]);
    K = np.array((0))
    def __init__(self, n, K, directed, loopy):
        """Initializes the graph and the number of blocks
        
        Inputs:
            n, directed, loopy as in Graph class
            K -- number of blocks"""
        Graph.__init__(self, n, directed, loopy )
        self.K = K

    def assign_block(self):
        """Generates the block assignment vector from self.nvec"""
        assert(self.nvec.size > 0 and np.any(self.nvec > 0))
        self.block_assignment = np.array([])
        for nb in self.nvec:
            self.block_assignment = np.append(self.block_assignment, np.ones(nb))
    
    def set_nvec(self,nvec):
        """Set the number of nodes in each block"""
        if nvec.sum() != self.n_nodes:
            raise GraphException('nvec must sum to n_nodes')
        if nvec.size != self.K:
            raise GraphException('size of nvec must equal K')
        self.nvec = nvec
        self.assign_block()
        
        

class RandomGraph( Graph ):
    """RandomGraph class--Wrapper class for various random graphs
    
    In the future this class may contain methods/variables that are
     useful for all random graphs."""
     
    def generate_adjacency(self): 
        """Classes that inherit from RandomGraph must implement a way to generate
        a new adjacency matrix from the same parameters"""
        raise NotImplementedError()
    
    def check_rare_event(self,prob):
        """Tests for rare events that have probaility at most prob
        
        Not implemented in base class"""
        raise NotImplementedError()
        
    def generate_uniform_adjacency(self, n, m, p, directed, loopy):
        """Helper function Generates a (possibly non-square) binary random matrix
        
        Generates an binary sparse.coo_matrix with each entry independent Berselfn(p)
        Params: n,m --  size of the matrix
                p -- Bernoulli parameter
                directed -- If false then generate symmetric matrix
                loopy -- If false then generates a hollow matrix
        
        To quickly generate the matrix generate geometric random variables to 
        generate the indices for each edge"""
        if n * m == 0:
            return sparse.coo_matrix(0, 1)
        
        idx =  np.floor(-np.log(np.random.rand(p*n*m+1))/p) # Get the jump sizes
        idx[1:] = idx[1:]+1 #add 1 to all jumps that are size 0 after the first 1
        idx = idx.cumsum() # cumulative sum to get the index of each edge
        while idx[-1]<n*m: #if you didn't reach the end, generate more
            idx = np.append(idx,
                            idx[-1]+np.ceil(-np.log(np.random.rand())/p))
        
        col = np.mod(idx[idx<n*m],m) # get column and row for each index
        row = np.floor(idx[idx<n*m]/m)
        
        good_idx = np.ones_like(col)>0  # Numbers which satisfy undirected or unloopy criteria
        if not loopy:
            good_idx = good_idx*(col!=row)
        if not directed:
            good_idx = good_idx*(col>row)
            col = col[good_idx>0]
            row = row[good_idx>0]
            A = sparse.coo_matrix((np.ones(col.size*2),
                                   (np.append(row,col),np.append(col,row))),
                                  shape=(n,m))
            return A
        else:
            col = col[good_idx>0]
            row = row[good_idx>0]
            A = sparse.coo_matrix((np.ones_like(col),(row,col)),shape=(n,m))
            return A

class ERGraph( RandomGraph ):
    """ERGraph class for Erdos-Renyi Random graphs
    
    Can be directed and/or loopy"""
    p=0
    def __init__(self, n, p, directed, loopy):
        """Initialize and ER Random Graph
        
        Parameters:
            n, directed, loopy -- see Graph class
            p --  probability of each edge"""
        Graph.__init__(self, n, directed, loopy)
        self.p = p
        self.generate_adjacency()
        
    def generate_adjacency(self):
        """Generate a new adjacency matrix with params given by attributes"""
        self.Adj = self.generate_uniform_adjacency(
                self.n_nodes, self.n_nodes, self.p, self.directed, self.loopy)
    
    def check_rare_event(self, prob):
        """Raises a RareEventException a graph with probability<prob is present
        
        Tests that the number of edges is approximately p*n.
        Based on Hoeffding bound. """
        t = np.sqrt(-np.log(prob)/(2*self.n_nodes))
        posible_edges = self.n_nodes*(self.n_nodes-1+self.loopy)
        if np.abs(self.Adj.data.sum()/posible_edges - self.p) > t:
            raise RareEventException(
                    prob,'The number of edges is not near its expectation')
        

class SBMGraph( BlockGraph, RandomGraph ):
    """SBMGraph class for Stochastic Block Model Graphs
    
    Generates a random block structure based on probability vector row 
    and assigns edges independently based on block memebership of nodes 
    and the P matrix."""
    P = np.array( [] )
    rho = np.array( [] )
    def __init__( self, n, P, rho, directed, loopy ):
        BlockGraph.__init__( self, n, rho.size, directed, loopy )
        self.P, self.rho, = P, rho
        self.generate_adjacency()

    def generate_adjacency( self ):
        """Generates the adjacency matrix for this stochastic blockmodel
        
        Useful to regenerate an SBMGraph with the same parameters"""
        self.nvec = np.random.multinomial( self.n_nodes, self.rho )
        self.assign_block()
        row, col = np.array( [] ), np.array( [] )
        idx_add = np.append( 0, self.nvec[:-1] ).cumsum()
        for bi in xrange( self.K ):
            if self.nvec[bi]==0:
                continue
            A = self.generate_uniform_adjacency( 
                                self.nvec[bi], self.nvec[bi], self.P[bi, bi],
                                self.directed, self.loopy )
            row = np.append( row, A.row + idx_add[bi] )
            col = np.append( col, A.col + idx_add[bi] )
            for bj in xrange( bi + 1, self.K ):
                if self.nvec[bj]==0:
                    continue
                # Generate the upper right block bi,bj
                A = self.generate_uniform_adjacency( 
                        self.nvec[bi],self.nvec[bj],self.P[bi, bj],
                        directed=True, loopy=True )
                row = np.append( row, A.row + idx_add[bi] )
                col = np.append( col, A.col + idx_add[bj] )
                if not self.directed:
                    row = np.append( row, A.col + idx_add[bj] )
                    col = np.append( col, A.row + idx_add[bi] )
                else:
                    # Generate lower left block bj,bi
                    A = self.generate_uniform_adjacency(
                            self.nvec[bi], self.nvec[bj], self.P[bi, bj],
                            directed=True, loopy=True )
                    row = np.append( row, A.col + idx_add[bj] )
                    col = np.append( col, A.row + idx_add[bi] )
        self.Adj = sparse.coo_matrix(( np.ones_like( row ), ( row, col ) ),
                                     shape=( self.n_nodes, self.n_nodes ) )
        
    def check_rare_event(self, prob):
        t = np.sqrt(-np.log(prob/self.K)/(2*self.n_nodes))
        if np.any(np.abs(self.nvec/self.n_nodes-self.rho)>t):
            raise RareEventException(prob,'Groups sizes and probabilities do not match')
        
class ErrorfulSBMGraph(SBMGraph):
    AdjTrue =  []
    def __init__(self, n, P, rho, eps, z, directed, loopy ):
        SBMGraph.__init__(self, n, P, rho, directed, loopy);
        self.AdjTrue = self.Adj
        self.generate_adjacency(self)
        
    def generate_adjacency(self):
        SBMGraph.generate_adjacency(self)
        

class RareEventException(Exception):
    """Exception class that should occur when something 'rare' happens"""
    prob = 1
    def __init__(self, prob, message=''):
        """Initialize with an upper bound on the probability of this event.
        
        Can also include a custom error message"""
        self.prob = prob
        self.message =( 'An event with probability less than '+str(prob)
                        +' has occurred.\n'+ message)

class GraphException(Exception):
    """Exceptions related to issues with graphs"""
    pass

def main():
    """An example usage of RandomGraph Module
    
    Generates some graphs, does some testing (deprecated)
    and displays a nice SBM"""
    p = .1
    import matplotlib.pyplot as plt
    for n in 10 * ( np.arange( 10 ) + 1 ):
        G =ERGraph( n, p, directed=False, loopy=False )
        A = G.Adj
        assert( A.diagonal().sum() == 0 )
        assert( np.all( A.todense() == A.todense().transpose() ) )

    
    P = np.eye(5) * .1 + np.ones((5, 5))*.1
    rho = np.array(( .1, .1, .2, .2, .4))
    G = SBMGraph(1000, P, rho, directed=False, loopy=False)
    A = G.Adj
    assert np.all(np.all(A.todense() == A.todense().transpose()))
    plt.figure()
    plt.matshow(A.todense())
    lat_vec = G.dot_product_embed(2)
    plt.figure()
    plt.scatter(lat_vec[:,0], lat_vec[:,1])

if __name__ == "__main__":
    main()
