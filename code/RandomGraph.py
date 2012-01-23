import networkx as nx
import numpy as np

def ER(n,p,seed=None,directed=False):    # write Fibonacci series up to n
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


def SBM(nvec,block_probs,seed=None):
	"""Return a graph sampled from a stochastic block model
	
    Parameters
    ----------
    nvec : array [k,1]
        The number of vertices per block; there are k blocks.
    B : array [k,k] in (0,1)^{k x k}
        Probability for edge creation for each block.
	seed : int, optional
	    Seed for random number generator (default=None). 
    
      
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
	
	Nvertices=nvec.sum() 		# total number of vertices
	Ngroups=len(nvec) 			# number of groups
	# Nedges=np.array((Ngroups,Ngroups)) 	# number of edges per block
	
	# G=nx.empty_graph(Nvertices,create_using=nx.DiGraph())
	# G.name="stochastic block model random graph sample with %s vertices and %s groups"%(Nvertices,Ngroups)
	AdjMat=np.zeros((Nvertices,Nvertices))

	for ii in xrange(Ngroups):
		for jj in xrange(Ngroups):
			blockSize=nvec[ii]*nvec[jj] 								# number of potential edges in the block
			block=np.zeros((nvec[ii],nvec[jj])) 						# make an empty block
			Nedges=np.random.binomial(blockSize,block_probs[ii,jj]) 	# sample number of edges
			edgelist=np.random.permutation(blockSize) 					# sample Nedges randomly from the set
			edgelist=edgelist[0:Nedges]
			block.flat[edgelist]=1 										# assign each edge in edgelist = 1 to block
			if ii==0: 													# put block in adjacency matrix
				startrow=0
			else:
				startrow=sum(nvec[0:ii])
			if jj==0:
				startcol=0
			else:
				startcol=sum(nvec[0:jj])
			
			AdjMat[startrow:startrow+nvec[ii],startcol:startcol+nvec[jj]]=block
			
	G=nx.to_networkx_graph(AdjMat,create_using=nx.DiGraph()) 
	return G
	
# add some aliases to common names
stochastic_block_model=SBM
