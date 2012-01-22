import networkx as nx

def ER(n,p):    # write Fibonacci series up to n
    G=nx.gnp_random_graph(n,p)
    return G
	# print G.adjacency_matrix