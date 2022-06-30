import networkx as nx
import numpy as np

Mclique = nx.algorithms.clique.max_weight_clique

def Comm(A, B):
    return np.allclose(A @ B, B @ A)

def Comm_Graph(Matrices):
    n = len(Matrices)
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    
    for i in range(n-1):
        for j in range(i+1, n):
            if Comm(Matrices[i], Matrices[j]):
                G.add_edge(i, j)
    return G

def iGreedy_Cliques(G):
    icliques = []
    while len(G.nodes) > 0:
        clique, _ = Mclique(G, None)
        icliques.append(clique)
        G.remove_nodes_from(clique)
    return icliques

def Greedy_Cliques(Matrices, cs=None, Ps=None):
    """ Input: Matrices 
        Output: (np) List of cliques of commuting matrices, sorted by largest first
    """
    G = Comm_Graph(Matrices)
    icliques = iGreedy_Cliques(G)
    cliques = [Matrices[iclique] for iclique in icliques]
    if cs is not None and Ps is not None:
        coeffs = [cs[iclique] for iclique in icliques]
        paulis = [Ps[iclique] for iclique in icliques]
        return cliques, coeffs, paulis
    return cliques

def Greedy_Insert(A, cliques):
    """ Input: Matrix A, List of cliques (commuting matrices), sorted by largest first
        Output: List of cliques with A intserted in, sorted by largest first
    """
    n = len(Matrices)
    found = None
    fclique = None
    i = 0
    
    while i < n:
        clique = cliques[i]
        comm = True
        for B in clique:
            if not Comm(A, B):
                comm = False
                break
        if comm == True:
            found = i
            fclique = clique
            break
    if found == None:
        cliques.append([A])
        return cliques
    
    del(cliques[i])
    fclique.append(A)
    m = len(fclique)
    inserted = False
    
    for i, clique in enumerate(cliques):
        if len(clique) < m:
            cliques.insert(i, fclique)
            inserted = True
            break
    if not inserted:
        cliques.append(fclique)
    
    return cliques