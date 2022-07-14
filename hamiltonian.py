import numpy as np
from functools import reduce

try:
    import openfermion as of
    import openfermionpyscf as ofpyscf
except ImportError:
    print("Installing OpenFermion and OpenFermion-PySCF...")

from scipy.sparse import linalg

import cirq
import openfermion as of
import openfermionpyscf as ofpyscf

from openfermion.transforms import *
from openfermion.chem import MolecularData
from openfermion.transforms import binary_code_transform
from openfermion.transforms import get_fermion_operator
from openfermion.linalg import eigenspectrum
from openfermion.transforms import normal_ordered, reorder
from openfermion.utils import up_then_down

I = np.identity(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
Paulis = [I, X, Y, Z]
sPaulis = ['I', 'X', 'Y', 'Z']
Paulis_dic = {'I':I, 'X':X, 'Y':Y, 'Z':Z}

I = np.identity(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
Paulis = [I, X, Y, Z]
sPaulis = ['I', 'X', 'Y', 'Z']
Paulis_dic = {'I':I, 'X':X, 'Y':Y, 'Z':Z}

def sigma_aj(n, a, j, s=False):
    paulis = [I for i in range(n)]
    paulis[j] = Paulis[a]
    return reduce(np.kron, paulis)

def Pauli_edge(paul, n, i, j):
    return i*'I' + paul + (j-i-1)*'I' + paul + (n-j-1)*'I'

def Heisenberg(G, Jx, Jy, Jz, h, C=False):
    """ Returns Heisenberg model based on graph G, 
    with edges (i, j) j > i
    """
    m = len(G)
    n = np.max(G)+1
    
    XXs = np.array([sigma_aj(n, 1, i)@sigma_aj(n, 1, j) for i, j in G])
    YYs = np.array([sigma_aj(n, 2, i)@sigma_aj(n, 2, j) for i, j in G])
    ZZs = np.array([sigma_aj(n, 3, i)@sigma_aj(n, 3, j) for i, j in G])
    Zs = np.array([sigma_aj(n, 3, i) for i in range(n)])
    
    PXXs = [Pauli_edge('X', n, i, j) for i, j in G]
    PYYs = [Pauli_edge('Y', n, i, j) for i, j in G]
    PZZs = [Pauli_edge('Z', n, i, j) for i, j in G]
    PZs = ['I'*i + 'Z' + 'I'*(n-i-1) for i in range(n)]
    
    H = 0.5*(Jx*np.sum(XXs, axis=0) + Jy*np.sum(YYs, axis=0)
             + Jz*np.sum(ZZs, axis=0) + h*np.sum(Zs, axis=0))
    Hs = np.concatenate([XXs, YYs, ZZs, Zs], axis=0)
    hs = np.array(m*[0.5*Jx] + m*[0.5*Jy] + m*[0.5*Jz] + n*[0.5*h])
    Ps = PXXs + PYYs + PZZs + PZs
    
    if not C:
        return H, Hs, hs, Ps
    
    iZs, iPZs = [[A] for A in Zs], [[s] for s in PZs]
    CHs = [XXs, YYs, ZZs] + iZs
    Chs = [m*[0.5*Jx], m*[0.5*Jy], m*[0.5*Jz]] + n*[[0.5*h]]
    CPs = [PXXs, PYYs, PZZs] + iPZs
    return H, CHs, Chs, CPs
    
def Heisenberg_1d(n, Jx, Jy, Jz, h, C=False):
    hs = n*[0.5*Jx, 0.5*Jy, 0.5*Jz, 0.5*h]
    Hs = []
    Ps = []
    for j in range(n):
        Hs += [sigma_aj(n, 1, j)@sigma_aj(n, 1, (j+1)%n),
             sigma_aj(n, 2, j)@sigma_aj(n, 2, (j+1)%n),
             sigma_aj(n, 3, j)@sigma_aj(n, 3, (j+1)%n),
             sigma_aj(n, 3, j)]
        Ps += ['I'*j+'XX'+'I'*(n-2-j), 'I'*j+'YY'+'I'*(n-2-j),
              'I'*j+'ZZ'+'I'*(n-2-j), 'I'*j+'Z'+'I'*(n-1-j)]
    Ps[-4:-1] = ['X' + 'I'*(n-2) + 'X', 'Y' + 'I'*(n-2) + 'Y', 'Z' + 'I'*(n-2) + 'Z']
    H = 0.5*np.sum(np.array([Jx*sigma_aj(n, 1, j)@sigma_aj(n, 1, (j+1)%n) +
                  Jy*sigma_aj(n, 2, j)@sigma_aj(n, 2, (j+1)%n) + 
                  Jz*sigma_aj(n, 3, j)@sigma_aj(n, 3, (j+1)%n) +
                  h*sigma_aj(n, 3, j) for j in range(n)]), axis = 0)
    if not C:
        return np.array(H), np.array(Hs), np.array(hs), np.array(Ps) 
    XXs, YYs = [Hs[4*i] for i in range(n)], [Hs[4*i+1] for i in range(n)]
    ZZs = [Hs[4*i+2] for i in range(n)]
    PXXs, PYYs = [Ps[4*i] for i in range(n)], [Ps[4*i+1] for i in range(n)]
    PZZs = [Ps[4*i+2] for i in range(n)]
    Chs = [n*[0.5*Jx], n*[0.5*Jy], n*[0.5*Jz]] + n*[[0.5*h]]
    CHs = [XXs, YYs, ZZs] + [[Hs[4*i+3]] for i in range(n)]
    CPs = [PXXs, PYYs, PZZs] + [[Ps[4*i+3]] for i in range(n)]
    return np.array(H), CHs, Chs, CPs

def Heisenberg_XXX(n, J, g):
    hs = n*[J, g*J]
    Hs = []
    Ps = []
    for j in range(n):
        Hs += [sigma_aj(n, 3, j)@sigma_aj(n, 3, (j+1)%n), 
               sigma_aj(n, 1, j)]
        Ps += ['I'*j+'ZZ'+'I'*(n-2-j), 'I'*j+'X'+'I'*(n-1-j)]
    Ps[-2] = 'Z' + 'I'*(n-2) + 'Z'
    H = 0.5*np.sum(np.array([J*sigma_aj(n, 3, j)@sigma_aj(n, 3, (j+1)%n) + 
                  g*J*sigma_aj(n, 1, j) for j in range(n)]), axis = 0)
    return np.array(H), np.array(Hs), np.array(hs), np.array(Ps) 

def pauli_from_typle(n, paulis):
    matrices = [I for i in range(n)]
    chars = ['I' for i in range(n)]
    for pauli in paulis:
        matrices[pauli[0]] = Paulis_dic[pauli[1]]
        chars[pauli[0]] = pauli[1]
    pauli_string = "".join(chars)
    pauli_matrix = reduce(np.kron, matrices)
    return pauli_string, pauli_matrix

def Paulis_from_Molecule(ham=None, geometry=None, basis=None, 
                         multiplicity=None, charge=None):
    if ham is None:
        ham = ofpyscf.generate_molecular_hamiltonian(geometry, 
                                                     basis, 
                                                     multiplicity, 
                                                     charge)
        ham_ferm_op = of.get_fermion_operator(ham)
    else:
        ham_ferm_op = ham
    ham_jw = of.jordan_wigner(ham_ferm_op)
    dic = ham_jw.terms
    typles = list(ham_jw.terms.keys())
    n = 0
    for typle in typles:
        for pauli in typle:
            n = n if pauli[0] < n else pauli[0]
    n += 1
    Hs, hs, Ps = [], [], []
    for typle in typles:
        pstring, pmatrix = pauli_from_typle(n, typle)
        Hs.append(pmatrix)
        Ps.append(pstring)
        hs.append(dic[typle])
    Hs, hs, Ps = np.array(Hs), np.real(np.array(hs)), np.array(Ps)
    # H = np.sum([hs[i]*Hs[i] for i in range(len(typles))], axis=0)
    H = of.get_sparse_operator(ham_jw).A
    return n, H, Hs, hs, Ps

def LiH_hamiltonian():
    geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))]
    molecule = MolecularData(geometry, 'sto-3g', 1,
                             description="1.45")
    molecule.load()
    molecular_hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices = [0], active_indices = [1,2])
    ham = normal_ordered(get_fermion_operator(molecular_hamiltonian))
    n, H, Hs, hs, Ps = Paulis_from_Molecule(ham)
    return n, H, Hs, hs, Ps
