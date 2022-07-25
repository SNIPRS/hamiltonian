import numpy as np
from functools import reduce
import time
from commute import *
from diagonalize import *
from phase import *
from QDrift import *
from hamiltonian import *
import sys

sys.setrecursionlimit(sys.getrecursionlimit()*4)
print(sys.getrecursionlimit())

def Simulation_cost(Chs, CPs):
    """ Returns the cost of the simulation,
    total costs, individual costs in [crzs, toffolis]
    """
    crzs, Toffolis = 0, 0
    icosts = []
    n = len(CPs[0][0])
    for Ch, CP in zip(Chs, CPs):
        _, CZ, _ = diag_results(CP, True)
        print(CZ, CP)
        _, _, cost = logic_min(CZ, Ch)
        crzs += cost[0]
        Toffolis += cost[1]
        icosts.append([cost[0], cost[1]])
    tcost = [crzs, Toffolis]
    return np.array(tcost), np.array(icosts)

def Hs_sum(CHs, Chs):
    """ Regroups the cliques according to coefficients
    """
    tolerance = 1e-3
    SHs, Shs = [], []
    for CH, Ch in zip(CHs, Chs):
        SH, Sh = [], []
        idx = np.argsort(Ch)
        sCH, sCh = np.array(CH)[idx][::-1], np.array(Ch)[idx][::-1]
        while len(sCh) > 0:
            if np.abs(sCh[0]) < tolerance:
                break
            Sh.append(sCh[0])
            SH.append(np.sum(sCH, axis=0))
            sCh = sCh - sCh[0]
            cut = (-1*tolerance > sCh) | (sCh > tolerance)
            sCh = sCh[cut]
            sCH = sCH[cut]
        SHs.append(SH); Shs.append(Sh) 
    return SHs, Shs

def Hs_sum_costs(SHs, Shs, icosts):
    Hs_s, hs_s, costs = [], [], []
    for SH, Sh, icost in zip(SHs, Shs, icosts):
        for H, h in zip(SH, Sh):
            Hs_s.append(H); hs_s.append(h), costs.append(icost)
    return np.array(Hs_s), np.array(hs_s), costs

geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1))]
basis = "sto-3g"
multiplicity = 1
charge = 0

n, Hm, Hs, hs, Ps = Paulis_from_Molecule(None, geometry, basis, multiplicity, charge)



terms = list(zip(Hs, hs, Ps))
terms1 = [list(term) for term in terms]
terms = np.array(terms1, dtype=object)

# Reorder 
terms[[5,9]] = terms[[9,5]]
terms[[10,14]] = terms[[14,10]]
terms[[9,10]] = terms[[10,9]]

# Z
terms[3][:2], terms[4][:2] = -terms[3][:2], -terms[4][:2]
terms[3][2], terms[4][2] = '-'+terms[3][2], '-'+terms[4][2]
# terms[1:5, 1] = np.mean(terms[1:5, 1])

# XY
terms[7][:2], terms[8][:2] = -terms[7][:2], -terms[8][:2]
terms[7][2], terms[8][2] = '-'+terms[7][2], '-'+terms[8][2]

# ZZ
# terms[9:13, 1] = np.mean(terms[9:13, 1])

for i, term in enumerate(terms):
    print(i, term[1:])

def get_clique(i, j):
    return np.sum(np.stack(terms[i:j][:, 0]), axis=0)

SI = get_clique(0,1)
SZ0 = get_clique(1,3)
SZ1 = get_clique(1,5)
SXY = get_clique(5,9)
SZZ0 = get_clique(9,10)
SZZ1 = get_clique(10,11)
SZZ2 = get_clique(9,13)
SZZ3 = get_clique(13,15)

Hs_s = np.array([SZ0, SZ1, SXY, SZZ0, SZZ1, SZZ2, SZZ3])
hs_s = np.array([terms[1][1]-terms[3][1], terms[3][1], terms[5][1], terms[9][1]-terms[11][1], terms[10][1]-terms[11][1], terms[11][1], terms[13][1]])
icosts = np.array([[2,2], [2,2], [1,3], [1,1], [1,1], [1,1], [1,2]])
Hs, hs = Hs[1:], hs[1:]


# Start simulation
t = 1
M = 15

rho = rand_rho(n)
Ns = [2**i + 10 for i in range(6, M)]
st = time.time()

# print(time.time()-st)
errors_costs = np.array([Error_cost(Hm, Hs_s, hs_s, t, rho, N, icosts, M=3000, threads=48) for N in Ns])
errors_costs1 = np.array([Error_cost(Hm, Hs, hs, t, rho, N, M=3000, threads=48) for N in Ns])
errors, errors1 = errors_costs[:, 0], errors_costs1[:, 0]
tcosts, rcosts = errors_costs[:, 1], errors_costs[:, 2]

np.savetxt('save/rcosts-H2.txt', rcosts)
np.savetxt('save/tcosts-H2.txt', tcosts)
np.savetxt('save/errors-H2.txt', errors)
np.savetxt('save/errors1-H2.txt', errors1)
np.savetxt('save/Ns-H2.txt', Ns)

