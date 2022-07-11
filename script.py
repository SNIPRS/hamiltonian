import numpy as np
from functools import reduce
import time
from commute import *
from diagonalize import *
from phase import *
from QDrift import *
from hamiltonian import *


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

n, Hm, Hs, hs, Ps = LiH_hamiltonian()

terms = list(zip(Hs, hs, Ps))
terms1 = [list(term) for term in terms]
terms = np.array(terms1, dtype=object)
terms = terms[np.abs(terms[:, 1]).argsort()]



# Reorder 
terms[[16,22]] = terms[[22,16]]
terms[[17,23]] = terms[[23,17]]
terms[[18,20]] = terms[[20,18]]
terms[[19,21]] = terms[[21,19]]

# XY
terms[5][:2], terms[6][:2] = -terms[5][:2], -terms[6][:2]
terms[5][2], terms[6][2] = '-'+terms[5][2], '-'+terms[6][2]


# Z
terms[22][:2], terms[23][:2] = -terms[22][:2], -terms[23][:2]
terms[22][2], terms[23][2] = '-'+terms[22][2], '-'+terms[23][2]

def get_clique(i, j):
    return np.sum(np.stack(terms[i:j][:, 0]), axis=0)

for i, term in enumerate(terms):
    print(i, term[1:])

groups = [(0,4), (4,8), (8,12), 
          (12,16), (16,20), (16,17),
         (17,18), (20,22), (22,24),
         (24,26)]
hs_s = np.array([terms[0][1], terms[4][1], terms[8][1],
         terms[12][1], terms[18][1], terms[16][1]-terms[18][1],
         terms[17][1]-terms[18][1], terms[20][1], terms[22][1],
         terms[24][1]])
Hs_s = np.array([get_clique(group[0], group[1]) for group in groups])
icosts = np.array([[2,3], [1,3], [2,3], 
                   [2,3], [1,1], [1,0], 
                   [1,0], [1,2], [1,0],
                   [1,0]])

# Get rid of first identity
Hs, hs = Hs[1:], hs[1:]

# Start simulation
t = 1
M = 15

rho = rand_rho(n)
Ns = [2**i + 10 for i in range(5, M)]
st = time.time()

# print(time.time()-st)
errors_costs = np.array([Error_cost(Hm, Hs_s, hs_s, t, rho, N, icosts, threads=12) for N in Ns])
errors_costs1 = np.array([Error_cost(Hm, Hs, hs, t, rho, N, threads=12) for N in Ns])
errors, errors1 = errors_costs[:, 0], errors_costs1[:, 0]
tcosts, rcosts = errors_costs[:, 1], errors_costs[:, 2]


np.savetxt('save/rcosts.txt', rcosts)
np.savetxt('save/tcosts.txt', tcosts)
np.savetxt('save/errors.txt', errors)
np.savetxt('save/errors1.txt', errors1)
np.savetxt('save/Ns.txt', Ns)
