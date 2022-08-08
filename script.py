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

def Simulation_cost(Chs, CPs, custom=False):
    """ Returns the cost of the simulation,
    total costs, individual costs in [crzs, toffolis]
    """
    crzs, Toffolis = 0, 0
    icosts = []
    n = len(CPs[0][0])
    for Ch, CP in zip(Chs, CPs):
        _, CZ, _ = diag_results(CP, True)
        _, _, cost = logic_min(CZ, Ch, custom)
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


n = 4
Jx, Jy, Jz, h = np.random.normal(loc=0, scale=1, size=4)
print('Coefficients: ', Jx, Jy, Jz, h)
# G = [[0,1], [0,5], [1,2], [1,4], [2,3], [3,4], [3,8], [4,5], [4,7], [5,6], [6,7], [7,8]]
# print('Graph: ', G)
Hm, CHs, Chs, CPs = Heisenberg_1d(n, Jx, Jy, Jz, h, True)
Hm, Hs, hs, Ps = Heisenberg_1d(n, Jx, Jy, Jz, h, False)
tcost, icosts = Simulation_cost(Chs, CPs, True)
SHs, Shs = Hs_sum(CHs, Chs)
Hs_s, hs_s, icosts = Hs_sum_costs(SHs, Shs, icosts)


# Start simulation
t = 1
M = 15

rho = rand_rho(n)
Ns = [2**i + 10 for i in range(6, M)]
st = time.time()

# print(time.time()-st)
errors_costs = []
for N in Ns:
    print(N)
    errors_costs.append(Error_cost(Hm, Hs_s, hs_s, t, rho, N, icosts, M=200, threads=24))
errors_costs = np.array(errors_costs)

errors_costs1 = []
for N in Ns:
    print(N)
    errors_costs1.append(Error_cost(Hm, Hs, hs, t, rho, N, M=200, threads=24))
errors_costs1 = np.array(errors_costs1)

errors, errors1 = errors_costs[:, 0], errors_costs1[:, 0]
tcosts, rcosts = errors_costs[:, 1], errors_costs[:, 2]

np.savetxt('save/rcosts-4H1d.txt', rcosts)
np.savetxt('save/tcosts-4H1d.txt', tcosts)
np.savetxt('save/errors-4H1d.txt', errors)
np.savetxt('save/errors1-4H1d.txt', errors1)
np.savetxt('save/Ns-4H1d.txt', Ns)

