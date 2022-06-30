import numpy as np
from functools import reduce
import time
import commute
import diagonalize
import phase
import QDrift
import hamiltonian
import simulation

n = 6
Jx, Jy, Jz, h = np.random.normal(loc=0, scale=1, size=4)
print('Coefficients: ', Jx, Jy, Jz, h)
# G = [[0,1], [0,5], [1,2], [1,4], [2,3], [3,4], [3,8], [4,5], [4,7], [5,6], [6,7], [7,8]]
# print('Graph: ', G)
Hm, CHs, Chs, CPs = Heisenberg_1d(n, Jx, Jy, Jz, h, True)
Hm, Hs, hs, Ps = Heisenberg_1d(n, Jx, Jy, Jz, h, False)
tcost, icosts = Simulation_cost(Chs, CPs)
SHs, Shs = Hs_sum(CHs, Chs)
Hs_s, hs_s, icosts = Hs_sum_costs(SHs, Shs, icosts)

t = 2
M = 15

rho = rand_rho(n)
Ns = [2**i + 10 for i in range(5, M)]
st = time.time()
errors_costs1 = np.array([Error_cost(Hm, Hs, hs, t, rho, N, threads=12) for N in Ns])
errors_costs = np.array([Error_cost(Hm, Hs_s, hs_s, t, rho, N, icosts, threads=12) for N in Ns])
errors, errors1 = errors_costs[:, 0], errors_costs1[:, 0]
tcosts, rcosts = errors_costs[:, 1], errors_costs[:, 2]

print(time.time()-st)

np.savetxt('save/rcosts.txt', rcosts)
np.savetxt('save/tcosts.txt', tcosts)
np.savetxt('save/errors.txt', errors)
np.savetxt('save/errors1.txt', errors1)
np.savetxt('save/Ns.txt', Ns)
