import numpy as np
import scipy.linalg
import scipy.stats
from functools import reduce
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
from tqdm import trange, tqdm

def QDrift(hs, t, eps, N=None):
    """
    Input: A list of Hamiltonian terms Hs, time t, precision eps
    Output: Ordered list of indicies j, j corresponding to exp(i*lamb*t*Hj/N), lamb, N
    """
    lamb = np.sum(np.abs(hs))
    prob = np.abs(hs)/lamb
    if N is None:
        N = int(np.ceil(2*lamb**2*t**2/eps))
    Vlist = np.random.choice(hs.size, N, p=prob)
    return Vlist, lamb, N

I = np.array([[1,0],[0,1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
Paulis = [I,X,Y,Z]
Signs = [1,-1,1,-1]

def random_H(n, L=6):
    Hjs = []
    hjs = []
    for i in range(L):
        pind = np.random.randint(0,4,size=n)
        sind = np.random.randint(0,4,size=n)
        Hj = Paulis[pind[0]]*Signs[sind[0]]
        for j in range(1,n):
            Hj = np.kron(Hj, Paulis[pind[j]]*Signs[sind[j]])
        hj = np.random.uniform(0,1)
        Hjs.append(Hj)
        hjs.append(hj)
    Hjs, hjs = np.array(Hjs), np.array(hjs)
    # hjs = hjs/np.sum(hjs)
    H = np.sum(Hjs*hjs.reshape(-1,1,1),axis=0)
    return H, Hjs, hjs

def rand_rho(n):
    n = 2**n
    U = scipy.stats.unitary_group.rvs(n)
    rho = np.zeros((n,n))
    rho[0,0] = 1
    rho = np.matmul(np.matmul(U,rho), U.conj().T)
    return rho


def Error_cost(H, Hs, hs, t, rho, N, icosts=None, M=100, threads=1):
    st = time.time()
    n = H.shape[0] # actually 2^n
    L = len(Hs)
    VC = np.zeros((n,n))*1j
    norm = 0
    lamb = np.sum(np.abs(hs))
    prob = np.abs(hs)/lamb
    tau = t*lamb/N
    rcost, tcost = 0, 0
    if icosts is None:
        icosts = np.array(L*[[1,0]])
    
    for i in tqdm(range(M), desc='Running N={0}'.format(N)):
        idx = np.random.choice(hs.size, N, p=prob)
        His, pis = Hs[idx], prob[idx]
        sgns = np.sign(hs[idx])
        if threads > 1:
            Vis = np.array([scipy.linalg.expm(1j*tau*His[j]*sgns[j]) for j in range(N)] +
                           (threads-N%threads)*[np.identity(n)])
            Visp = np.split(Vis, threads)
            with Pool(threads) as p:
                Vi_pooled = np.array(p.map(np.linalg.multi_dot, Visp))
            Vi = np.linalg.multi_dot(Vi_pooled)
        elif threads == 1:
            Vis = np.array([scipy.linalg.expm(1j*tau*His[j]*sgns[j]) for j in range(N)])
            Vi = np.linalg.multi_dot(Vis)
        else:
            raise ValueError
        Vi = np.array(Vi).reshape((n,n))
        pi = np.prod(pis)
        tcost += np.sum([icosts[j][0] + icosts[j][1] for j in idx])
        rcost += np.sum([icosts[j][0] for j in idx])
        VC += np.linalg.inv(Vi) @ rho @ Vi
        norm += 1
    
    VC = VC/norm
    U = scipy.linalg.expm(1j*t*H)
    UC = np.linalg.inv(U) @ rho @ U
    Eps = np.linalg.norm(UC - VC)
    tcost, rcost = tcost/M, rcost/M
    return np.array([Eps, tcost, rcost])

def Error_line(H, Hs, hs, ts, eps, R=10, N=None):
    A, _, _ = random_H(n)
    psi = rand_rho(n)[0].reshape((-1,1))
    psi = psi/np.linalg.norm(psi)
    res = []
    
    for t in ts:
        errors = []
        for i in range(R):
            Vlist, lamb, N = QDrift(hs, t, eps, N)
            tau = t*lamb/N
            U, nU = scipy.linalg.expm(1j*t/N*H), scipy.linalg.expm(-1j*t/N*H)
            Vs, nVs = [scipy.linalg.expm(1j*tau*Hs[i]) for i in Vlist], [scipy.linalg.expm(-1j*tau*Hs[i]) for i in Vlist[::-1]]
            V, nV = reduce(np.matmul, Vs), reduce(np.matmul, nVs)
            pred = np.sum(np.array([psi.T.conj() @ v @ A @ nv @ psi for v, nv in zip(Vs, nVs)]), axis=0)/N
            actual = psi.T.conj() @ U @ A @ nU @ psi 
            errors.append(pred-actual)
        res.append(np.abs(np.mean(errors).flatten()[0]))
    return res
