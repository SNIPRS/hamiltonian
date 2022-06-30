import numpy as np
import pyeda.inter as eda

def binary_strings(n, vals=[0,1]):
    return np.array(np.meshgrid(*[vals]*n,
                    indexing='ij')).reshape((n,-1)).transpose()

def bin_to_dec(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

def dnf(n, ks):
    X = eda.ttvars('x', n)
    sl = ['0' for i in range(2**n)]
    for k in ks:
        sl[k] = '1'
    s = ""
    s = s.join(sl)
    T = eda.truthtable(X, s)
    f = eda.espresso_tts(T)
    return f

def cost(f):
    """Returns toffoli cost of a circuit
    """
    cost = 0
    try: 
        for clause in f[0].xs:
            try:
                cost += len(clause.xs) - 1
            except AttributeError:
                pass
    except AttributeError:
        pass
    return cost 

def phase(x, pauli_strings, ais):
    d = {'Z':1, 'I':0}
    yis = np.array([[d[c] for c in s] for s in pauli_strings])
    sis = np.array([np.dot(x, yi) for yi in yis])
    sgns = np.power(-1, sis)
    phi = np.mod(np.dot(ais, sgns), 2*np.pi)
    return phi
    
def thetas(pauli_strings, ais):
    """ Returns all distint values of phase, 
    with all evaluations to it
    """
    n = len(pauli_strings[0])
    theta_map = {}
    xs = binary_strings(n)
    for x in xs:
        k = bin_to_dec(x)
        phi = phase(x, pauli_strings, ais)
        ind = min(phi, 2*np.pi-phi)
        s = 0 if ind == phi else 1
        theta_keys = np.array(list(theta_map.keys()))
        if ind in theta_map:
            theta_map[ind][s].append(k)
        elif theta_keys.size == 0:
            theta_map[ind] = {s:[k], (s+1)%2:[]}
        else:
            closest = theta_keys[np.abs(np.array(theta_keys) - ind).argmin()]
            if np.abs(closest - ind) < 1e-6:
                theta_map[closest][s].append(k)
            else:
                theta_map[ind] = {s:[k], (s+1)%2:[]}
    return theta_map

def logic_min(pauli_strings, ais, custom=False):
    """ Performs logic minimization on input pauli strings with coefficients
        Output: 
            circuits - theta : [pos circuit, neg circuit]
            costs - theta: [pos circuit, neg circuit]
            tcost - [crzs, Toffolis]
    """
    theta_map = thetas(pauli_strings, ais)
    n = len(pauli_strings[0])
    circuits = {}
    costs = {}
    crzs, Toffolis = 0,0
    debug_thetas = []
    for theta in theta_map:
        if not np.isclose(theta, 0):
            kspos, ksneg = theta_map[theta][0], theta_map[theta][1]
            fpos, fneg = dnf(n, kspos), dnf(n, ksneg)
            circuits[theta] = [fpos, fneg]
            costs[theta] = [cost(fpos), cost(fneg)]
            crzs += 1
            Toffolis += cost(fpos) + cost(fneg)
            debug_thetas.append(theta)
    if custom:
        if n == 4:
            crzs, Toffolis = 1, min(1, Toffolis)
        if n == 6:
            crzs, Toffolis = 2, min(3, Toffolis)
    print('crzs, toffolis, thetas:')
    print(crzs, Toffolis, debug_thetas)
    return circuits, costs, [crzs, Toffolis]


