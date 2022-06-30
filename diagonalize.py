import numpy as np

def Swap(A,i,j,r):
    if i !=j:
        if r: # swap rows
            A[[i,j],:] = A[[j,i],:]
        else: # swap columns
            A[:,[i,j]] = A[:,[j,i]]

def Sweep(A,i,j,r):
    if r: # add i to j
        A[j,:] = (A[j,:] + A[i,:]) % 2
    else:
        A[:,j] = (A[:,j] + A[:,i]) % 2

def FSwap(X,Z,S,i,j,r):
    Swap(X,i,j,r); Swap(Z,i,j,r)
    if r:
        Swap(S,i,j,r)

def FSweep(X,Z,S,i,j,r):
    Sweep(X,i,j,r); Sweep(Z,i,j,r); Sweep(S,i,j,r)
    
def H(X,Z,i):
    temp = np.copy(X[:,i])
    X[:,i] = Z[:,i]
    Z[:,i] = temp
    
def SP(X,Z,S,i):
    S = (S + X[:,i] * Z[:,i]) % 2
    Z[:,i] = (Z[:,i] + X[:,i]) % 2

def CX(X,Z,S,i,j):
    Xi, Xj, Zi, Zj = X[:,i], X[:,j], Z[:,i], Z[:,j] 
    S = (S + (Xi*Zj*(Xj+Zi+1))) % 2
    Z[:,i] = (Zi+Zj) % 2
    X[:,j] = (Xi+Xj) % 2
    
def CZ(X,Z,S,i,j):
    Xi, Xj, Zi, Zj = X[:,i], X[:,j], Z[:,i], Z[:,j]
    S = (S + (Xi*Xj*(Zj+Zi+1))) % 2
    Z[:,i] = (Zi+Xj) % 2
    Z[:,j] = (Xi+Zj) % 2

def Alg1(T):
    """ Diagonalizes the X block
        Input: Tableau T = [X,Z,S] of size m*(2n+1)
        Output: Updated T with off diagonal X entries 0, Rank k
    """
    T = np.copy(T)
    assert T.shape[1] % 2 == 1
    m = T.shape[0]
    n = T.shape[1]//2
    X = T[:,0:n]
    Z = T[:,n:2*n]
    S = T[:,2*n:2*n+1]
    
    T0 = np.copy(T)
    X0 = T0[:,0:n]
    Z0 = T0[:,n:2*n]
    S0 = T0[:,2*n:2*n+1]
    
    k=0
    gates = []
    swaps = []
    
    while k<m and k<n:
        X1 = X[k:, k:]
        if np.any(X1==1):
            i,j = np.nonzero(X1==1)[0][0] + k, np.nonzero(X1==1)[1][0] + k
            FSwap(X,Z,S,i,k,True)
            FSwap(X,Z,S,j,k,False)
            FSwap(X0,Z0,S0,j,k,False)
            swaps.append(('swapt',i,k))
            swaps.append(('swapf',j,k))
            for l in range(m):
                if l != k and X[l,k] == 1:
                    FSweep(X,Z,S,k,l,True)
                    swaps.append(('sweep',k,l))
            k+=1
        else:
            break
    
    kx = k
    while k<m and k<n:
        Z1 = Z[k:, k:]
        if np.any(Z1==1):
            i,j = np.nonzero(Z1==1)[0][0] + k, np.nonzero(Z1==1)[1][0] + k
            FSwap(X,Z,S,i,k,True)
            FSwap(X,Z,S,j,k,False)
            FSwap(X0,Z0,S0,j,k,False)
            swaps.append(('swapt',i,k))
            swaps.append(('swapf',j,k))
            for l in range(m):
                if l != k and Z[l,k] == 1:
                    FSweep(X,Z,S,k,l,True)
                    swaps.append(('sweep',k,l))
            k+=1
        else:
            break
    
    for j in range(kx,k):
        H(X,Z,j)
        H(X0,Z0,j)
        gates.append(('H',j,j))
    
    for i in range(k):
        for j in range(k,n):
            if X[i,j]==1:
                CX(X,Z,S,i,j)
                CX(X0,Z0,S0,i,j)
                gates.append(('CX',i,j))
    
    assert np.count_nonzero(X[:min(n,m),:min(n,m)] - np.diag(np.diagonal(X))) == 0
    assert np.allclose(Z[:k,:k], Z[:k,:k].T)
    T = np.concatenate((X,Z,S), axis=1)
    return T,T0, gates, swaps

def Alg2(T, T0=None):
    """ Pairwise update of Z, clear X
        Input: Tableau T with diagonal X with rank k
        Output: Updated T with X block entries set to zero
    """
    
    T = np.copy(T)
    assert T.shape[1] % 2 == 1
    m = T.shape[0]
    n = T.shape[1]//2
    X = T[:,0:n]
    Z = T[:,n:2*n]
    S = T[:,2*n:2*n+1]
    
    if T0 is not None:
        X0 = T0[:,0:n]
        Z0 = T0[:,n:2*n]
        S0 = T0[:,2*n:2*n+1]
    
    k = np.linalg.matrix_rank(X)
    gates = []
    
    for i in range(1, k):
        for j in range(i):
            if Z[i,j] == 1:
                CZ(X,Z,S,i,j)
                gates.append(('CZ',i,j))
                if T0 is not None:
                    CZ(X0,Z0,S0,i,j)
    for i in range(k):
        if Z[i,i] == 1:
            SP(X,Z,S,i)
            gates.append(('SP',i,i))
            if T0 is not None:
                SP(X0,Z0,S0,i)
                
        H(X,Z,i)
        gates.append(('H',i,i))
        if T0 is not None:
            H(X0,Z0,i)
    
    assert np.all(X==0)
    T = np.concatenate((X,Z,S), axis=1)
    return T, T0, gates

def Alg3(T): # NON UPDATED
    """ Pairwise update of Z, clear X
        Input: Tableau T with diagonal X with rank k
        Output: Updated T with X block entries set to zero
    """
    T = np.copy(T)
    assert T.shape[1] % 2 == 1
    m = T.shape[0]
    n = T.shape[1]//2
    X = T[:,0:n]
    Z = T[:,n:2*n]
    S = T[:,2*n:2*n+1]
    
    k = np.linalg.matrix_rank(X)
    gates = []
    
    for i in range(k):
        if np.sum(Z[i,0:i+1]) % 2 == 0:
            SP(X,Z,S,i)
            gates.append(('S',i,i))
        for j in range(i):
            if Z[i,j] == 1:
                CX(X,Z,S,i,j)
                FSweep(X,Z,S,j,i, True)
    
    for i in range(k):
        SP(X,Z,S,i)
        H(X,Z,i)
    
    assert np.all(X==0)
    T = np.concatenate((X,Z,S), axis=1)
    return T

def Alg4(n):
    """ Random generator sets for commuting Paulis
        Input: Pauli size n
        Output: Tableau with n generators for random maximally commuting n-Pauli set. 
    """
    X, Z = np.zeros((n,n),dtype=int), np.zeros((n,n),dtype=int)
    for i in range(n):
        r = np.random.randint(0, 2**(n+1-i)+1)
        X[i,i] = 1
        if r == 2**(n+1-i):
            H(X,Z,i)
        else:
            for j in range(i,n):
                Z[i,j], Z[j,i] = r % 2, r % 2
                r = r//2
    S = np.random.randint(0,2,(n,1))
    T = np.concatenate((X,Z,S), axis=1)
    return T

def tableaus(L):
    """ Returns tableau corresponding to list of Pauli strings
    """
    rows = []
    ref = {'I': (0,0), 'Z': (0,1), 'X': (1,0), 'Y': (1,1)}
    for i in range(len(L)):
        string = L[i]
        if '-' in string:
            string = string[1:]
            s = [1]
        else:
            s = [0]
        x, z = [], []
        for c in string:
            x.append(ref[c][0]); z.append(ref[c][1])
        rows.append(x+z+s)
    return np.array(rows)
        
def paulis(T):
    """ Returns list of Pauli strings corresponding to tableau
    """
    T = np.copy(T)
    m = T.shape[0]
    n = T.shape[1]//2
    X = T[:,0:n]
    Z = T[:,n:2*n]
    S = T[:,2*n:2*n+1]
    
    ref = [['I', 'Z'],['X', 'Y']]
    paulis = []
    for i in range(m):
        string = '-' if S[i] == 1 else ''
        for j in range(n):
            string = string + ref[int(X[i,j])][int(Z[i,j])]
        paulis.append(string)
    return paulis
    
def diag_results(T, ret=False):
    """ Display results, or returns Tableau, Z strings, diagonalizing gates
    """
    if type(T[0]) is type(np.str_('a')) or type(T[0]) is type('a'):
        T = tableaus(T)
    T = np.copy(T)
    m = T.shape[0]
    n = T.shape[1]//2
    X = T[:,0:n]
    Z = T[:,n:2*n]
    S = T[:,2*n:2*n+1]
    
    T0, T00, gates, swaps = Alg1(T)
    T1, T11, gates1 = Alg2(T0, T00)
    
    X = T11[:,0:n]
    Z = T11[:,n:2*n]
    S = T11[:,2*n:2*n+1]
    
    Tgates = gates + gates1
    Fgates = []
    permutation = []
    
    for i in range(n):
        curr = i 
        for swap,a,b in swaps[::-1]:
            if swap == 'swapf':
                if a == curr:
                    curr = b 
                if b == curr:
                    curr = a
        permutation.append(curr)
        
    for swap, a, b in swaps[::-1]:
        if swap == 'swapf':
            FSwap(X,Z,S,a,b,False)
        # elif swap == 'swapt':
            # FSwap(X,Z,S,a,b,True)
        # else:
            # FSweep(X,Z,S,a,b,True)
    
    pauli_strings = paulis(T11)
    for tup in Tgates:
        Fgates.append((tup[0], permutation[tup[1]], 
                       permutation[tup[2]]))
    
    # print('Original: \n', T, '\n')
    # print('Raw Output: \n', T1, '\n')
    # print('Final Tableau: \n', T11, '\n')
    if not ret:
        print('Pauli strings: ')
        print(pauli_strings)
        print('\n')
        print('Gates: ')
        print(Fgates)
    if ret:
        return T11, pauli_strings, Fgates
    
