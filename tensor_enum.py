#!/usr/bin/env python3

import numpy as np
import itertools
from sympy import *

def genbstring(n):
    return np.array([list(i) for i in itertools.product([0,1],repeat=n)])


def stabilizerWt(stab):
    n = stab.shape[0]//2
    wt = 0
    for i in range(n):
        if (stab[i]+stab[i+n])>0:
            wt += 1
    return wt


def XZWt(stab):
    n = stab.shape[0]//2
    Xwt = 0
    Zwt = 0
    for i in range(n):
        if (stab[i]>0):
            Xwt += 1
        if (stab[i+n]>0):
            Zwt += 1
    return Xwt, Zwt


def scalarEnum(H):
    rows, cols = H.shape
    n = cols//2
    enum = np.zeros(n+1).astype(int)
    bitstr = genbstring(rows)
    for i in range(2**rows):
        v = bitstr[i]
        stabilizer = (np.einsum("i,ij->j",v,H))%2
        wt = stabilizerWt(stabilizer)
        enum[wt] = enum[wt] + 1
    return enum


def doubleEnum(H):
    rows, cols = H.shape
    n = cols//2
    # enum = csr_matrix((n+1, n+1), dtype = np.int)
    enum = np.zeros((n+1,n+1)).astype(int)
    bitstr = genbstring(rows)
    for i in range(2**rows):
        v = bitstr[i]
        stabilizer = (np.einsum("i,ij->j",v,H))%2
        Xwt, Zwt = XZWt(stabilizer)
        enum[Xwt,Zwt] += 1
    return enum


def macWilliams(A,r_size):
    z, w = symbols('z w')
    Bz = 0
    for d in range(1,r_size+1):
        Bz += A[d-1]*((w-z)/2)**(d-1)*((w+3*z)/2)**(r_size-d)
    res = np.zeros(r_size)
    simBz = expand(simplify(Bz))
    print(simBz)
    for i in range(r_size):
        # stupid sympy cannot handle symbol**0 cleverly
        if i == 0:
            res[i] = float(simBz.coeff(w**(r_size-i-1)))
        elif i == r_size -1:
            res[i] = float(simBz.coeff(z**i))
        else:
            res[i] = float(simBz.coeff(z**(i)).coeff(w**(r_size-i-1)))
    return res/res[0]


def macWilliamsDouble(A,r_size):

    x, y, z, w = symbols('x y z w',commutative=True)
    n = r_size - 1
    Bz = 0
    K = 2**(n-np.log(A.sum())/np.log(2))
    for i in range(r_size):
        for j in range(r_size):
            Bz += A[i,j]*(z+w)**(n-i)*(z-w)**i*((x+y)/2)**(n-j)*((x-y)/2)**j
    simBz = expand(simplify(Bz))
    simBz = simBz.subs([(x,1),(z,1)])
    Bp = np.zeros((r_size, r_size))
    for i in range(r_size):
        for j in range(r_size):
            if i != (r_size-1):
                if j!= (r_size-1):
                    Bp[i,j]=float(simBz.coeff(y**(r_size-1-i)).coeff(w**(r_size-1-j)))
                else:
                    Bp[i,j]=float(simBz.coeff(y**(r_size-1-i)).subs(w,0)   )
            else:
                if j!= (r_size-1):
                    Bp[i,j]=float(simBz.coeff(w**(r_size-1-j)).subs(y,0))
                else:
                    Bp[i,j]=simBz.subs([(w,0),(y,0)])
    Bp = Bp * K
    Bp = np.rot90(np.fliplr(Bp), k=-1)
    return Bp.T
