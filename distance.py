#!/usr/bin/env python3
from itertools import product, chain, permutations, combinations
from functools import reduce


pauli_products = {
    "II": "I",
    "IX": "X",
    "IY": "Y",
    "IZ": "Z",
    "XI": "X",
    "XX": "I",
    "XY": "Z",
    "XZ": "Y",
    "YI": "Y",
    "YX": "Z",
    "YY": "I",
    "YZ": "X",
    "ZI": "Z",
    "ZX": "Y",
    "ZY": "X",
    "ZZ": "I"
}
def multiply_paulis_without_phase(pstr1, pstr2):
    output = ""
    for p1, p2 in zip(pstr1, pstr2):
        output += pauli_products[p1 + p2]

    return output

def powerset(gens):
    s = list(gens)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generated_group(gens, rep=None, incl_identity=True):
    if rep is None:
        rep = len(gens[0]) * "I"
    for g in powerset(gens):
        if len(g) > 0:
            yield reduce(lambda P, Q: multiply_paulis_without_phase(P, Q), g, rep)
        elif incl_identity:
            yield rep

def pauli_weight(pstr):
    return len([op for op in pstr if op != 'I'])

def distance(gens, xbars, zbars, n_qubits):
    min_weight = n_qubits
    for Pbar in generated_group(xbars + zbars, incl_identity=False):
        p_min_weight = min([pauli_weight(op) for op in generated_group(gens, Pbar)])
        if p_min_weight < min_weight:
            min_weight = p_min_weight
    return min_weight
    # x_min = min([pauli_weight(op) for op in generated_group(gens, logical_x)])
    # z_min = min([pauli_weight(op) for op in generated_group(gens, logical_z)])
    # logical_y = multiply_paulis_without_phase(logical_x, logical_z)
    # y_min = min([pauli_weight(op) for op in generated_group(gens, logical_y)])
    # return min(x_min, y_min, z_min)


# gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
# # for i in coset(gens):
# #     print((i))

# logical_x = ["XXXXX"]
# logical_z = ["ZZZZZ"]
# distance(gens, logical_x, logical_z)
# # logical_y = multiply_paulis_without_phase(logical_x, logical_z)
# # for i in coset(gens, logical_y):
# #     print((i))
