#!/usr/bin/env python3


import distance
from gym import Env, spaces
from itertools import product, combinations, chain
from functools import reduce
from math import floor, sqrt
from collections import defaultdict
import copy
import numpy as np
import scipy.optimize
import sympy
from tensor_enum import *

STABILIZERS = ["IIXXXX", "IIZZZZ", "ZIZZII", "IZZIZI", "IXXXII", "XIXIXI"]

class Legoenv(Env):
    def __init__(self, max_tensors=7):

        self.bad_action_reward = -10
        self.bad_code_reward = -.5
        self.base_distance = 2
        self.tensor_size = len(STABILIZERS)
        self.num_tensor_types = 1
        self.debug_mode = False

        # include the max number of legs willing to accommodate
        self.max_tensors = max_tensors
        self.max_legs = self.max_tensors * self.tensor_size
        self.min_legs = 3

        # Reward shaping
        self.r_add_tensor = .1
        self.r_single_trace = .05
        self.r_self_trace = -.1
        self.r_terminate = .15

        # set up states and actions
        self.num_leg_combinations = self.max_legs * (self.max_legs -1)//2
        # observation space is specified by types of tensors specified as well as the (n choose 2) options for connectivity
        self.observation_space = spaces.MultiDiscrete([self.num_tensor_types + 1] * self.max_tensors + [2] * self.num_leg_combinations + [2])
        # action space corresponds to changing observation with the option to terminate at the end
        # since only one action can be chosen, we only need a single action rather than an array of them
        self.action_space = spaces.Discrete((self.num_tensor_types) * self.max_tensors + self.num_leg_combinations + 1)


        # keeping track of check matrices
        self.cmat = None
        self.num_legs = 0
        self.tensor_info = dict()  # tensor index : [included in self.cmat, legs]
        self.legs_to_tensor = defaultdict(lambda: -1)  # inverse map of above # default dict if no tensor there
        self.legs_to_cmat_indices = dict()  # convert physical legs to check matrix indices
        self.tensor_to_components = {i: i for i in range(self.max_tensors)}


        self.available_legs = [] # labels of qubit numbers
        self.contracted_legs = [] # list of tuples

        self.state = np.zeros(self.observation_space.shape)
        self.terminal_state = self.state.copy()
        self.terminal_state[-1] = 1
        self.actions = []


    def get_connected_components(self):
        """Returns a dict from tensor idx to connected component"""
        tensor_to_components = {i: i for i in range(self.max_tensors)}
        for pair in self.contracted_legs:
            leg1, leg2 = pair
            t1 = self.legs_to_tensor[leg1]
            t2 = self.legs_to_tensor[leg2]
            c1 = tensor_to_components[t1]
            c2 = tensor_to_components[t2]
            if c1 != c2:
                if c1 < c2:
                    tensor_to_components[t2] = c1
                else:
                    tensor_to_components[t1] = c2
        return tensor_to_components

    def find_disconnected_component_legs(self):
        """Checks for qubit indices that are not contracted with the first connected component

        Used right before calculating code distances to avoid trivially
        finding distances of the base tensor."""
        tensor_to_components = self.get_connected_components()
        # Look at the connected component with first available leg
        # This is relevant e.g. when the first tensor is completely contracted
        idxs_to_exclude = []
        if len(self.available_legs):
            main_component = tensor_to_components[self.legs_to_tensor[self.available_legs[0]]]
            for qubit in self.available_legs[1:]:
                if tensor_to_components[self.legs_to_tensor[qubit]] != main_component:
                    idxs_to_exclude.append(self.legs_to_cmat_indices[qubit])

        return idxs_to_exclude


    def get_leg_indices_from_contraction_index(self, linear_idx, include_collisions=True):
        """Given idx which represents an (i,j) tuple (of which there are (n choose 2)),
        return (i,j).
        If include_collisions is True also include the indices for pairs (i, .) and (., j)"""
        # first get col, row indices
        n = self.max_legs

        row_idx = np.triu_indices(n, k=1)[0][linear_idx]  # leg 1
        col_idx = np.triu_indices(n, k=1)[1][linear_idx]  # leg 2

        if not include_collisions:
            return row_idx, col_idx

        shared_row_or_col_idxs = []

        # make sure nothing else connects to leg 2
        for col in range(row_idx, n):
            idx = (n * (n-1))//2 - (n - row_idx) * ((n - row_idx) - 1)//2 + col - row_idx - 1
            shared_row_or_col_idxs.append(idx)

        # make sure leg 1 doesn't connect with anything else
        for row in range(col_idx):
            idx = (n * (n-1))//2 - (n - row) * ((n - row) - 1)//2 + col_idx - row - 1
            shared_row_or_col_idxs.append(idx)

        # make sure nothing else connects to leg 1
        for row in range(row_idx):
            idx = (n * (n-1))//2 - (n - row) * ((n - row) - 1)//2 + row_idx - row - 1
            shared_row_or_col_idxs.append(idx)

        # make sure leg 2 doesn't connect with anything else
        for col in range(col_idx, n):
            idx = (n * (n-1))//2 - (n - col_idx) * ((n - col_idx) - 1)//2 + col- col_idx - 1
            shared_row_or_col_idxs.append(idx)

        return (row_idx, col_idx), shared_row_or_col_idxs


    def step(self, action):
        info = {'debug': []}
        self.actions.append(action)  # for debugging purposes
        done = False
        reward = 0.01 # small reward for taking valid actions

        num_tensor_actions = self.num_tensor_types * self.max_tensors
        if action < num_tensor_actions:
            # double check that tensor is not already selected
            # double check that previous tensor spots are already filled
            tensor_idx, tensor_type = action // self.num_tensor_types, action % self.num_tensor_types
            if self.state[tensor_idx] != 0 or (tensor_idx != 0 and self.state[tensor_idx -1] == 0):
                reward = self.bad_action_reward
                self.state = self.terminal_state
                done = True
                info['debug'].append("Invalid tensor choice")
            else:
                self.state[tensor_idx] = 1 + tensor_type
                tensor_legs = (self.num_legs, self.num_legs + self.tensor_size)
                self.num_legs = self.num_legs + self.tensor_size
                # insert tensor type as unconnected
                # add new legs based on tensor type
                # connected = True if tensor_idx == 0 else False
                self.tensor_info[tensor_idx] = [True, tuple(range(*tensor_legs))]
                for l in range(*tensor_legs):
                    self.legs_to_tensor[l] = tensor_idx
                self.available_legs.extend(list(range(*tensor_legs)))

                info['debug'].append("Added new tensor of kind: " + str(tensor_type))
                reward = self.r_add_tensor
                if tensor_idx == 0:
                    self.tensor_info[tensor_idx] = (True, tuple(range(*tensor_legs)))
                    self.cmat = T6_Stabilizer(0).check_matrix
                    self.legs_to_cmat_indices = {l: l for l in range(*tensor_legs)}
                else:
                    cmat_size = self.cmat.n_qubits
                    self.cmat.zero_trace(T6_Stabilizer(0).check_matrix)
                    for i, l in enumerate(range(*tensor_legs)):
                        self.legs_to_cmat_indices[l] = cmat_size + i

        elif action < num_tensor_actions + self.num_leg_combinations:
            # verify that legs aren't already contracted
            (leg1, leg2), possible_conflicted_contractions = self.get_leg_indices_from_contraction_index(action - num_tensor_actions)
            self.contracted_legs.append((leg1, leg2))
            info['debug'].append("Contracting legs: " + str(leg1) + " " + str(leg2))


            for contraction_idx in possible_conflicted_contractions:
                if self.state[contraction_idx + self.max_tensors] != 0:
                    reward = self.bad_action_reward
                    self.state = self.terminal_state
                    done = True

            if not done:
                tensor_1, tensor_2 = self.legs_to_tensor.get(leg1), self.legs_to_tensor.get(leg2)
                # TODO: this statement is obsolete with zero_trace happening
                if tensor_1 is None or tensor_2 is None or tensor_1 < 0 or tensor_2 < 0:
                    reward = self.bad_action_reward
                    self.state = self.terminal_state
                    done = True
                else:
                    # TODO: fix up available_legs, actually might not be necessary
                    self.state[action] = 1  # state update is binary for whether those are connected
                    # check whether either one or both are connected
                    # TODO: remove these since this is trivially true
                    leg1_connected, leg2_connected = self.tensor_info[tensor_1][0], self.tensor_info[tensor_2][0]
                    if leg1_connected and leg2_connected:  # self trace
                        # Check whether this is a self trace on connected components
                        if self.tensor_to_components[tensor_1] == self.tensor_to_components[tensor_2]:
                            reward = self.r_self_trace
                        else:
                            # Update the connected component
                            reward = self.r_single_trace
                            self.tensor_to_components[tensor_2] = self.tensor_to_components[tensor_1]


                        idx1, idx2 = self.legs_to_cmat_indices[leg1], self.legs_to_cmat_indices[leg2]
                        assert idx1 < idx2
                        self.cmat = self.cmat.self_trace(idx1, idx2)
                        del self.legs_to_cmat_indices[leg1]
                        del self.legs_to_cmat_indices[leg2]
                        for k, v in self.legs_to_cmat_indices.items():
                            # shift columns left if after idx1 or idx 2
                            if v > idx2:
                                self.legs_to_cmat_indices[k] = v-2
                            elif v > idx1:
                                self.legs_to_cmat_indices[k] = v-1
                        self.available_legs.remove(leg1)
                        self.available_legs.remove(leg2)
                        if self.cmat.n_qubits <= self.min_legs or self.cmat.mat is None:
                            done = True
                            print("HA you did something very bad in a different way")
                            reward = self.bad_code_reward
                            # self.state = self.terminal_state
                    else:
                        done=True
                        print("Oops!")
                        reward = -2


        else:
            done = True
            self.state[-1] = 1

            reward = self.calculate_reward()
            info['debug'].append("Choose to terminate")
            if self.debug_mode:
                print("Final Distance is: ", reward)

        return self.state, reward, done, info

    def render(self):
        # print debugging stuff here
        print(self.cmat.n_qubits)

    def calculate_reward(self):
        leg_idxs_to_drop = self.find_disconnected_component_legs()
        if len(leg_idxs_to_drop) > 0:
            old_cmat = copy.deepcopy(self.cmat)
            self.cmat.drop_columns(leg_idxs_to_drop)

        if self.cmat is not None:
            if self.available_legs[0] != 0:
                reward = -2 - self.base_distance
            else:
                reward = self.cmat.css_distance_first_qubit() - self.base_distance
                # add term to penalize too many legs
        else:
            reward = 0
        if len(leg_idxs_to_drop) > 0:
            # restore the full check matrix
            self.cmat = old_cmat
        return reward

    def reset(self, state=None):
        self.available_legs = [] # labels of qubit numbers
        self.contracted_legs = [] # list of tuples
        self.num_legs = 0
        self.tensor_info = dict()  # tensor index : [included in self.cmat, legs]
        self.legs_to_tensor = defaultdict(lambda: -1)  # inverse map of above # default dict if no tensor there
        self.legs_to_cmat_indices = dict()  # convert physical legs to check matrix indices
        self.tensor_to_components = {i: i for i in range(self.max_tensors)}

        if state is None:
            self.state = np.zeros(self.observation_space.shape)
        else:
            self.state = state
        self.cmat = None
        self.actions = []
        return self.state

    def state_to_tuple_key(self):
        terminated = (self.state[-1] == 1)
        return tuple([int(np.sum(self.state[:self.max_tensors]))] + self.contracted_legs + [terminated])

class Biased_Legoenv(Legoenv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_add_tensor = .1
        self.r_single_trace = .05
        self.r_self_trace = -.1
        self.r_terminate = .15

    def calculate_reward(self, px=.01, pz=.05, p_thresh=1.0747e-5):
        leg_idxs_to_drop = self.find_disconnected_component_legs()
        if len(leg_idxs_to_drop) > 0:
            old_cmat = copy.deepcopy(self.cmat)
            self.cmat.drop_columns(leg_idxs_to_drop)


        if self.cmat is not None and self.available_legs[0] == 0:
            mat_to_check = self.cmat.mat
            nrow, ncol = mat_to_check.shape
            check_correctable_dumb = mat_to_check[0,0] == 1
            if not check_correctable_dumb:
                reward = -10
            else:
                removed_mat = np.delete(np.delete(mat_to_check,[0,ncol//2],axis=1), [0,nrow//2], axis=0)
                A = doubleEnum(removed_mat)
                B = macWilliamsDouble(A, ncol//2)
                # distance = list(zip(*np.where((B-A) > 0)))[0][1] # first index of the form (0, .)
                n, m = B.shape
                a_err = 0
                b_err = 0
                for i in range(n):
                    for j in range(m):
                        prob_factor = (px**i)*((1-px)**((n-1)-i))*(pz**j)*((1-pz)**((m-1)-j))
                        a_err += A[i,j] * prob_factor
                        b_err += B[i,j] * prob_factor

                # p_err = b_err - a_err # unnormalized
                p_err = 1 - a_err/b_err #normalized
                if p_err <= 0:
                    print(self.state_to_tuple_key())
                    reward = -10
                else:
                    reward = np.log(p_thresh)-np.log(p_err)
                # (p_thresh - p_err)/p_thresh * 50 # 2% improvement gives a score of 1
        else:
             reward = -8

        if len(leg_idxs_to_drop) > 0:
            # restore the full check matrix
            self.cmat = old_cmat

        return reward



class T6_Stabilizer(object):
    def __init__(self, legs, check_matrix=None, stabilizers=STABILIZERS):
        self.stabilizers = stabilizers
        if check_matrix is None:
            self.check_matrix = Check_Matrix(self.stabilizers)
        else:
            self.check_matrix = check_matrix
        if isinstance(legs, int):
            self.available_legs = range(legs, legs + len(self.stabilizers[0]))
        else:
            self.available_legs = legs

    def combine(self, other, idx1, idx2):
        legs = [l for l in (self.available_legs + other.available_legs) if (l != idx1 and l != idx2)]
        check_matrix = self.check_matrix.single_trace(other.check_matrix, idx1, idx2)
        stabilizers = check_matrix.generate_stabilizers()
        return T6_Stabilizer(legs, check_matrix, stabilizers)



class Check_Matrix(object):
    def __init__(self, stabilizers):
        # TODO: dict from legs to indices
        self.n_qubits = len(stabilizers[0])
        self.n_rows = len(stabilizers)
        self.mat = np.zeros((self.n_rows, 2 * self.n_qubits))
        for i, p_str in enumerate(stabilizers):
            for j, p in enumerate(p_str):
                if p == "X" or p == "Y":
                    self.mat[i, j] = 1
                if p == "Z" or p == "Y":
                    self.mat[i, j+self.n_qubits] = 1

    def convert_array_to_stabilizer(self, arr):
        x_powers = arr[:self.n_qubits]
        z_powers = arr[self.n_qubits:]
        stab = ""
        for (xp, zp) in zip(x_powers, z_powers):
            if xp and zp:
                stab += "Y"
            elif xp:
                stab += "X"
            elif zp:
                stab += "Z"
            else:
                stab += "I"
        return stab

    def generate_stabilizers_from_mat(self):
        stabs = []
        for row in self.mat:
            stabs.append(self.convert_array_to_stabilizer(row))
        return stabs

    def powerset(self, gens):
        s = list(gens)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def matrix_weight(self, x_gens, z_gens, max_dist=0):
        """Compute CSS distance using generators.
        Terminate early if any of the generators have weight less than max_dist"""
        x_mat = np.vstack([[1 if p == "X" else 0 for p in stab] for stab in x_gens])
        z_mat = np.vstack([[1 if p == "Z" else 0 for p in stab] for stab in z_gens])
        if np.any(np.sum(x_mat, axis=1) < max_dist) or np.any(np.sum(z_mat, axis=1) < max_dist):
            return 0
        min_weight = len(x_gens[0])
        for row_selects in self.powerset(range(len(x_gens))):
            if len(row_selects) > 0:
                row_sum = np.sum(x_mat[tuple([row_selects])], axis=0) % 2 # TODO: future deprecation warning
                if not np.all(row_sum == 0):
                    row_weight = np.sum(row_sum)
                    if row_weight < min_weight:
                        min_weight = row_weight
        for row_selects in self.powerset(range(len(z_gens))):
            if len(row_selects) > 0:
                row_sum = np.sum(z_mat[tuple([row_selects])], axis=0) % 2
                if not np.all(row_sum == 0):
                    row_weight = np.sum(row_sum)
                    if row_weight < min_weight:
                        min_weight = row_weight
        return min_weight

    def css_distance_first_qubit(self):
        all_generators = self.generate_stabilizers_from_mat()
        x_generators = []
        z_generators = []
        for stab in all_generators:
            if "X" in stab:
                x_generators.append(stab)
            else:
                z_generators.append(stab)
        if len(x_generators) == 0 or len(z_generators) == 0:
            return 0
        Xs = []
        Zs = []
        Ixs = []
        Izs = []
        for gen in x_generators:
            op = gen[1:]
            if gen[0] == "X":
                Xs.append(op)
            else:
                Ixs.append(op)
        for gen in z_generators:
            op = gen[1:]
            if gen[0] == "Z":
                Zs.append(op)
            else:
                Izs.append(op)
        if len(Xs) == 0 or len(Zs) == 0:
            return 0
        x_weights = np.sum([[1 if p == "X" else 0 for p in X_bar] for X_bar in Xs], axis=1)
        z_weights = np.sum([[1 if p == "Z" else 0 for p in Z_bar] for Z_bar in Zs], axis=1)
        xk_distance = distance.distance(Ixs, Xs, [], self.n_qubits -1)
        zk_distance = distance.distance(Izs, [], Zs, self.n_qubits -1)
        return min(xk_distance, zk_distance)

    def css_distance(self):
        # compute X and Z distances separately
        min_weight = self.n_qubits

        all_generators = self.generate_stabilizers_from_mat()
        x_generators = []
        z_generators = []
        for stab in all_generators:
            if "X" in stab:
                x_generators.append(stab)
            else:
                z_generators.append(stab)
        if len(x_generators) == 0 or len(z_generators) == 0:
            return 0

        # loop over choice of logical qubit
        logical_X_ks = []
        logical_Z_ks = []
        logical_Ix_ks = []
        logical_Iz_ks = []
        for k in range(len(x_generators[0])):
            X_k = []
            Z_k = []
            Ix_k = []
            Iz_k = []
            for gen in x_generators:
                op = gen[:k] + gen[k+1:]
                if gen[k] == "X":
                    X_k.append(op)
                else:
                    Ix_k.append(op)
            for gen in z_generators:
                op = gen[:k] + gen[k+1:]
                if gen[k] == "Z":
                    Z_k.append(op)
                else:
                    Iz_k.append(op)
            logical_X_ks.append(X_k)
            logical_Z_ks.append(Z_k)
            logical_Ix_ks.append(Ix_k)
            logical_Iz_ks.append(Iz_k)

        best_distance = 0
        for X_ks, Z_ks, Ix_ks, Iz_ks in zip(logical_X_ks, logical_Z_ks, logical_Ix_ks, logical_Iz_ks):
            if len(X_ks) == 0 or len(Z_ks) == 0:
                continue
            x_weights = np.sum([[1 if p == "X" else 0 for p in X_bar] for X_bar in X_ks], axis=1)
            z_weights = np.sum([[1 if p == "Z" else 0 for p in Z_bar] for Z_bar in Z_ks], axis=1)
            if np.any(x_weights < best_distance) or np.any(z_weights < best_distance):
                continue
            xk_distance = distance.distance(Ix_ks, X_ks, [], self.n_qubits -1)
            zk_distance = distance.distance(Iz_ks, [], Z_ks, self.n_qubits -1)
            k_distance = min(xk_distance, zk_distance)
            # print(k_distance)
            if k_distance > best_distance:
                best_distance = k_distance

        return best_distance


    def drop_columns(self, legs_to_drop):
        idxs_to_drop = [l for l in legs_to_drop] + [l + self.n_qubits for l in legs_to_drop]
        self.mat = np.delete(self.mat, idxs_to_drop, axis=1)
        self.n_qubits -= len(legs_to_drop)
        self.mat = self.mat[self.mat.any(1)] # drop empty rows
        self.n_rows = self.mat.shape[0]
        return self

    def generate_group(self):
        generators = self.generate_stabilizers_from_mat()
        stab_group = set()
        num_generators = len(generators)
        for bits in product([0, 1], repeat=num_generators):
            if np.sum(bits) > 0:
                p_str_list = [(1, generators[i]) for i, bit in enumerate(bits) if bit]
                p_str = reduce(lambda a, b: self.multiply_pauli_strings_reduce(a, b), p_str_list)
                stab_group.add(p_str[1])
        return stab_group

    def multiply_paulis(self, p1, p2):
        lookup = {'XY': (1j, 'Z'),
                'YX': (-1j, 'Z'),
                'YZ': (1j, 'X'),
                'ZY': (-1j, 'X'),
                'ZX': (1j, 'Y'),
                'XZ': (-1j, 'Y'),
                'IX': (1, "X"),
                'XI': (1, "X"),
                'IY': (1, "Y"),
                'YI': (1, "Y"),
                'IZ': (1, "Z"),
                'ZI': (1, "Z"),
                "XX": (1, "I"),
                "YY": (1, "I"),
                "ZZ": (1, "I"),
                "II": (1, "I")
                }
        return lookup[p1+p2]

    def multiply_pauli_strings_reduce(self, p1, p2):
        pauli_output = []
        c1, c2 = p1[0], p2[0]
        pauli_list1, pauli_list2 = p1[1], p2[1]
        coeff = c1 * c2
        for (p1, p2) in zip(pauli_list1, pauli_list2):
            c, p = self.multiply_paulis(p1, p2)
            coeff *= c
            pauli_output.append(p)
        return coeff, ''.join(pauli_output)

    def check_commute(self, p1, p2):
        return self.multiply_pauli_strings_reduce(p1, p2) == self.multiply_pauli_strings_reduce(p2, p1)

    def parse_generators(self, stabs):
        """separates logical codewords from stabilizers assuming that the first qubit
        is the new "logical" qubit

        Searches to make sure that the qubit is "correctable" first.
        """
        for i in range(len(stabs[0])):
            identities = []
            xbars = []
            zbars = []
            for s in stabs:
                remaining_gen = s[:i] + s[i+1:]
                if s[i] == "I":
                    identities.append(remaining_gen)
                elif s[i] == "X":
                    xbars.append(remaining_gen)
                elif s[i] == "Z":
                    zbars.append(remaining_gen)
            if len(xbars) > 0 and len(zbars) > 0:
                return identities, xbars, zbars
        return None
    def find_code_distance(self):
        '''Computes code distance from check matrix
        Assuming the check matrix is for stabilizers of a state, we can convert it to codewords by treating
        wlog the first qubit as the "logical" qubit.
        This requires finding all of the stabilizers, then finding the min weight codeword on the "physical" qubits
        '''
        generators = self.generate_stabilizers_from_mat()
        parsed_generators = self.parse_generators(generators)
        if parsed_generators is not None:
            return distance.distance(*parsed_generators)
            # stab_from_tn = q.StabilizerCode(*parsed_generators)
            # return stab_from_tn.distance
        else:
            return 0

    def row_reduce_X(self, col_index, swap_dest, other_X_row=None):
        # Returns false if all entries are 0 in the column index
        # find non_zero_rows
        non_zero_rows = []
        for i in range(self.n_rows):
            if self.mat[i, col_index] != 0:
                non_zero_rows.append(i)
        if len(non_zero_rows) == 0:
            return False
        elif len(non_zero_rows) == 1:
            if (other_X_row is not None) and (self.mat[non_zero_rows[0], other_X_row] != 0):
                return False
        # zero out the remaining rows
        elif non_zero_rows[0] == other_X_row:

            non_zero_rows[0], non_zero_rows[1] = non_zero_rows[1], non_zero_rows[0]
        for row_idx in non_zero_rows[1:]:

            self.mat[[row_idx]] = (self.mat[[row_idx]] + self.mat[[non_zero_rows[0]]]) % 2

        # swap it to the top
        self.mat[[non_zero_rows[0], swap_dest]] = self.mat[[swap_dest, non_zero_rows[0]]]
        return True

    def row_reduce_Z(self, z_index, swap_dest, x_index, other_idx=None):
        # Returns false if all entries are 0 in the column (index + n_qubits)
        # other_idx is a tuple containing z and x column indices for the second column
        non_zero_rows = []
        for i in reversed(range(self.n_rows)):
            if self.mat[i, z_index] != 0:
                non_zero_rows.append(i)
        # print(non_zero_rows)
        # should there be a check here to make sure it's not length 1?
        if len(non_zero_rows) == 0:
            return False
        elif len(non_zero_rows) == 1:
            # print(self.mat[non_zero_rows[0], x_index])
            if self.mat[non_zero_rows[0], x_index] != 0:  # check if the X column was already zero'd out
                return False
            if other_idx is not None:
                if self.mat[non_zero_rows[0], other_idx[0]] != 0 or self.mat[non_zero_rows[0], other_idx[1]] != 0:
                    return False
        else:
            # zero out the remaining rows
            for row_idx in non_zero_rows[1:]:
                self.mat[[row_idx]] = (self.mat[[row_idx]] + self.mat[[non_zero_rows[0]]]) % 2

        # TODO: see if there are other issues with degenerate z1 correction
        self.mat[[non_zero_rows[0], swap_dest]] = self.mat[[swap_dest, non_zero_rows[0]]]

        return True


    def check_correctable(self, index):
        bool1 = self.row_reduce_X(index, 0)

        bool2 = self.row_reduce_Z(index+self.n_qubits, int(bool1), index)
        return bool1 and bool2

    def transpose_column(self, idx1, idx2):
        self.mat[:, [idx1, idx2]] = self.mat[:, [idx2, idx1]]

    def shift_column_left(self, col, swap_dest):
        num_swaps = 0
        while col - swap_dest > num_swaps:
            self.transpose_column(col-num_swaps, col-1-num_swaps)
            num_swaps += 1

    def num_independent_eq(self, mat):
        mat_reduced = scipy.optimize._remove_redundancy._remove_redundancy_id(mat, np.zeros_like(mat[:, 0]))[0]
        return len(mat_reduced)


    def self_trace(self, idx1, idx2):
        # swap X's into the first two columns

        self.shift_column_left(idx1, 0)
        self.shift_column_left(idx2, 1)
        self.shift_column_left(idx1 + self.n_qubits, self.n_qubits)
        self.shift_column_left(idx2 + self.n_qubits, self.n_qubits + 1)
        # print(self.mat)

        # Arrange in the form D.9
        X1_bool = self.row_reduce_X(0, 0)
        X2_bool = self.row_reduce_X(1, int(X1_bool), 0)
        Z1_bool = self.row_reduce_Z(self.n_qubits, sum(map(int, [X1_bool, X2_bool])), 0)
        Z2_bool = self.row_reduce_Z(self.n_qubits+1, sum(map(int, [X1_bool, X2_bool, Z1_bool])), 1, (0, self.n_qubits))
        # print(X1_bool, X2_bool, Z1_bool, Z2_bool)
        # print(self.mat)

        # find whether both sides can correct the qubit
        q1_correctable = X1_bool and Z1_bool
        q2_correctable = X2_bool and Z2_bool
        # num_corr_rows  = sum(map(int, [X1_bool, X2_bool, Z1_bool, Z2_bool]))
        # if num_corr_rows > 0:
        #     num_independent_pivots = self.num_independent_eq(self.mat[:num_corr_rows,:])
        # else:
        #     num_independent_pivots = 0

        # print(num_independent_pivots)

        new_mat = []
        buffer_size = self.n_qubits-2 # not needed since all coefficients were the same for rows
        # TODO: Probably don't want to do the deletion here
        M = np.delete(self.mat, [0, 1, 0+self.n_qubits, 1+self.n_qubits], axis=1)
        if q1_correctable and q2_correctable:
            new_mat.append((M[0] + M[1]) % 2)
            new_mat.append((M[2] + M[3]) % 2)
        else:
            # generate all possible linear combinations of the first 3 rows
            first_three_rows = self.mat[:3,:]
            r12 = (first_three_rows[[0]] + first_three_rows[[1]])[0] % 2
            r13 = (first_three_rows[[0]] + first_three_rows[[2]])[0] % 2
            r23 = (first_three_rows[[1]] + first_three_rows[[2]])[0] % 2
            r123 = np.sum(first_three_rows, axis=0) % 2
            row_combinations = [*first_three_rows, r12, r13, r23, r123]
            for row in row_combinations:
                matching = (row[0] == row[1]) and (row[self.n_qubits] == row[self.n_qubits+1])
                if matching:
                    new_mat.append(np.delete(row, [0, 1, self.n_qubits, self.n_qubits + 1]))

        if q1_correctable and q2_correctable:
            for i in range(4, self.n_rows):
                new_mat.append(M[[i]])
        else:
            for i in range(3, self.n_rows):
                new_mat.append(M[[i]])

        self.n_qubits -= 2
        n_rows = len(new_mat)
        if n_rows > 0:
            self.mat = np.vstack(new_mat)
            # perform row reduction
            self.mat = np.array(sympy.Matrix(np.vstack(new_mat)).rref()[0]).astype(int) % 2
            # remove the all-zero rows
            self.mat = self.mat[~np.all(self.mat == 0, axis=1)]
            # assert self.mat.shape == (len(new_mat), 2 * self.n_qubits)
        else:
            self.mat = None
        self.n_rows = len(self.mat)

        return self

    def zero_trace(self, other):
        new_mat = []
        buffer_size_1 = self.n_qubits
        buffer_size_2 = other.n_qubits
        for i in range(len(self.mat)):
            new_mat.append(np.hstack((self.mat[i,:buffer_size_1], np.zeros(buffer_size_2), self.mat[i,buffer_size_1:], np.zeros(buffer_size_2))))
        for i in range(len(other.mat)):
            new_mat.append(np.hstack((np.zeros(buffer_size_1), other.mat[i,:buffer_size_2], np.zeros(buffer_size_1), other.mat[i,buffer_size_2:])))
        self.mat = np.vstack(new_mat)

        self.n_rows = len(self.mat)
        self.n_qubits += other.n_qubits
        return self

    def single_trace(self, other, idx1, idx2):
        # TODO: better name for successful row reduction
        # TODO: double check that this also puts the i, j X/Z entries in the first row
        q1_corr = self.check_correctable(idx1)
        q2_corr = other.check_correctable(idx2)

        new_mat = []
        buffer_size_1 = self.n_qubits-1
        buffer_size_2 = other.n_qubits-1
        M1_start = 2 if q1_corr else 1
        M2_start = 2 if q2_corr else 1
        # M1, M2 = np.zeros((self.n_qubits, 2*self.n_qubits)), np.zeros((other.n_qubits, 2* other.n_qubits)) # just to introduce it in global scope here
        M1 = np.delete(self.mat, [idx1, idx1+self.n_qubits], axis=1)
        M2 = np.delete(other.mat, [idx2, idx2+other.n_qubits], axis=1)
        if q1_corr and q2_corr:
            # eq. D.3 in https://arxiv.org/pdf/2109.08158.pdf
            new_mat.append(np.hstack((M1[0,:buffer_size_1], M2[0,:buffer_size_2],
                        M1[0,buffer_size_1:], M2[0,buffer_size_2:])))
            new_mat.append(np.hstack((M1[1,:buffer_size_1], M2[1,:buffer_size_2],
                        M1[1,buffer_size_1:], M2[1,buffer_size_2:])))

        elif q1_corr or q2_corr:
            # skip the second row since they only match on one row
            # the row being appended is always row 0
            if q1_corr:
            # DONE: What happens if i, j = 0? Skip
                i, j = other.mat[0, [idx2, idx2+other.n_qubits]]
                if i == 0 and j == 0:
                    M2_start -= 1
                else:

                    self.mat[[0]] = (i * self.mat[[0]] + j * self.mat[[1]]) % 2
                    M1[[0]] = (i * M1[[0]] + j * M1[[1]]) % 2
                    new_mat.append(np.hstack((M1[0,:buffer_size_1], M2[0,:buffer_size_2],
                                M1[0,buffer_size_1:], M2[0,buffer_size_2:])))
            else:
                i, j = self.mat[0, [idx1, idx1+self.n_qubits]]
                if i == 0 and j == 0:
                    M1_start -= 1
                else:
                    M2[[0]] = (i * M2[[0]] - j * M2[[1]]) % 2
                    new_mat.append(np.hstack((M1[0,:buffer_size_1], M2[0,:buffer_size_2],
                                M1[0,buffer_size_1:], M2[0,buffer_size_2:])))

        else:
            i1, j1 = self.mat[0, [idx1, idx1+self.n_qubits]]
            i2, j2 = other.mat[0, [idx2, idx2+other.n_qubits]]
            ij1_is_zero = True if (i1 == 0 and j1 == 0) else False
            ij2_is_zero = True if (i2 == 0 and j2 == 0) else False
            if ij1_is_zero:
                M1_start -= 1
            if ij2_is_zero:
                M2_start -= 1

            if ij1_is_zero and ij2_is_zero:
                pass
            elif (i1, j1) == (i2, j2): # special to mod 2
                new_mat.append(np.hstack((M1[0,:buffer_size_1], M2[0,:buffer_size_2],
                            M1[0,buffer_size_1:], M2[0,buffer_size_2:])))

        #A1 0 B1 0
        for i in range(M1_start, self.n_rows):
            candidate_row = np.hstack((M1[i,:buffer_size_1], # A_1
                                        np.zeros(buffer_size_2), # 0
                                        M1[i,buffer_size_1:], # B_1
                                        np.zeros(buffer_size_2))) # 0
            if np.any(candidate_row != 0):
                new_mat.append(candidate_row)
        # 0 A2 0 B2
        for i in range(M2_start, other.n_rows):
            candidate_row = np.hstack((np.zeros(buffer_size_1), # 0
                                        M2[i,:buffer_size_2], # A_2
                                        np.zeros(buffer_size_1), # 0
                                        M2[i,buffer_size_2:])) #B_2
            if np.any(candidate_row != 0):
                new_mat.append(candidate_row)

        # create new check matrix obj here
        # print(new_mat)
        self.mat = np.vstack(new_mat)

        self.n_rows = len(self.mat)
        self.n_qubits += other.n_qubits - 2
        return self
