import copy
import itertools
import numpy as np
import topology as topo

# phase 1
def sds_phase1_or_3(n_sub: list[int], n_sub_facts: list[list[int]]) -> list[np.ndarray]:
    n = sum(n_sub)
    max_period = max(len(fact) for fact in n_sub_facts)
    return_A_list = []
    for i in range(max_period):
        A_full = np.zeros((n, n))
        start = 0
        for sub_size, sub_facts in zip(n_sub, n_sub_facts):
            a_sub = topo.dynamic_hypercuboid(i, sub_facts)
            stop = start + sub_size
            A_full[start:stop, start:stop] = a_sub
            start = stop
        return_A_list.append(A_full)
    return return_A_list


# Phase 2
def sds_phase2(n_sub: list[int], n_sub_facts: list[list[int]]) -> list[np.ndarray]:
    s = 0
    n = sum(n_sub)
    num_blocks = len(n_sub)
    return_A_list2 = []
    for s_block in range(num_blocks - 1):
        A_full = np.zeros((n, n))
        m_self = n_sub[s_block]
        m_all = sum(n_sub[s_block:])
        m_k_rest = sum(n_sub[s_block + 1 :])
        # Previous
        A_full[:s, :s] = np.eye(s)
        # A_11
        A_full[s : s + m_self, s : s + m_self] = np.eye(m_self)
        A_full[s : s + m_k_rest, s : s + m_k_rest] = np.eye(m_k_rest) * m_k_rest / m_all
        # A_12
        A_full[s : s + m_k_rest, s + m_self :] = np.eye(m_k_rest) * m_self / m_all
        # A_21
        A_full[s + m_self :, s : s + m_k_rest] = np.eye(m_k_rest) * m_self / m_all
        # A_22
        A_full[s + m_self : s + m_all, s + m_self : s + m_all] = (
            np.eye(m_k_rest) * m_k_rest / m_all
        )

        return_A_list2.append(A_full)
        s += m_self
    return return_A_list2


def rhb_phase_2(n_sub: list[int]) -> np.ndarray:
    assert len(n_sub) > 1
    n = sum(n_sub)
    num_blocks = len(n_sub)
    A = np.zeros((n, n))
    
    s = sum(n_sub[:-1]) # construct the matrix in the reverse order
    n1 = n_sub[-2]
    n2 = n_sub[-1]
    # The last A_22
    A[s:, s:] = np.eye(n2)
    A[s, s] = n2 ** 2 / n - n2 + 1  #alpha_2
    for s_block in range(num_blocks - 2, -1, -1):
        n1 = n_sub[s_block]
        s -= n1 # reverse order

        # A_11
        A[s:s+n1, s:s+n1] = np.diag([1] * n1)
        A[s, s] = n1**2/n - n1 + 1 # alpha
        # A_12 and A_21
        for other_block in range(s_block+1, num_blocks):
            rest = sum(n_sub[s_block+1:other_block])
            A[s+n1+rest, s+rest] = n1 * n_sub[other_block] / n
            A[s+rest, s+n1+rest] = n1 * n_sub[other_block] / n
        # Leave the A_22 untouched because it is already handled
        
    return A

def get_all_sds_matrices(
    n_sub: list[int], n_sub_facts: list[list[int]]
) -> list[np.ndarray]:
    return_A_list1 = sds_phase1_or_3(n_sub, n_sub_facts)
    return_A_list2 = sds_phase2(n_sub, n_sub_facts)
    return_A_list3 = copy.deepcopy(return_A_list1)
    return list(itertools.chain(return_A_list1, return_A_list2, return_A_list3))

def get_all_dshb_matrices(
    n_sub: list[int], n_sub_facts: list[list[int]]
) -> list[np.ndarray]:
    return_A_list1 = sds_phase1_or_3(n_sub, n_sub_facts)
    return_A_list2 = sds_phase2(n_sub, n_sub_facts)
    A_together = return_A_list2[0]
    for A in return_A_list2[1:]:
        A_together = A @ A_together 
    return_A_list3 = copy.deepcopy(return_A_list1)
    return list(itertools.chain(return_A_list1, [A_together], return_A_list3))

def get_all_rhb_matrices(
    n_sub: list[int], n_sub_facts: list[list[int]]
) -> list[np.ndarray]:
    return_A_list1 = sds_phase1_or_3(n_sub, n_sub_facts)
    A = rhb_phase_2(n_sub)
    return_A_list3 = copy.deepcopy(return_A_list1)
    return list(itertools.chain(return_A_list1, [A], return_A_list3))