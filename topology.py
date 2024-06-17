"""Helper function defined for various basic topology (weighted adjacency matrix)"""

import functools
import numpy as np


@functools.cache
def _dynamic_exp2(iter, size):
    W = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if (j == i) or (np.log2((j - i) % size) == iter):
                W[i, j] = 1 / 2
    return W


def dynamic_exp2(iter, size=8):
    tau = np.ceil(np.log2(size))
    _iter = iter % tau
    return _dynamic_exp2(_iter, size)


def ring(iter, size):
    x = np.array([0.0 for i in range(size)])
    x[0], x[1], x[-1] = 1.0, 1.0, 1.0
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    return topo


def star(iter, size):
    topo = np.empty((size, size))
    topo = np.eye(size) * (size - 1) / size
    for i in range(size):
        topo[i, 0] = 1 / size
        topo[0, i] = 1 / size
    return topo


@functools.cache
def _static_exp2(size):
    # assert size & (size-1) == 0
    x = np.array([1.0 if i & (i - 1) == 0 else 0 for i in range(size)])
    x /= x.sum()
    topo = np.empty((size, size))
    for i in range(size):
        topo[i] = np.roll(x, i)
    return topo


def static_exp2(iter, size):
    del iter
    return _static_exp2(size)


def decompose(n, factors):
    result = []
    residue = n
    for k in range(1, len(factors)):
        base = np.prod(factors[k:])
        quotient, residue = divmod(residue, base)
        result.append(quotient)
    result.append(residue)
    return result


def diff_by_one(list1: list[int], list2: list[int]) -> bool:
    if len(list1) != len(list2):
        raise ValueError
    diff = 0
    for i, j in zip(list1, list2):
        diff += 0 if i == j else 1
    return diff == 1


def diff_by_shift_one(list1: list[int], list2: list[int]) -> bool:
    if len(list1) != len(list2):
        raise ValueError
    diff = 0
    for i, j in zip(list1[1:], list2[:-1]):
        diff += 0 if i == j else 1
    return diff == 0


@functools.cache
def _static_hypercuboid(factors):
    size = int(np.prod(factors))
    W = np.zeros((size, size))
    for i in range(size):
        i_decomposed = decompose(i, factors)
        for j in range(size):
            if i == j:
                W[i, j] = 1
                continue
            j_decomposed = decompose(j, factors)
            if diff_by_one(i_decomposed, j_decomposed):
                W[i, j] = 1
    return W / W.sum(axis=1)


def static_hypercuboid(iter, factors):
    del iter
    return _static_hypercuboid(tuple(factors))


@functools.cache
def _static_static_de_bruijn(factors):
    size = int(np.prod(factors))
    W = np.zeros((size, size))
    for i in range(size):
        i_decomposed = decompose(i, factors)
        for j in range(size):
            j_decomposed = decompose(j, factors)
            if diff_by_shift_one(i_decomposed, j_decomposed):
                W[i, j] = 1
    return W / W.sum(axis=1)


def static_de_bruijn(iter, factors):
    del iter
    return _static_static_de_bruijn(tuple(factors))


@functools.cache
def _dynamic_hypercuboid(iter, factors):
    W = 1
    for i, f in enumerate(factors):
        W_i = np.ones([f, f]) / f if i == iter else np.eye(f)
        W = np.kron(W_i, W)
    return W


def dynamic_hypercuboid(iter, factors):
    _iter = iter % len(factors)
    return _dynamic_hypercuboid(_iter, tuple(factors))
