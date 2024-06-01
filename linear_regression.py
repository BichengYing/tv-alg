"""Helper Functions for Linear Regression for multiple workers via numpy.

We want to solve the least square problem
$$
  \min_x \frac{1}{n} \sum_{i=1}^n\|A_i x - b_i\|_2^2
$$
"""

import numpy as np


def generate_linear_regression_data(
    num_data: int, dimension: int, num_workers: int, noise_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates N data pairs (A_i, b_i) of dimension d that follow a linear
    regression model.
    """
    A = np.random.randn(num_data, dimension)
    x_o = np.random.randn(dimension, 1)
    ns = noise_level * np.random.randn(num_data, 1)
    b = A.dot(x_o) + ns

    x_opt = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)

    As = []
    bs = []
    for i in range(num_workers):
        As.append(A[i::num_workers])
        bs.append(b[i::num_workers])
    return As, bs, x_opt


# def grad_fn_agent(x, Aarr, barr, worker_index, reg, minibatch: int | None = None):
#   """ Returns the gradient of the least squares problem with non-convex
#   regularizer.
#   """
#   A = Aarr[worker_index]
#   b = barr[worker_index]
#   if isinstance(minibatch, int):
#     subsample = np.random.choice(A.shape[0], size=minibatch, replace=False)
#     A = A[subsample, :]
#     b = b[subsample, :]
#   return A.T.dot(A.dot(x) - b) + reg*(2*x)/(x**2+1)**2


def grad_fn_agent(x, Aarr, barr, worker_index, reg, noise=False):
    """Returns the gradient of the least squares problem with non-convex
    regularizer.
    """
    A = Aarr[worker_index]
    b = barr[worker_index]
    if noise:
        mu, sigma = 0, 0.0001  # mean and standard deviation
        s = np.random.normal(mu, sigma, x.shape)
        return A.T.dot(A.dot(x) - b) + reg * (2 * x) / (x**2 + 1) ** 2 + s
    return A.T.dot(A.dot(x) - b) + reg * (2 * x) / (x**2 + 1) ** 2


def conj_grad_fun_agent(y, Aarr, barr, worker_index):
    A = Aarr[worker_index]
    b = barr[worker_index]
    return np.linalg.inv(A.T @ A) @ (0.5 * y + A.T @ b)


def grad_fn_global(xfull, Aarr, barr, reg):
    """
    Returns \nabla f(\bar{x}^k).
    """
    x_avg = np.average(xfull, axis=1)[:, np.newaxis]
    grad_global = np.zeros_like(x_avg)
    for a in range(len(Aarr)):
        grad_global += grad_fn_agent(x_avg, Aarr, barr, a, reg)
    grad_global /= len(Aarr)
    return grad_global


def grad_avg(xfull, Aarr, barr, reg):
    """
    Returns \overline \f(\bar{x}^k).
    """
    x_avg = np.average(xfull, axis=1)[:, np.newaxis]
    grad_avg = np.zeros_like(x_avg)
    for a in range(len(Aarr)):
        grad_avg += grad_fn_agent(xfull[:, [a]], Aarr, barr, a, reg)
    grad_avg /= len(Aarr)
    return grad_avg
