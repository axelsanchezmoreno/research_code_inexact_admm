import numpy as np

def cg_condition(C, x, b):
    num = np.linalg.norm(C @ x - b, ord=2)
    den = max(np.linalg.norm(b, ord=2), 1.0)
    return (num / den) < 1e-8

def x_cond(D, x_p, y_p, w1, d, tau, sigma, beta):
    lhs = (2.0 / beta) * np.abs((w1 - x_p).T @ d)
    rhs = sigma * np.linalg.norm(D @ x_p - y_p, ord=2) ** 2
    return (lhs + tau * np.linalg.norm(d) <= rhs)

def inexact_error(D, x_p, y_p, y_n, beta):
    t1_num = np.linalg.norm(D @ x_p - y_p, ord=2)
    t2_num = np.linalg.norm(beta * D.T @ (y_n - y_p), ord=2)

    m1t1_den = np.linalg.norm(D @ x_p)
    m1t2_den = np.linalg.norm(y_p)

    m2t1_den = np.linalg.norm(D.T @ y_p)

    den_t1 = max(m1t1_den, m1t2_den, 1.0)
    den_t2 = max(1.0, m2t1_den)

    result = max(t1_num / den_t1, t2_num / den_t2)

    return result 