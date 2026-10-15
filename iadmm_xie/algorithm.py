import numpy as np
import time

from x_problem import x_subproblem
from conditions import inexact_error

def iadmm_algorithm(A, D, w1, b, c, mu, beta, tau, sigma, eta, m, n, s, method_init):
    ## Define x_p and y_p (primal) | l_d (dual)
    x_p, y_p, l_d = np.zeros(n), np.zeros(s), np.zeros(s)

    while True: 
        # Step 1 - Computing x_p in subproblem
        x_p, d = x_subproblem(method_init, A, D, x_p, y_p, l_d, w1, c, beta, tau, sigma, mu)

        ## Step 2 - Solving y-subproblem
        y_n = np.sign(D @ x_p - (l_d / beta)) * np.maximum(np.abs(D @ x_p - (l_d / beta)) - (1/beta), 0)

        error = inexact_error(D, x_p, y_p, y_n, beta)
        if error < eta:
            print(f"Error passed! error = {error} | eta = {eta}")
            return
        else: print(f"Error fail: error = {error} | eta = {eta}")

        # Updating y_p
        y_p = y_n

        ## Step 3 - Updating l_d and w1
        l_d = l_d - tau * beta * (D @ x_p - y_n - b)
        w1 = w1 - tau * beta * d
