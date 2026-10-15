import numpy as np

## Comments:
## This code implements the inexact ADMM with relative error 
## criterion algorithm. The main loop is performing the minimization problem. 
## Author(s): Jiaxin Xie

from data_generation import gen_data
from algorithm import iadmm_algorithm

def main():
    # Dimensions of the problems, sparsity of vector definition
    m, n = 2048, 256
    s = int(np.round(0.1 * m))

    matrix_init = 'dct'               ## Specifying how the D matrix is defined
    method_def = 'chol'               ## Specify how we solve the x-subproblem ('cg' or 'chol')

    ## Constants to algorithm 1 and algorithm 2 in the paper 'On inexact ADMMs with relative error criteria.'
    mu = 1000
    beta = 25
    sigma = 0.1
    eta = 0.7
    alpha = 0.5
    gamma = 1.0 - alpha
    tau_sup = (2.0 * gamma - 1.0 + np.sqrt(4.0 * gamma ** 2.0 + 1.0)) / (2.0 * gamma)
    tau = 0.5 * tau_sup

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    b = np.zeros(s)
    w1 = np.zeros(n)
    for trial in range(1, 11):  
        print("*" * 60)
        A, D, x_sol, c = gen_data(m, n, s, matrix_init)
        iadmm_algorithm(A, D, w1, b, c, mu, beta, tau, sigma, eta, m, n, s, method_def)

    print("*" * 60, "\n")

    # A, D = np.random.randn(m, n), np.random.randn(s, n)
    # c = np.random.randn(m)
    # # Defining the unknown variables
    # x_p, y_p, l_d = np.zeros(n), np.random.randn(s), np.random.randn(s)

    # cg_dict = {
    #     "A" : A,
    #     "D" : D,
    #     "y" : y_p,
    #     "l" : l_d,
    #     "c" : c,
    #     "mu" : mu,
    #     "beta" : beta, 
    #     "sigma" : sigma
    # }

    # # Executing the CG method
    # conjugate_class = cg_class(**cg_dict)
    # x_p = conjugate_class.cg_method(x_p)

if __name__ == '__main__':
	main()
