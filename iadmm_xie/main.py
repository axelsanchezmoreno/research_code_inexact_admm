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

    # Defining the constants, mu, beta, and sigma
    mu, beta, sigma = 0.01, 25, 0.5

    print(f"Problem: m, n, s = {m}, {n}, {s}\n")

    for trial in range(1, 11):
        A, D, x_sol, c = gen_data(m, n, s, matrix_init)
        iadmm_algorithm(A, D, c, mu, beta, sigma, m, n, s, method_def)

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
