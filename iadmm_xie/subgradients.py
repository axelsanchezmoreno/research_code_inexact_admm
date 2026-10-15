def grad_f():
    pass

def df_subgrad(A, D, x_p, y_p, l_d, c, beta, mu):
    return (1.0 / mu) * A.T @ (A @ x_p - c) - D.T @ (l_d - beta * (D @ x_p - y_p))

# def dg_subgrad():
#     pass