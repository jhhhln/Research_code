from sdp import solve_sdp
from lipstiz import optimize_s,optimize_y
import numpy as np

#结果就是所需要的，不需要转换 表示的就是规定的Z_(i,F)的含义
def Z1_y(y,I,m,c,r,b):
    lambda_val=0
    s=0
    value=solve_sdp(I, m, c, r, b, y, s, lambda_val)
    value=-value
    return value

def Z2_y(y,I,m,c,r,b):
    _,R=optimize_s(I, m, c, r, b, y, lambda_val=1,num_iterations=30)
    return R

def Z3_y(y, I, m, c, r, b):
    lambda_min = 0
    lambda_max = 1
    tolerance = 1e-3
    lambda_star = None

    while lambda_max - lambda_min > tolerance:
        lambda_mid = (lambda_min + lambda_max) / 2
        s_opt_mid, value = optimize_s(I, m, c, r, b, y, lambda_val=lambda_mid,num_iterations=30)

        if value <= 0:
            lambda_min = lambda_mid
            lambda_star = lambda_mid
        else:
            lambda_max = lambda_mid

    if lambda_star is None:
        lambda_star = (lambda_min + lambda_max) / 2
    return lambda_star



if __name__ == "__main__":
    I = 1
    m = np.array([1, 0.3])
    c = 1
    r = 2
    b = 1
    y = 0.5

    Z1_value=Z1_y(y,I,m,c,r,b)
    print("Z1 value:", Z1_value)

    Z2_value=Z2_y(y,I,m,c,r,b)
    print("Z2 value:", Z2_value)

    Z3_value=Z3_y(y,I,m,c,r,b)
    print("Z3 value:", Z3_value)