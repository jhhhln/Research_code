from sdp_with_a_b import solve_sdp
from R_star_concave import optimize_s
import numpy as np

#结果就是所需要的，不需要转换 表示的就是规定的Z_(i,F)的含义
def Z1_y(y,I,m,c,r,a,b):
    lambda_val=0
    s=0
    value=solve_sdp(I, m, c, r,a, b, y, s, lambda_val)
    value=-value
    return value

def Z2_y(y,I,m,c,r,a,b):
    _,R=optimize_s(I, m, c, r, a,b, y, lambda_val=1)
    return R

def Z3_y(y,I,m,c,r,a,b):
    lambda_min=0
    lambda_max=1
    tolerance=1e-4
    lambda_star=0
    
    while lambda_max - lambda_min > tolerance:
        lambda_mid = (lambda_min + lambda_max) / 2
        s_opt_mid, value = optimize_s(I, m, c, r, a,b, y, lambda_val=lambda_mid)
        if value <= 0:
            lambda_min = lambda_mid
            lambda_star = lambda_mid
        else:
            lambda_max = lambda_mid
    return lambda_star




if __name__ == "__main__":
    I = 1
    m = np.array([1,0.3])
    c = 0.3
    r = 1
    b = 1
    y=0.38

    # Z1_value=Z1_y(y,I,m,c,r,b)
    # print("Z1 value:", Z1_value)

    Z2_value=Z2_y(y,I,m,c,r,a,b)
    print("Z2 value:", Z2_value)

    # Z3_value=Z3_y(y,I,m,c,r,b)
    # print("Z3 value:", Z3_value)