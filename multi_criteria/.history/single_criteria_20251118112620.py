import numpy as np
from lipstiz import optimize_y


#输出的就是每个准则下的最优表现
def compute_Z1_values(I, m, c, r, a,b):
    y_opt_Z1,Z1=optimize_y(I, m, c, r, a,b, lambda_val=0, lipo_iters=30)
    Z1=-Z1
    return Z1,y_opt_Z1

def compute_Z2_values(I, m, c, r,a, b):
    y_opt_Z2,Z2=optimize_y(I, m, c, r,a, b, lambda_val=1)
    return Z2,y_opt_Z2

def compute_Z3_values(I, m, c, r, b):
    lambda_min, lambda_max = 0, 1 
    epsilon = 1e-3
    Z3 = 0
    
    while lambda_max - lambda_min > epsilon:
        lambda_mid = (lambda_min + lambda_max) / 2
        y_opt_mid, g_opt_mid = optimize_y(I, m, c, r, b, lambda_val=lambda_mid)
        
        if g_opt_mid <= 0:
            lambda_min = lambda_mid
            Z3 = lambda_mid

        else:
            lambda_max = lambda_mid
    #y_opt_Z3,_=optimize_y(I, m, c, r, b, lambda_val=Z3)
    
    return Z3,y_opt_mid


if __name__ == "__main__":
    I = 1
    m = np.array([1,0.3])
    c = 0.3
    r = 1
    b = 1

    # Z1,y1=compute_Z1_values(I, m, c, r, b)
    # print("Z1 value:", Z1," at y:",y1)
    Z2,y2=compute_Z2_values(I, m, c, r, b)
    print("Z2 value:", Z2," at y:",y2)
    # Z3,y3=compute_Z3_values(I, m, c, r, b)
    # print("Z3 value:", Z3," at y:",y3)