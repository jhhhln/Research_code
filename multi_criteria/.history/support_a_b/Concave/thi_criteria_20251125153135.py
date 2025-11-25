import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from Zi_y import Z1_y,Z2_y,Z3_y
from single_criteria import compute_Z1_values,compute_Z2_values,compute_Z3_values


def check_feasibility(rho, Z1_star, Z2_star, Z3_star,c,r,I,m,a,b):
    def violation(y):
        Z1_val=Z1_y(y,I,m,c,r,b)
        Z2_val=Z2_y(y,I,m,c,r,b)
        Z3_val=Z3_y(y,I,m,c,r,b)
        return max(0, rho * Z1_star - Z1_val, Z2_val - Z2_star / rho, rho * Z3_star - Z3_val)
    result = minimize_scalar(violation, bounds=(0, b), method='bounded')
    return result.fun <= 1e-3,result.x

def find_rho_star_and_y(c, r, I, m, b):
    Z1,y_opt_Z1=compute_Z1_values(I, m, c, r, b)
    Z2,y_opt_Z2=compute_Z2_values(I, m, c, r, b)
    Z3,y_opt_Z3=compute_Z3_values(I, m, c, r, b)
    rho_min, rho_max = 0, 1
    tolerance = 1e-3
    
    best_rho = 0
    best_y = None
    found_feasible_solution = False  
    
    while rho_max - rho_min > tolerance:
        rho_mid = (rho_min + rho_max) / 2
        is_feasible, y = check_feasibility(rho_mid, Z1, Z2, Z3, c, r, I, m, b)
        
        if is_feasible:
            rho_min = rho_mid
            best_rho = rho_mid
            best_y = y
            found_feasible_solution = True 
        else:
            rho_max = rho_mid

    if not found_feasible_solution:
        print(f"警告：对于参数 c={c}, r={r}, I={I}, m={m}, b={b} 未找到可行解")
        best_y = None  
        best_rho = 0       
    
    return best_rho, best_y