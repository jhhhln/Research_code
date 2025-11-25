import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from Zi_y import Z1_y,Z2_y,Z3_y
from single_criteria import compute_Z1_values,compute_Z2_values,compute_Z3_values


def check_feasibility(rho, Z1_star, Z2_star, Z3_star,c,r,I,m,b):
    def violation(y):
        Z1_val=Z1_y(y,I,m,c,r,b)
        Z2_val=Z2_y(y,I,m,c,r,b)
        Z3_val=Z3_y(y,I,m,c,r,b)
        return max(0, rho * Z1_star - Z1_val, Z2_val - Z2_star / rho, rho * Z3_star - Z3_val)
    result = minimize_scalar(violation, bounds=(0, b), method='bounded')
    return result.fun <= 1e-4,result.x

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
def run_numerical_experiment_multi(c_values,r,I,m,b):
    results = []
    for c in c_values:
        try:
            rho,multi_y=find_rho_star_and_y(c,r,I,m,b)
            print(rho)
            
            multi_y_profit_uniform=uniform_expected_profit(multi_y, b, r, c)
            mu_t=m[1]
            m_2=m[2]
            multi_y_profit_norm=expected_profit_norm(multi_y, mu_t, m_2, b, r, c)

            
            # y_exp,profit_exp=opt_exp(mu_t, b, r, c)
            # y1_profit_exp=expected_profit_exp(y1, mu_t, b, r, c)
            # y2_profit_exp=expected_profit_exp(y2, mu_t, b, r, c)
            # y3_profit_exp=expected_profit_exp(y3, mu_t, b, r, c)
        except Exception as e:
            print(f"Error at c={c:.1f}: {str(e)}")
            Z1, Z2, Z3, y1, y2, y3 = (np.nan,)*6
        # Z1_y1=Z1_y(y1,I,m,c,r,b)
        # Z1_y2=Z1_y(y2,I,m,c,r,b)
        # Z1_y3=Z1_y(y3,I,m,c,r,b)
        # Z2_y1=Z2_y(y1,I,m,c,r,b)
        # Z2_y2=Z2_y(y2,I,m,c,r,b)
        # Z2_y3=Z2_y(y3,I,m,c,r,b)
        # Z3_y1=Z3_y(y1,I,m,c,r,b)
        # Z3_y2=Z3_y(y2,I,m,c,r,b)
        # Z3_y3=Z3_y(y3,I,m,c,r,b)

        results.append({
            'c': c,
            'r': r,
            'm':m,
            'b':b,
            'I':I,
            'c/r': c/r, 
            'rho':rho,
            'multi_y':multi_y,
            'multi_y_profit_uniform':multi_y_profit_uniform,
            'multi_y_profit_norm':multi_y_profit_norm,

    
            # 'y_exp':y_exp,
            # 'profit_exp':profit_exp,
            # 'y1_profit_exp':y1_profit_exp,
            # 'y2_profit_exp':y2_profit_exp,
            # 'y3_profit_exp':y3_profit_exp,


            # # 'y_exp':y_exp,
            # 'y_normal':y_normal,
            # # 'y_uni':y_uni,
            # 'Z1_y1':Z1_y1,
            # 'Z2_y1':Z2_y1,
            # 'Z3_y1':Z3_y1,
            # 'Z1_y2':Z1_y2,
            # 'Z2_y2':Z2_y2,
            # 'Z3_y2':Z3_y2,
            # 'Z1_y3':Z1_y3,
            # 'Z2_y3':Z2_y3,
            # 'Z3_y3':Z3_y3,
            # 'Z1_y_exp':Z1_y_exp,
            # 'Z2_y_exp':Z2_y_exp,
            # 'Z3_y_exp':Z3_y_exp,
            # 'Z1_y_normal':Z1_y_normal,
            # 'Z2_y_normal':Z2_y_normal,
            # 'Z3_y_normal':Z3_y_normal,
            # 'Z1_y_uni':Z1_y_uni,
        })
    results_df = pd.DataFrame(results)
    return results_df
# 使用示例
if __name__ == "__main__":
    I = 2
    m = np.array([1,2,6])
    r = 1
    b = 5
    c_values=result = [round(i * 0.05, 2) for i in range(1, 20)]  

    results_df = run_numerical_experiment_multi(c_values,r,I,m,b)
    results_df.to_csv('multi_rc_vary_meanvariance_2.csv', index=False)
    print("实验结果已保存")