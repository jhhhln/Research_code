import numpy as np
from scipy.optimize import minimize_scalar, minimize
from sdp_with_a_b import solve_sdp


def optimize_s(I, m, c, r,a,b, y, lambda_val):
    a2 = (lambda_val * b - b + y) / lambda_val
    a3 = y
    a4 = (lambda_val*a-a+y) / lambda_val
    theoretical_intervals = [(-np.inf, a2), (a2, a3), (a3, a4), (a4, np.inf)]
    #s\in[0,b]
    valid_intervals = []
    for interval in theoretical_intervals:
        left = max(a, interval[0])
        right = min(b, interval[1])
        if left < right:
            valid_intervals.append((left, right))
    
    best_s_overall = None
    best_value_overall = -np.inf
    
    for i, (left, right) in enumerate(valid_intervals):

        endpoint_values = []
        for s in [left, right]:
            val = solve_sdp(I, m, c, r,a, b, y, s, lambda_val)
            endpoint_values.append((s, val))
        try:
            result = minimize_scalar(
                lambda s: -solve_sdp(I, m, c, r,a, b, y, s, lambda_val), 
                bounds=(left, right),
                method='bounded'
            )
            interior_s = result.x
            interior_value = -result.fun
            
            if not np.isclose(interior_s, left) and not np.isclose(interior_s, right):
                endpoint_values.append((interior_s, interior_value))
        except Exception as e:
            print(f" 内部极值搜索失败: {str(e)}")
        
        # 确定区间最优
        if endpoint_values:
            local_best_s, local_best_value = max(endpoint_values, key=lambda x: x[1])
            
            # 更新全局最优
            if local_best_value >= best_value_overall:
                best_s_overall = local_best_s
                best_value_overall = local_best_value

    return best_s_overall, best_value_overall


def optimize_y(I, m, c, r, a,b, lambda_val):
    if lambda_val==0:
        s_opt =-0.1
        def objective_y(y):
            return solve_sdp(I, m, c, r,a, b, y, s_opt, lambda_val)
        y_min, y_max =a,b
        
        result = minimize(
            objective_y,
            x0=np.array([y_max]),
            bounds=[(y_min, y_max)],
            method='L-BFGS-B',
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 1000}
        )

        y_opt = result.x[0]
        min_max_g = result.fun
        return y_opt, min_max_g
    else:
        def objective(y):
            _, max_g = optimize_s(I, m, c, r,a, b, y, lambda_val)
            return max_g 
        result = minimize_scalar(objective, bounds=(0, b), method='bounded')
        y_opt = result.x
        g_opt=objective(y_opt)
        return y_opt, g_opt


if __name__ == "__main__":
    I = 1
    m = np.array([1, 2])
    c = 0.4
    r =0.5

    a=1
    b =3

    lambda_val =1
    y_opt, g_opt = optimize_y(I, m, c, r,a, b, lambda_val)
    print(f"Optimal y: {y_opt}, Min max g(s,y): {g_opt}")