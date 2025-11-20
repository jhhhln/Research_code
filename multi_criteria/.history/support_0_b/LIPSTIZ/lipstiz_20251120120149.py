import numpy as np
from scipy.special import comb
import random
from scipy.optimize import minimize_scalar, minimize
from sdp import solve_sdp
from fractions import Fraction

def optimize_s(I, m, c, r, b, y, lambda_val, s_min, s_max, num_iterations):
    L = lambda_val * (r + c)
    evaluated_s_points = []
    evaluated_g_values = []
    
    s_1 = random.uniform(s_min, s_max)
    g_1_raw = solve_sdp(I, m, c, r, b, y, s_1, lambda_val)
    if isinstance(g_1_raw, (float, int)):
        g_1 = float(g_1_raw)
    else:
        g_1 = float('-inf')
    
    evaluated_s_points.append(s_1)
    evaluated_g_values.append(g_1)
    
    best_s = s_1
    max_g_value = float(g_1)
    
    current_eval_count = 1

    # 3. 迭代
    while current_eval_count < num_iterations:
        s_candidate = random.uniform(s_min, s_max)
        
        # 计算候选点的上限 (UB)
        # UB(s_candidate) = min_{i=1...t} (g(s_i, y) + L * abs(s_candidate - s_i))
        upper_bound_at_candidate = float('inf')
        for i in range(len(evaluated_s_points)):
            g_val = evaluated_g_values[i]
            if g_val is None:
                continue
            ub_i = g_val + L * abs(s_candidate - evaluated_s_points[i])
            if ub_i < upper_bound_at_candidate:
                upper_bound_at_candidate = ub_i
        
        # 决策规则：如果候选点是潜在的最大值，则进行评估
        if upper_bound_at_candidate is not None and max_g_value is not None and float(upper_bound_at_candidate) >= float(max_g_value):
            g_candidate_raw = solve_sdp(I, m, c, r, b, y, s_candidate, lambda_val)
            if isinstance(g_candidate_raw, (float, int)):
                g_candidate = float(g_candidate_raw)
            else:
                g_candidate = float('-inf')
            evaluated_s_points.append(s_candidate)
            evaluated_g_values.append(g_candidate)
            current_eval_count += 1
            # 更新观察到的最佳值
            if max_g_value is None or g_candidate > max_g_value:
                max_g_value = g_candidate
                best_s = s_candidate
        if current_eval_count >= num_iterations:
            break

    # 4. 输出
    return best_s, max_g_value


def optimize_y(I, m, c, r, b, lambda_val, y_min, y_max, s_min, s_max, lipo_iters):
    if lambda_val==0:
        s_opt =-0.1
        def objective_y(y):
            return solve_sdp(I, m, c, r, b, y, s_opt, lambda_val)
        y_min, y_max =0,b
        
        result = minimize(
            objective_y,
            x0=np.array([y_max]),
            bounds=[(y_min, y_max)],
            method='L-BFGS-B',
            options={'ftol': 1e-8, 'gtol': 1e-6, 'maxiter': 1000}
        )

        optimal_y = result.x[0]
        min_g_value = result.fun

    else:
        def min_g_y(y_val):
            _, max_g = optimize_s(I, m, c, r, b, y_val, lambda_val, s_min, s_max, lipo_iters)
            return max_g 

        res = minimize_scalar(
            min_g_y,
            bounds=(y_min, y_max),
            method='bounded',
            options={'disp': True}
        )
        if getattr(res, 'success', False):
            optimal_y = getattr(res, 'x', None)
            min_g_value = getattr(res, 'fun', 0)
            print(f'关于y的优化成功,最优y: {optimal_y}, 最小g(s,y): {min_g_value}')
        else:
            optimal_y = None
            min_g_value = None
            print(f'关于y的优化失败: {getattr(res, "message", "未知错误")}')
    return optimal_y, min_g_value



if __name__ == "__main__":
    I = 1
    m = np.array([1, 4])
    c = 12
    r = 13
    b = 5
    y = 0.5
    s = 0.8
    lambda_val =1

    # optimal_value = solve_sdp(I, m, c, r, b, y, s, lambda_val)
    # print("Optimal value of the SDP:", optimal_value)
    print(optimize_y(I, m, c, r, b, lambda_val, a, b, 0, b, 30))