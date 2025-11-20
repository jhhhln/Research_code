import numpy as np
import random
from scipy.optimize import minimize_scalar, minimize
from support_a_b.Concave.sdp_with_a_b import solve_sdp


def optimize_s(I, m, c, r, a, b, y, lambda_val, num_iterations):
    L = lambda_val * (r + c)
    
    evaluated_s_points = []
    evaluated_g_values = []

    # 确保范围正确
    s_min = a
    s_max = b

    # 1. 初始化：随机采样第一个点
    s_1 = random.uniform(s_min, s_max)
    g_1_raw = solve_sdp(I, m, c, r, a, b, y, s_1, lambda_val)
    
    if isinstance(g_1_raw, (float, int)):
        g_1 = float(g_1_raw)
    else:
        g_1 = float('-inf')
    
    evaluated_s_points.append(s_1)
    evaluated_g_values.append(g_1)
    
    best_s = s_1
    max_g_value = float(g_1)
    
    current_eval_count = 1

    # --- [关键修改] 引入拒绝计数器 ---
    consecutive_rejections = 0
    max_rejections_allowed = 100  # 允许连续失败的次数，超过则认为已收敛

    # 2. 迭代
    while current_eval_count < num_iterations:
        s_candidate = random.uniform(s_min, s_max)
        
        # 计算候选点的上限 (UB)
        upper_bound_at_candidate = float('inf')
        valid_history = False

        for i in range(len(evaluated_s_points)):
            g_val = evaluated_g_values[i]
            # 跳过无效值
            if g_val == float('-inf') or g_val is None:
                continue
            
            valid_history = True
            # Lipschitz 上界计算: f(x) <= f(xi) + L * |x - xi|
            ub_i = g_val + L * abs(s_candidate - evaluated_s_points[i])
            if ub_i < upper_bound_at_candidate:
                upper_bound_at_candidate = ub_i

        # 决策规则：如果 UB >= 当前最大值，则评估
        # 注意：增加了 valid_history 检查，防止列表全为 -inf 时出错
        if valid_history and upper_bound_at_candidate >= max_g_value:
            g_candidate_raw = solve_sdp(I, m, c, r, a, b, y, s_candidate, lambda_val)
            
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
            
            # [关键修改] 成功评估一次，重置拒绝计数器
            consecutive_rejections = 0

        else:
            # [关键修改] 候选点被拒绝，计数器 +1
            consecutive_rejections += 1
            
            # 如果连续拒绝次数过多，说明搜索空间已极小，强制退出
            if consecutive_rejections > max_rejections_allowed:
                # print(f"LIPO converged early (Search space exhausted) at eval {current_eval_count}")
                break

    # 4. 输出
    return best_s, max_g_value


def optimize_y(I, m, c, r,a, b, lambda_val, lipo_iters):
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

        optimal_y = result.x[0]
        min_g_value = result.fun

    else:
        def min_g_y(y_val):
            _, max_g = optimize_s(I, m, c, r, a,b, y_val, lambda_val, lipo_iters)
            return max_g 
        y_min, y_max =a,b
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
    m = np.array([1, 6])
    c = 3
    r = 10

    a=3
    b = 10


    lambda_val =0
    # optimal_value = solve_sdp(I, m, c, r,a, b, y, s, lambda_val)
    # print("Optimal value of the SDP:", optimal_value)
    print(optimize_y(I, m, c, r,a, b, lambda_val, 30))