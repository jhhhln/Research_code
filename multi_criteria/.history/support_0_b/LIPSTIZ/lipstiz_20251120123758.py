import numpy as np
import random
from scipy.optimize import minimize_scalar, minimize
from sdp import solve_sdp

import numpy as np
import random
from sdp import solve_sdp

def optimize_s(I, m, c, r, b, y, lambda_val, num_iterations):
    # 1. 确定 Lipschitz 常数 k
    # 论文中 k 需要已知 [cite: 90]，这里你定义为 L
    L = lambda_val * (r + c)
    
    evaluated_s_points = []
    evaluated_g_values = []
    s_min = 0
    s_max = b
    
    # 2. 初始化 (LIPO Step 1: Initialization [cite: 95])
    # 随机采样第一个点
    s_1 = random.uniform(s_min, s_max)
    g_1_raw = solve_sdp(I, m, c, r, b, y, s_1, lambda_val)
    
    if isinstance(g_1_raw, (float, int)):
        g_1 = float(g_1_raw)
    else:
        g_1 = float('-inf')
    
    evaluated_s_points.append(s_1)
    evaluated_g_values.append(g_1)
    
    best_s = s_1
    max_g_value = g_1
    
    current_eval_count = 1
    
    # --- 新增：防止死循环的计数器 ---
    consecutive_rejections = 0
    max_rejections_allowed = 1000  # 允许连续失败的次数，可根据精度要求调整
    
    # 3. 迭代 (LIPO Step 2: Iterations [cite: 96])
    while current_eval_count < num_iterations:
        # 采样新点
        s_candidate = random.uniform(s_min, s_max)
        
        # 计算候选点的上限 (UB)
        # UB(s) = min ( f(Xi) + k * ||s - Xi|| ) [cite: 92]
        upper_bound_at_candidate = float('inf')
        
        valid_history = False # 标记是否存在有效的历史数据用于计算UB
        
        for i in range(len(evaluated_s_points)):
            g_val = evaluated_g_values[i]
            if g_val == float('-inf'):
                continue
            
            valid_history = True
            # 一维距离 abs(s - s_i)
            ub_i = g_val + L * abs(s_candidate - evaluated_s_points[i])
            
            if ub_i < upper_bound_at_candidate:
                upper_bound_at_candidate = ub_i
        
        # 决策规则：只有当 UB >= 当前最大值时才评估 
        # LIPO 核心逻辑：只评估有潜力的点
        if valid_history and upper_bound_at_candidate >= max_g_value:
            # 满足条件，执行评估
            g_candidate_raw = solve_sdp(I, m, c, r, b, y, s_candidate, lambda_val)
            
            if isinstance(g_candidate_raw, (float, int)):
                g_candidate = float(g_candidate_raw)
            else:
                g_candidate = float('-inf')
            
            evaluated_s_points.append(s_candidate)
            evaluated_g_values.append(g_candidate)
            
            current_eval_count += 1
            
            # 更新最佳值
            if g_candidate > max_g_value:
                max_g_value = g_candidate
                best_s = s_candidate
            
            # 重置拒绝计数器，因为我们找到了一个有效点
            consecutive_rejections = 0
            
        else:
            # 不满足 Lipschitz 条件，跳过评估（Rejection）
            consecutive_rejections += 1
            
            # --- 核心修改：防止卡死 ---
            # 如果连续很多次都找不到比当前最大值更有潜力的点，说明已经收敛或空间极小
            if consecutive_rejections > max_rejections_allowed:
                # print(f"Converged early after {current_eval_count} evaluations.")
                break

    # 4. 输出 (LIPO Step 3: Output [cite: 98])
    return best_s, max_g_value


def optimize_y(I, m, c, r, b, lambda_val, lipo_iters):
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
            _, max_g = optimize_s(I, m, c, r, b, y_val, lambda_val, lipo_iters)
            return max_g 
        y_min, y_max = 0, b
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
    print(optimize_y(I, m, c, r, b, lambda_val,30))