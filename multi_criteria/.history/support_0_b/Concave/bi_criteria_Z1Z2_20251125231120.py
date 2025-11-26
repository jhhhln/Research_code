import numpy as np
from scipy.optimize import minimize_scalar
from single_criteria import compute_Z1_values,compute_Z2_values,compute_Z1_values
from Zi_y import Z1_y,Z2_y,Z1_y
import pandas as pd
import matplotlib.pyplot as plt
import csv

def binary_search_root(f, a, b, tol=1e-4, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("f(a) 和 f(b) 必须异号")
    
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b)/2

def find_feasible_bounds(I, m, c, r, b, b_val, Z2_star, y_opt_Z2):
    # 定义约束函数
    def constraint_func(y):
        return b_val*Z2_y(y, I, m, c, r, b) - Z2_star
    
    # 寻找左边界 y_min
    try:
        if constraint_func(0)<=0:
            y_min=0
        else:
            y_min = binary_search_root(constraint_func, 0, y_opt_Z2)
    except ValueError:
        raise ValueError("no y_min,no solution")
    
    # 寻找右边界 y_max
    try:
        if constraint_func(b)<=0:
            y_max=b
        else:
            y_max = binary_search_root(constraint_func, y_opt_Z2, b)
    except ValueError:
        y_max = b  # 当约束在右半区间始终满足时取最大值
    
    return y_min, y_max

def compute_optimal_y_and_value(I, m, c, r, b, b_val, Z1_star, Z2_star, y_opt_Z2):

    # 找到可行区间 [y_min, y_max]
    y_min, y_max = find_feasible_bounds(I, m, c, r, b, b_val, Z2_star, y_opt_Z2)
    
    # 定义目标函数
    def neg_Z1_y(y):
        return -Z1_y(y, I, m, c, r, b)
    
    # 在可行区间内寻找最优解
    result = minimize_scalar(neg_Z1_y, bounds=(y_min, y_max), method='bounded')
    y_candidate = result.x
    Z1_candidate = -result.fun
    
    # 计算边界点值
    Z1_y_min = Z1_y(y_min, I, m, c, r, b)
    Z1_y_max = Z1_y(y_max, I, m, c, r, b)
    
    # 选择最优解
    if Z1_y_min > Z1_candidate and Z1_y_min > Z1_y_max:
        y_opt = y_min
        Z1_opt = Z1_y_min
    elif Z1_y_max > Z1_candidate and Z1_y_max > Z1_y_min:
        y_opt = y_max
        Z1_opt = Z1_y_max
    else:
        y_opt = y_candidate
        Z1_opt = Z1_candidate
    
    return y_opt, Z1_opt / Z1_star

def concave_maximization_over_convex_set(I, m, c, r, b, b_val, Z1_star=None, Z2_star=None, y_opt_Z1=None, y_opt_Z2=None):

    # 如果未提供理想点，则计算它们
    if Z1_star is None or y_opt_Z1 is None:
        Z1_star, y_opt_Z1 = compute_Z1_values(I, m, c, r, b)
    if Z2_star is None or y_opt_Z2 is None:
        Z2_star, y_opt_Z2 = compute_Z2_values(I, m, c, r, b)
    
    # 检查Z1最优点是否满足约束
    Z2_at_Z1_opt = Z2_y(y_opt_Z1, I, m, c, r, b)
    Z2_normalized = Z2_star / Z2_at_Z1_opt
    
    if Z2_normalized >= b_val:
        return y_opt_Z1, Z1_y(y_opt_Z1, I, m, c, r, b) / Z1_star
    
    # 寻找边界解
    return compute_optimal_y_and_value(I, m, c, r, b, b_val, Z1_star, Z2_star, y_opt_Z2)

def generate_pareto_frontier(I, m, c, r, b, num_points, filename):
    from joblib import Parallel, delayed
    import pandas as pd
    import csv
    import numpy as np

    # === Step 1：计算理想点 ===
    Z1_star, y_opt_Z1 = compute_Z1_values(I, m, c, r, b)
    Z2_star, y_opt_Z2 = compute_Z2_values(I, m, c, r, b)

    print(f"Z1* = {Z1_star:.4f}, Z2* = {Z2_star:.4f}")

    # 权重 alpha
    alpha_values = [i / 100 for i in range(2, 100, 2)]

    print("并行计算开始...")

    # === Step 2：封装单个 alpha 的求解函数（不改变内部逻辑） ===
    def solve_one_alpha(alpha):
        try:
            b_val = alpha
            y_opt, normalized_Z1 = concave_maximization_over_convex_set(
                I, m, c, r, b, b_val,
                Z1_star, Z2_star, y_opt_Z1, y_opt_Z2
            )

            Z1_actual = Z1_y(y_opt, I, m, c, r, b)
            Z2_actual = Z2_y(y_opt, I, m, c, r, b)

            return {
                'alpha': alpha,
                'y_opt': y_opt,
                'Z1_actual': Z1_actual,
                'Z2_actual': Z2_actual,
                'normalized_Z1': normalized_Z1,
                'Z1_ratio': Z1_actual / Z1_star,
                'Z2_ratio': Z2_star / Z2_actual
            }
        except Exception as e:
            print(f"⚠️ alpha={alpha} 出错: {e}")
            return None

    # === Step 3：并行化 α-loop（核心加速点） ===
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(solve_one_alpha)(alpha) for alpha in alpha_values
    )

    # 过滤 None（出错的）
    results = [res for res in results if res is not None]

    # === Step 4：保存 CSV (一次写入，不再反复 open/close) ===
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

    print(f"所有结果已保存到 {filename}")

    # === Step 5：处理 Pareto 点 ===
    pareto_points = np.array([(row['Z1_actual'], row['Z2_actual']) for _, row in df.iterrows()])

    return df, pareto_points, Z1_star, Z2_star






# def generate_pareto_frontier(I, m, c, r, b, num_points, filename):
#     # 计算理想点
#     Z1_star, y_opt_Z1 = compute_Z1_values(I, m, c, r, b)
#     Z2_star, y_opt_Z2 = compute_Z2_values(I, m, c, r, b)
    
#     print(f"Z1* = {Z1_star:.4f}, Z1* = {Z2_star:.4f}")
    
#     # 创建权衡参数数组 (从0到1)
#     alpha_values = [i / 100 for i in range(2, 100, 2)]
    
#     # 创建CSV文件并写入表头
#     fieldnames = ['alpha', 'y_opt', 'Z1_actual', 'Z2_actual', 'normalized_Z1', 'Z1_ratio', 'Z2_ratio']
    
#     # 检查文件是否存在，若存在则备份旧文件

    
#     # 写入表头
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
    
#     # 逐个alpha值计算并立即保存结果
#     for i, alpha in enumerate(alpha_values):
#         try:
#             print(f"\n处理进度: {i+1}/{len(alpha_values)} (alpha={alpha})")
            
#             # 计算约束边界值
#             b_val = alpha
            
#             # 求解优化问题
#             y_opt, normalized_Z1 = concave_maximization_over_convex_set(
#                 I, m, c, r, b, b_val, Z1_star, Z2_star, y_opt_Z1, y_opt_Z2)
            
#             # 计算实际目标值
#             Z1_actual = Z1_y(y_opt, I, m, c, r, b)
#             Z2_actual = Z2_y(y_opt, I, m, c, r, b)
            
#             # 准备数据行
#             row_data = {
#                 'alpha': alpha,
#                 'y_opt': str(y_opt.tolist()),  # 将数组转换为字符串
#                 'Z1_actual': Z1_actual,
#                 'Z2_actual': Z2_actual,
#                 'normalized_Z1': normalized_Z1,
#                 'Z1_ratio': Z1_actual / Z1_star,
#                 'Z2_ratio': Z2_star / Z2_actual
#             }
            
#             # 立即追加结果到CSV
#             with open(filename, 'a', newline='') as csvfile:
#                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#                 writer.writerow(row_data)
#                 csvfile.flush()  # 确保数据立即写入磁盘
#                 print(f"结果已保存: alpha={alpha}")
                
#         except Exception as e:
#             print(f"\n⚠️ 处理alpha={alpha}时出错: {str(e)}")
#             print("已保存之前的结果，跳过当前alpha继续运行...")
    
#     # 读取完整结果
#     df = pd.read_csv(filename)
#     print(f"所有结果已保存到 {filename}")
    
#     # 提取帕累托点用于绘图
#     pareto_points = np.array([(row['Z1_actual'], row['Z2_actual']) for _, row in df.iterrows()])
    
#     return df, pareto_points, Z1_star, Z2_star
if __name__ == "__main__":
    # # 参数设置
    I = 1
    m = np.array([1, 3])
    c = 1
    r = 2
    b = 5
    print("Begin")
    
    results_df, pareto_points, Z1_star, Z2_star = generate_pareto_frontier(
        I, m, c, r, b, num_points=2, filename="pareto_results_Z1Z2_I1.csv"
    )
    

    I = 2
    m = np.array([1,3,12])
    c = 1
    r = 2
    b = 5
    
    print("开始生成帕累托前沿...")
    results_df, pareto_points, Z1_star, Z2_star = generate_pareto_frontier(
        I, m, c, r, b, num_points=20, filename="pareto_results_Z1Z2_I2.csv"
    )