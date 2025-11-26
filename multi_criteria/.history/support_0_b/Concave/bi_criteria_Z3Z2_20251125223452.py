import numpy as np
from scipy.optimize import minimize_scalar
from single_criteria import compute_Z3_values,compute_Z2_values,compute_Z3_values
from Zi_y import Z3_y,Z2_y,Z3_y
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

def compute_optimal_y_and_value(I, m, c, r, b, b_val, Z3_star, Z2_star, y_opt_Z2):

    # 找到可行区间 [y_min, y_max]
    y_min, y_max = find_feasible_bounds(I, m, c, r, b, b_val, Z2_star, y_opt_Z2)
    
    # 定义目标函数
    def neg_Z3_y(y):
        return -Z3_y(y, I, m, c, r, b)
    
    # 在可行区间内寻找最优解
    result = minimize_scalar(neg_Z3_y, bounds=(y_min, y_max), method='bounded')
    y_candidate = result.x
    Z3_candidate = -result.fun
    
    # 计算边界点值
    Z3_y_min = Z3_y(y_min, I, m, c, r, b)
    Z3_y_max = Z3_y(y_max, I, m, c, r, b)
    
    # 选择最优解
    if Z3_y_min > Z3_candidate and Z3_y_min > Z3_y_max:
        y_opt = y_min
        Z3_opt = Z3_y_min
    elif Z3_y_max > Z3_candidate and Z3_y_max > Z3_y_min:
        y_opt = y_max
        Z3_opt = Z3_y_max
    else:
        y_opt = y_candidate
        Z3_opt = Z3_candidate
    
    return y_opt, Z3_opt / Z3_star

def concave_maximization_over_convex_set(I, m, c, r, b, b_val, Z3_star=None, Z2_star=None, y_opt_Z3=None, y_opt_Z2=None):

    # 如果未提供理想点，则计算它们
    if Z3_star is None or y_opt_Z3 is None:
        Z3_star, y_opt_Z3 = compute_Z3_values(I, m, c, r, b)
    if Z2_star is None or y_opt_Z2 is None:
        Z2_star, y_opt_Z2 = compute_Z2_values(I, m, c, r, b)
    
    # 检查Z3最优点是否满足约束
    Z2_at_Z3_opt = Z2_y(y_opt_Z3, I, m, c, r, b)
    Z2_normalized = Z2_star / Z2_at_Z3_opt
    
    if Z2_normalized >= b_val:
        return y_opt_Z3, Z3_y(y_opt_Z3, I, m, c, r, b) / Z3_star
    
    # 寻找边界解
    return compute_optimal_y_and_value(I, m, c, r, b, b_val, Z3_star, Z2_star, y_opt_Z2)

def generate_pareto_frontier(I, m, c, r, b, num_points, filename):
    # 计算理想点
    Z3_star, y_opt_Z3 = compute_Z3_values(I, m, c, r, b)
    Z2_star, y_opt_Z2 = compute_Z2_values(I, m, c, r, b)
    
    print(f"Z3* = {Z3_star:.4f}, Z3* = {Z2_star:.4f}")
    
    # 创建权衡参数数组 (从0到1)
    alpha_values = [i / 100 for i in range(2, 100, 2)]
    
    # 创建CSV文件并写入表头
    fieldnames = ['alpha', 'y_opt', 'Z3_actual', 'Z2_actual', 'normalized_Z3', 'Z3_ratio', 'Z2_ratio']
    
    # 检查文件是否存在，若存在则备份旧文件

    
    # 写入表头
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # 逐个alpha值计算并立即保存结果
    for i, alpha in enumerate(alpha_values):
        try:
            print(f"\n处理进度: {i+1}/{len(alpha_values)} (alpha={alpha})")
            
            # 计算约束边界值
            b_val = alpha
            
            # 求解优化问题
            y_opt, normalized_Z3 = concave_maximization_over_convex_set(
                I, m, c, r, b, b_val, Z3_star, Z2_star, y_opt_Z3, y_opt_Z2)
            
            # 计算实际目标值
            Z3_actual = Z3_y(y_opt, I, m, c, r, b)
            Z2_actual = Z2_y(y_opt, I, m, c, r, b)
            
            # 准备数据行
            row_data = {
                'alpha': alpha,
                'y_opt': str(y_opt.tolist()),  # 将数组转换为字符串
                'Z3_actual': Z3_actual,
                'Z2_actual': Z2_actual,
                'normalized_Z3': normalized_Z3,
                'Z3_ratio': Z3_actual / Z3_star,
                'Z2_ratio': Z2_star / Z2_actual
            }
            
            # 立即追加结果到CSV
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(row_data)
                csvfile.flush()  # 确保数据立即写入磁盘
                print(f"结果已保存: alpha={alpha}")
                
        except Exception as e:
            print(f"\n⚠️ 处理alpha={alpha}时出错: {str(e)}")
            print("已保存之前的结果，跳过当前alpha继续运行...")
    
    # 读取完整结果
    df = pd.read_csv(filename)
    print(f"所有结果已保存到 {filename}")
    
    # 提取帕累托点用于绘图
    pareto_points = np.array([(row['Z3_actual'], row['Z2_actual']) for _, row in df.iterrows()])
    
    return df, pareto_points, Z3_star, Z2_star