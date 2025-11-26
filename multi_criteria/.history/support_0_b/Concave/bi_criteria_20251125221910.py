import numpy as np
from scipy.optimize import minimize_scalar
from single_criteria import compute_Z1_values,compute_Z2_values,compute_Z3_values
from Zi_y import Z1_y,Z2_y,Z3_y
import pandas as pd
import matplotlib.pyplot as plt

def binary_search_root(f, a, b, tol=1e-3, max_iter=100):
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

#寻找约束条件的可行边界
def find_feasible_bounds(I, m, c, r, b, b_val, Z3_star, y_opt_Z3):
    # 定义约束函数
    def constraint_func(y):
        return Z3_y(y, I, m, c, r, b) - b_val * Z3_star

    try:
        y_min = binary_search_root(constraint_func, 0, y_opt_Z3)
    except ValueError:
        raise ValueError("无法找到 y_min, 可能约束无解")

    try:
        y_max = binary_search_root(constraint_func, y_opt_Z3, b)
    except ValueError:
        y_max = b  # 当约束在右半区间始终满足时取最大值
    
    return y_min, y_max

def compute_optimal_y_and_value(I, m, c, r, b, b_val, Z1_star, Z3_star, y_opt_Z3):
    """
    在可行区间内计算最优y和归一化的目标值
    :return: (y_opt, normalized_Z1)
    """
    # 找到可行区间 [y_min, y_max]
    y_min, y_max = find_feasible_bounds(I, m, c, r, b, b_val, Z3_star, y_opt_Z3)
    
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
    
    # 返回最优y和归一化的Z1值
    return y_opt, Z1_opt / Z1_star

def concave_maximization_over_convex_set(I, m, c, r, b, b_val, Z1_star=None, Z3_star=None, y_opt_Z1=None, y_opt_Z3=None):
    """
    凸集上的凹函数最大化算法
    :param b_val: 约束边界值 (0-1)
    :return: (y_opt, normalized_Z1)
    """
    # 如果未提供理想点，则计算它们
    if Z1_star is None or y_opt_Z1 is None:
        Z1_star, y_opt_Z1 = compute_Z1_values(I, m, c, r, b)
    if Z3_star is None or y_opt_Z3 is None:
        Z3_star, y_opt_Z3 = compute_Z3_values(I, m, c, r, b)
    
    # 检查Z1最优点是否满足约束
    Z3_at_Z1_opt = Z3_y(y_opt_Z1, I, m, c, r, b)
    Z3_normalized = Z3_at_Z1_opt / Z3_star
    
    if Z3_normalized >= b_val:
        return y_opt_Z1, Z1_y(y_opt_Z1, I, m, c, r, b) / Z1_star
    
    # 寻找边界解
    return compute_optimal_y_and_value(I, m, c, r, b, b_val, Z1_star, Z3_star, y_opt_Z3)

def generate_pareto_frontier(I, m, c, r, b, num_points=3, filename="pareto_results.csv"):

    # 计算理想点
    Z1_star, y_opt_Z1 = compute_Z1_values(I, m, c, r, b)
    Z3_star, y_opt_Z3 = compute_Z3_values(I, m, c, r, b)
    
    print(f"Z1* = {Z1_star:.4f}, Z3* = {Z3_star:.4f}")
    
    # 创建权衡参数数组 (从0到1)
    alpha_values = np.linspace(0, 1, num_points)
    
    # 创建数据收集列表
    results = []
    
    for alpha in alpha_values:
        # 计算约束边界值
        b_val = alpha
        
        # 求解优化问题
        y_opt, normalized_Z1 = concave_maximization_over_convex_set(
            I, m, c, r, b, b_val, Z1_star, Z3_star, y_opt_Z1, y_opt_Z3)
        
        # 计算实际目标值
        Z1_actual = Z1_y(y_opt, I, m, c, r, b)
        Z3_actual = Z3_y(y_opt, I, m, c, r, b)
        
        # 记录所有相关数据
        results.append({
            'alpha': alpha,
            'y_opt': y_opt,
            'Z1_actual': Z1_actual,
            'Z3_actual': Z3_actual,
            'normalized_Z1': normalized_Z1,
            'Z1_ratio': Z1_actual / Z1_star,
            'Z3_ratio': Z3_actual / Z3_star
        })
    
    # 转换为DataFrame并保存为CSV
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"结果已保存到 {filename}")
    
    # 提取帕累托点用于绘图
    pareto_points = np.array([(row['Z1_actual'], row['Z3_actual']) for row in results])
    
    return df, pareto_points, Z1_star, Z3_star

if __name__ == "__main__":
    # 参数设置
    I = 1
    m = np.array([1, 3])
    c = 1.0
    r = 2
    b = 5
    
    print("开始生成帕累托前沿...")
    
    # 生成帕累托前沿并保存结果
    results_df, pareto_points, Z1_star, Z3_star = generate_pareto_frontier(
        I, m, c, r, b, num_points=20, filename="pareto_results_I1.csv"
    )
    
    # 提取关键点用于绘图
    y_opt_Z1 = compute_Z1_values(I, m, c, r, b)[1]
    y_opt_Z3 = compute_Z3_values(I, m, c, r, b)[1]
    
    Z1_values = pareto_points[:, 0]
    Z3_values = pareto_points[:, 1]
    
    # 绘制帕累托前沿
    plt.figure(figsize=(12, 8))
    
    # 帕累托前沿
    plt.plot(Z3_values, Z1_values, 'b-', linewidth=2, label='Pareto Frontier')
    

    # 参考线
    plt.axhline(y=Z1_star, color='g', linestyle='--', alpha=0.3)
    plt.axvline(x=Z3_star, color='r', linestyle='--', alpha=0.3)
    

    # 图表设置
    plt.xlabel('Z3', fontsize=12)
    plt.ylabel('Z1', fontsize=12)
    plt.title('Pareto Frontier:Z3 vs Z1', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存结果
    plt.savefig('pareto_frontier_I1.png', dpi=300)
    print("帕累托前沿图已保存为 'pareto_frontier.png'")
    plt.show()
    
    # 参数设置
    I = 1
    m = np.array([1, 2])
    c = 1.0
    r = 2
    b = 5
    
    print("开始生成帕累托前沿...")
    
    # 生成帕累托前沿并保存结果
    results_df, pareto_points, Z1_star, Z3_star = generate_pareto_frontier(
        I, m, c, r, b, num_points=20, filename="pareto_results_I1_2.csv"
    )
    
    # 提取关键点用于绘图
    y_opt_Z1 = compute_Z1_values(I, m, c, r, b)[1]
    y_opt_Z3 = compute_Z3_values(I, m, c, r, b)[1]
    
    Z1_values = pareto_points[:, 0]
    Z3_values = pareto_points[:, 1]
    
    # 绘制帕累托前沿
    plt.figure(figsize=(12, 8))
    
    # 帕累托前沿
    plt.plot(Z3_values, Z1_values, 'b-', linewidth=2, label='Pareto Frontier')
    

    # 参考线
    plt.axhline(y=Z1_star, color='g', linestyle='--', alpha=0.3)
    plt.axvline(x=Z3_star, color='r', linestyle='--', alpha=0.3)
    

    # 图表设置
    plt.xlabel('Z3', fontsize=12)
    plt.ylabel('Z1', fontsize=12)
    plt.title('Pareto Frontier:Z3 vs Z1', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存结果
    plt.savefig('pareto_frontier_I1_2.png', dpi=300)
    print("帕累托前沿图已保存为 'pareto_frontier.png'")
    plt.show()
    