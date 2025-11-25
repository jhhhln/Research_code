import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  
from Zi_y import Z1_y, Z2_y, Z3_y
from single_criteria import compute_Z1_values, compute_Z2_values, compute_Z3_values
from support_a_b.Concave.thi_criteria import find_rho_star_and_y


def run_single_experiment(c, r, I, m, b):
    """单个参数组合的实验任务（可并行运行）"""
    try:
        rho, multi_y = find_rho_star_and_y(c, r, I, m, b)
    except Exception as e:
        print(f"Error at c={c:.2f}, m={m}: {e}")
        return {
            'mu': m[1], 'c': c, 'r': r, 'I': I, 'b': b, 'm': m.tolist(),
            'c/r': c / r, 'rho': np.nan, 'multi_y': np.nan,
            'Z1_multi': np.nan, 'Z2_multi': np.nan, 'Z3_multi': np.nan,
            'Z1_star': np.nan, 'Z2_star': np.nan, 'Z3_star': np.nan,
            'y1': np.nan, 'y2': np.nan, 'y3': np.nan
        }

    # 多目标
    Z1_multi = Z1_y(multi_y, I, m, c, r, b)
    Z2_multi = Z2_y(multi_y, I, m, c, r, b)
    Z3_multi = Z3_y(multi_y, I, m, c, r, b)

    # 单目标
    Z1_star, y1 = compute_Z1_values(I, m, c, r, b)
    Z2_star, y2 = compute_Z2_values(I, m, c, r, b)
    Z3_star, y3 = compute_Z3_values(I, m, c, r, b)

    return {
        'mu': m[1], 'c': c, 'r': r, 'I': I, 'b': b, 'm': m.tolist(),
        'c/r': c / r,
        'rho': rho, 'multi_y': multi_y,
        'Z1_multi': Z1_multi, 'Z2_multi': Z2_multi, 'Z3_multi': Z3_multi,
        'Z1_star': Z1_star, 'Z2_star': Z2_star, 'Z3_star': Z3_star,
        'y1': y1, 'y2': y2, 'y3': y3
    }


def run_numerical_experiment_multi(mu_values, c, r, I, b, max_workers=None):
    """多进程执行所有 μ 的实验（带 tqdm 进度条）"""
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for mu in mu_values:
            m = np.array([1, mu])        # ← 正确生成 μ 对应的 m
            futures.append(
                executor.submit(run_single_experiment, c, r, I, m, b)
            )

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Running experiments"):
            results.append(future.result())

    # μ 递增排序
    results_df = pd.DataFrame(results).sort_values('mu')

    return results_df




if __name__ == "__main__":
    mu_values = np.linspace(1, 20, 20)  
    c = 5
    r = 10
    I = 30
    b = 50

    # 调用函数
    df = run_numerical_experiment_multi(mu_values, c, r, I, b, max_workers=8)

    # 查看结果
    print(df)

    # 保存
    df.to_csv("experiment_results.csv", index=False)
    print("Saved to experiment_results.csv")

    # 第二组实验
    I = 1
    m = np.array([1, 2])
    r = 1
    b = 5
    c_values = np.arange(0.1, 0.4, 0.05)

    print("\n=== Running experiment 2 (m = [1, 2]) ===")
    results_df = run_numerical_experiment_multi(c_values, r, I, m, b, max_workers=6)
    results_df.to_csv('multi_rc_vary_I2.csv', index=False)
    print("✅ 实验结果 multi_rc_vary_I2.csv 已保存")
