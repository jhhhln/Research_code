import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  
from Zi_y import Z1_y, Z2_y, Z3_y
from single_criteria import compute_Z1_values, compute_Z2_values, compute_Z3_values
from thi_criteria import find_rho_star_and_y
from true_dis import (
    opt_uniform, uniform_expected_profit,
    opt_norm, expected_profit_norm,
    opt_truncnorm, profit_given_y_truncnorm
)


def run_single_experiment(c, r, I, m, b):
    """单个 c 的实验任务（可并行运行）"""
    try:
        rho, multi_y = find_rho_star_and_y(c, r, I, m, b)
    except Exception as e:
        print(f"Error at c={c:.2f}: {e}")
        return {
            'c': c, 'r': r, 'm': m.tolist(), 'b': b, 'I': I,
            'c/r': c / r, 'rho': np.nan, 'multi_y': np.nan,
            'Z1_multi': np.nan, 'Z2_multi': np.nan, 'Z3_multi': np.nan,
            'Z1_star': np.nan, 'Z2_star': np.nan, 'Z3_star': np.nan,
            'y1': np.nan, 'y2': np.nan, 'y3': np.nan
        }

    # 计算多目标结果
    Z1_multi = Z1_y(multi_y, I, m, c, r, b)
    Z2_multi = Z2_y(multi_y, I, m, c, r, b)
    Z3_multi = Z3_y(multi_y, I, m, c, r, b)

    # 单目标最优值
    Z1_star, y1 = compute_Z1_values(I, m, c, r, b)
    Z2_star, y2 = compute_Z2_values(I, m, c, r, b)
    Z3_star, y3 = compute_Z3_values(I, m, c, r, b)


    return {
        'c': c, 'r': r, 'm': m.tolist(), 'b': b, 'I': I,
        'c/r': c / r,
        'rho': rho, 'multi_y': multi_y,
        'Z1_multi': Z1_multi, 'Z2_multi': Z2_multi, 'Z3_multi': Z3_multi,
        'Z1_star': Z1_star, 'Z2_star': Z2_star, 'Z3_star': Z3_star,
        'y1': y1, 'y2': y2, 'y3': y3
    }


def run_numerical_experiment_multi(c_values, r, I, m, b, max_workers=None):
    """多进程执行所有 c 的实验（带 tqdm 进度条）"""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_experiment, c, r, I, m, b) for c in c_values]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running experiments"):
            results.append(future.result())
    results_df = pd.DataFrame(results).sort_values('c')
    return results_df


def compute_single_case(args):
    """计算单个 c 对应的结果"""
    c, r, I, m, b = args
    try:
        # 主目标最优值
        Z1_star, y1 = compute_Z1_values(I, m, c, r, b)
        Z2_star, y2 = compute_Z2_values(I, m, c, r, b)
        Z3_star, y3 = compute_Z3_values(I, m, c, r, b)

        # 交叉评估
        Z1_y2 = Z1_y(y2, I, m, c, r, b)
        Z1_y3 = Z1_y(y3, I, m, c, r, b)
        Z2_y1 = Z2_y(y1, I, m, c, r, b)
        Z2_y3 = Z2_y(y3, I, m, c, r, b)
        Z3_y1 = Z3_y(y1, I, m, c, r, b)
        Z3_y2 = Z3_y(y2, I, m, c, r, b)

        return {
            'c': c, 'r': r, 'I': I, 'm': m, 'b': b,
            'c/r': c / r,
            'y1': y1, 'y2': y2, 'y3': y3,
            'Z1_star': Z1_star, 'Z2_star': Z2_star, 'Z3_star': Z3_star,
            'Z1_y2': Z1_y2, 'Z1_y3': Z1_y3,
            'Z2_y1': Z2_y1, 'Z2_y3': Z2_y3,
            'Z3_y1': Z3_y1, 'Z3_y2': Z3_y2
        }

    except Exception as e:
        print(f"❌ Error at c={c:.3f}: {e}")
        return {
            'c': c, 'r': r, 'I': I, 'm': m, 'b': b,
            'c/r': c / r,
            'y1': np.nan, 'y2': np.nan, 'y3': np.nan,
            'Z1_star': np.nan, 'Z2_star': np.nan, 'Z3_star': np.nan,
            'Z1_y2': np.nan, 'Z1_y3': np.nan,
            'Z2_y1': np.nan, 'Z2_y3': np.nan,
            'Z3_y1': np.nan, 'Z3_y2': np.nan
        }


def run_numerical_experiment_parallel(c_values, r, I, m, b, max_workers=6):
    """多进程并行计算 + tqdm 进度条"""
    args_list = [(c, r, I, m, b) for c in c_values]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_single_case, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running I={I}, m={m}"):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.sort_values(by='c', inplace=True)
    return df

def run_numerical_experiment_parallel_sigma(sigma_values, r, I, m, b, max_workers=6):
    """多进程并行计算 + tqdm 进度条"""
    args_list = [(c, r, I, m, b) for sigma in sigma_values]
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_single_case, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running I={I}, m={m}"):
            results.append(future.result())

    df = pd.DataFrame(results)
    df.sort_values(by='c', inplace=True)
    return df

def run_numerical_experiment_mub(mu_values,r,I,c,b):
    results = []
    for mu in mu_values:
        m=np.array([1,mu])
        try:
            Z1_star,y1=compute_Z1_values(I, m, c, r, b)
            print(y1)
            Z2_star,y2=compute_Z2_values(I, m, c, r, b)
            print(y2)
            Z3_star,y3=compute_Z3_values(I, m, c, r, b)
            print(y3)
        except Exception as e:
            print(f"Error at c={c:.1f}: {str(e)}")
            Z1, Z2, Z3, y1, y2, y3 = (np.nan,)*6
        Z1_y2=Z1_y(y2,I,m,c,r,b)
        Z1_y3=Z1_y(y3,I,m,c,r,b)
        Z2_y1=Z2_y(y1,I,m,c,r,b)
        Z2_y3=Z2_y(y3,I,m,c,r,b)
        Z3_y1=Z3_y(y1,I,m,c,r,b)
        Z3_y2=Z3_y(y2,I,m,c,r,b)


        results.append({
            'c': c,
            'r': r,
            'mu':mu,
            'b':b,
            'I':I,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'Z1_star':Z1_star,
            'Z2_star':Z2_star,
            'Z3_star':Z3_star,
            'Z2_y1':Z2_y1,
            'Z3_y1':Z3_y1,
            'Z1_y2':Z1_y2,
            'Z3_y2':Z3_y2,
            'Z1_y3':Z1_y3,
            'Z2_y3':Z2_y3,
        })
    results_df = pd.DataFrame(results)
    return results_df





# if __name__ == "__main__":
#     # 第一组实验
#     I = 2
#     m = np.array([1, 2.5,9])
#     r = 1
#     b = 5
#     c_values = np.arange(0.05, 0.95, 0.05)

#     df1 = run_numerical_experiment_parallel(c_values, r, I, m, b, max_workers=6)
#     df1.to_csv("cr_vary_result_I2.csv", index=False)
#     print("✅ I=2 实验结果已保存\n")

#     I = 1
#     m = np.array([1, 2.5])
#     r = 1
#     b = 5
#     c_values = np.arange(0.05, 0.95, 0.05)

#     df1 = run_numerical_experiment_parallel(c_values, r, I, m, b, max_workers=6)
#     df1.to_csv("cr_vary_result_I1.csv", index=False)
#     print("✅ I=1 实验结果已保存\n")

if __name__ == "__main__":
    I=2
    sigma_values=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
    r=1
    c=0.3
    b=5
    mu=2.5



# if __name__ == "__main__":
#     # 第一组实验
#     I = 1
#     m = np.array([1, 2.5])
#     r = 1
#     b = 5
#     c_values = np.arange(0.1, 0.5, 0.05)

#     print("\n=== Running experiment 1 (m = [1, 3]) ===")
#     results_df = run_numerical_experiment_multi(c_values, r, I, m, b, max_workers=6)
#     results_df.to_csv('multi_rc_vary_I2.csv', index=False)
#     print("✅ 实验结果 multi_rc_vary_I1.csv 已保存")

#     # 第二组实验
#     I = 1
#     m = np.array([1, 2])
#     r = 1
#     b = 5
#     c_values = np.arange(0.1, 0.4, 0.05)

#     print("\n=== Running experiment 2 (m = [1, 2]) ===")
#     results_df = run_numerical_experiment_multi(c_values, r, I, m, b, max_workers=6)
#     results_df.to_csv('multi_rc_vary_I2.csv', index=False)
#     print("✅ 实验结果 multi_rc_vary_I2.csv 已保存")
