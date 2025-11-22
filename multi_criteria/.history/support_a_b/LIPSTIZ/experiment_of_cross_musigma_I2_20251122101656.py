import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  
from Zi_y import Z1_y, Z2_y, Z3_y
from single_criteria import compute_Z1_values, compute_Z2_values, compute_Z3_values


def run_single_musigma(sigma,mu,r, I, c,a,b):
    m = np.array([1, mu, mu**2+sigma])
    try:
        Z1_star, y1 = compute_Z1_values(I, m, c, r, a,b)
        Z2_star, y2 = compute_Z2_values(I, m, c, r,a,b)
        Z3_star, y3 = compute_Z3_values(I, m, c, r,a,b)

        Z1_y2 = Z1_y(y2, I, m, c, r,a, b)
        Z1_y3 = Z1_y(y3, I, m, c, r, a,b)
        Z2_y1 = Z2_y(y1, I, m, c, r,a, b)
        Z2_y3 = Z2_y(y3, I, m, c, r,a, b)
        Z3_y1 = Z3_y(y1, I, m, c, r, a,b)
        Z3_y2 = Z3_y(y2, I, m, c, r, a,b)

        return {
            'c': c,
            'r': r,
            'mu': mu,
            'sigma':sigma,
            'a':a,
            'b': b,
            'I': I,
            'sigma/': sigma / mu,
            'y1': y1,
            'y2': y2,
            'y3': y3,
            'Z1_star': Z1_star,
            'Z2_star': Z2_star,
            'Z3_star': Z3_star,
            'Z2_y1': Z2_y1,
            'Z3_y1': Z3_y1,
            'Z1_y2': Z1_y2,
            'Z3_y2': Z3_y2,
            'Z1_y3': Z1_y3,
            'Z2_y3': Z2_y3,
        }

    except Exception as e:
        print(f"⚠️ Error at μ={mu:.3f}: {str(e)}")
        return None



def run_parallel_experiment_musigma(sigma_values, r,mu,I, c,a, b, max_workers=None):
    """
    使用多进程并行执行 μ/b 实验
    max_workers: 默认 None 表示使用所有可用 CPU 核心
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_musigma, sigma, mu, r, I, c,a, b): sigma
            for sigma in sigma_values
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Running μ experiments"):
            res = future.result()
            if res is not None:
                results.append(res)

    return pd.DataFrame(results)


if __name__ == "__main__":
    # 参数设置
    r = 1
    I = 2
    c = 0.1
    b =5
    a=1
    mu=2
    sigma_values =[0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]

    df_mu = run_parallel_experiment_musigma(sigma_values, r,mu,I, c,a,b, max_workers=4)

    df_mu.to_csv('cross_musigma_1.csv', index=False)
    print("✅ 实验完成，结果已保存为 multi_mu_b_experiment.csv")

    r = 1
    I = 2
    c = 0.3
    b =5
    a=1
    mu=2
    sigma_values =[0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]

    df_mu = run_parallel_experiment_musigma(sigma_values, r,mu,I, c,a,b, max_workers=4)

    df_mu.to_csv('cross_musigma_2.csv', index=False)
    print("✅ 实验完成，结果已保存为 multi_mu_b_experiment.csv")

    r = 1
    I = 2
    c = 0.5
    b =5
    a=1
    mu=2
    sigma_values =[0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]

    df_mu = run_parallel_experiment_musigma(sigma_values, r,mu,I, c,a,b, max_workers=4)

    df_mu.to_csv('cross_musigma_3.csv', index=False)
    print("✅ 实验完成，结果已保存为 multi_mu_b_experiment.csv")


    r = 1
    I = 2
    c = 0.7
    b =5
    a=1
    mu=2
    sigma_values =[0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]

    df_mu = run_parallel_experiment_musigma(sigma_values, r,mu,I, c,a,b, max_workers=4)

    df_mu.to_csv('cross_musigma_4.csv', index=False)
    print("✅ 实验完成，结果已保存为 multi_mu_b_experiment.csv")