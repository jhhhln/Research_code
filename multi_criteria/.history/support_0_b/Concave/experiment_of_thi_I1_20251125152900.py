



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
