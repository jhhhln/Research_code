import pandas as pd
df = pd.read_csv("cross_mub_4.csv")  
df = df.sort_values(by="mu", ascending=True)
df.to_csv("cross_mub_3.csv", index=False)

df = pd.read_csv("cross_mub_3.csv")
df["Z1_y2_over_Z1_star"] = df["Z1_y2"] / df["Z1_star"]
df["Z1_y3_over_Z1_star"] = df["Z1_y3"] / df["Z1_star"]

df["Z2_star_over_Z2_y1"] = df["Z2_star"] / df["Z2_y1"]
df["Z2_star_over_Z2_y3"] = df["Z2_star"] / df["Z2_y3"]

df["Z3_y1_over_Z3_star"] = df["Z3_y1"] / df["Z3_star"]
df["Z3_y2_over_Z3_star"] = df["Z3_y2"] / df["Z3_star"]
df.to_csv("cross_mub_3.csv", index=False)
print("计算完成，已保存为 data_with_ratios.csv")

