import pandas as pd

file_path = "cross_mub_4.csv"   
df = pd.read_csv(file_path)

df["min_y2"] = df[["Z1_y2_over_Z1_star", "Z3_y2_over_Z3_star"]].min(axis=1)
df["min_y3"] = df[["Z1_y3_over_Z1_star", "Z2_star_over_Z2_y3"]].min(axis=1)
df["min_y1"] = df[["Z2_star_over_Z2_y1", "Z3_y1_over_Z3_star"]].min(axis=1)

# === 3. 保存结果 ===
output_path = "cross_mub_4.csv"
df.to_csv(output_path, index=False)