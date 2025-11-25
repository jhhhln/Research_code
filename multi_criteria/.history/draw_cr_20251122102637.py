import pandas as pd
import matplotlib.pyplot as plt


file_path = "cross_mub_1.csv" 
df = pd.read_csv(file_path)

# df = df.sort_values(by="c/r")
# cr = df["c/r"]
df = df.sort_values(by="mu")
cr = df["mu"]

y_cols = ["y1", "y2", "y3"]
x_cols = ["x1", "x2", "x3"]
star_cols = ["y_star_norm", "y_star_uni"]

# ===== 4. 开始绘图 =====
plt.figure(figsize=(12, 8))

# y1,y2,y3
for col in y_cols:
    plt.plot(cr, df[col], marker='o', linewidth=2, label=col)

# # # x1,x2,x3
for col in x_cols:
    plt.plot(cr, df[col], marker='s', linewidth=2, linestyle='--', label=col)

# y_star_norm, y_star_uni
# for col in star_cols:
#     plt.plot(cr, df[col], linewidth=3, linestyle='-', label=col)

# ===== 5. 图像格式 =====
plt.xlabel("c / r", fontsize=14)
plt.ylabel("Decision Values", fontsize=14)
plt.title("Decision Variables vs c/r", fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# ===== 6. 显示 =====
plt.show()
