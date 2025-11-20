import pandas as pd
import matplotlib.pyplot as plt

file_path = 'cross_mub_4.csv'
df = pd.read_csv(file_path)

for col in ['min_y1', 'min_y2', 'min_y3']:
    df[col] = df[col].fillna(0)
    df[col] = df[col].clip(lower=0)


df = df.sort_values(by='mu/b')

# === 4. 绘制图像 ===
plt.figure(figsize=(6, 4.5))

plt.plot(df['mu/b'], df['min_y1'], marker='^',
         label=r'$Worst Relative Performance of $y_1$', color='#2ca02c')   
plt.plot(df['mu/b'], df['min_y2'], marker='o',
         label=r'$Worst Relative Performance of $y_2$', color='#1f77b4')
plt.plot(df['mu/b'], df['min_y3'], marker='s',  
         label=r'$Worst Relative Performance of $y_3$', color='#ff7f0e')        
  


# === 5. 添加图例与标签 ===
plt.xlabel(r'$\frac{\mu}{b}$', fontsize=12)
plt.ylabel(r'Worst Relative Performance of $y$', fontsize=12)
plt.title(r'Worst Relative Performance of $y$ vs. $\frac{\mu}{b}$ given $\frac{c}{r}=0.5$', fontsize=14)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.savefig('cross4_1.pdf', format='pdf', bbox_inches='tight')
plt.show()