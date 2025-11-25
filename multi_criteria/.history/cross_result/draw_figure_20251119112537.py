import pandas as pd
import matplotlib.pyplot as plt

file_path = 'cross_mub_4.csv'
df = pd.read_csv(file_path)

for col in ['Z1_y2_over_Z1_star', 'Z3_y2_over_Z3_star']:
    df[col] = df[col].fillna(0)
    df[col] = df[col].clip(lower=0)

# === 3. 按 mu/b 排序 ===
df = df.sort_values(by='mu/b')

# === 4. 绘制图像 ===
plt.figure(figsize=(6, 4.5))

plt.plot(df['mu/b'], df['Z1_y2_over_Z1_star'], marker='o',
         label=r'$\frac{Z_1(y_2,\mathcal{F})}{Z_1^*(\mathcal{F})}$',color='#1f77b4')
plt.plot(df['mu/b'], df['Z3_y2_over_Z3_star'], marker='s',
         label=r'$\frac{Z_3(y_2,\mathcal{F})}{Z_3^*(\mathcal{F})}$',color= '#ff7f0e')

# === 5. 添加图例与标签 ===
plt.xlabel(r'$\frac{\mu}{b}$', fontsize=12)
plt.ylabel(r'Relative Performance', fontsize=12)
plt.title(r'Relative Performance of $y_2$ vs. $\frac{\mu}{b}$ given $\frac{c}{r}=0.5$', fontsize=14)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.savefig('cross4_2.pdf', format='pdf', bbox_inches='tight')
plt.show()



for col in ['Z2_star_over_Z2_y1', 'Z3_y1_over_Z3_star']:
    df[col] = df[col].fillna(0)
    df[col] = df[col].clip(lower=0)

# === 3. 按 mu/b 排序 ===
df = df.sort_values(by='mu/b')

# === 4. 绘制图像 ===
plt.figure(figsize=(6, 4.5))

plt.plot(df['mu/b'], df['Z2_star_over_Z2_y1'], marker='o',
         label=r'$\frac{Z_2^*(\mathcal{F})}{Z_2(y_1,\mathcal{F})}$',color='#1f77b4')
plt.plot(df['mu/b'], df['Z3_y1_over_Z3_star'], marker='s',
         label=r'$\frac{Z_3(y_1,\mathcal{F})}{Z_3^*(\mathcal{F})}$',color= '#ff7f0e')

# === 5. 添加图例与标签 ===
plt.xlabel(r'$\frac{\mu}{b}$', fontsize=12)
plt.ylabel(r'Relative Performance', fontsize=12)
plt.title(r'Relative Performance of $y_1$ vs. $\frac{\mu}{b}$ given $\frac{c}{r}=0.3$', fontsize=14)
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.savefig('cross4_1.pdf', format='pdf', bbox_inches='tight')
plt.show()

for col in ['Z1_y3_over_Z1_star', 'Z2_star_over_Z2_y3']:
    df[col] = df[col].fillna(0)
    df[col] = df[col].clip(lower=0)
# === 3. 按 mu/b 排序 ===
df = df.sort_values(by='mu/b')
# === 4. 绘制图像 ===
plt.figure(figsize=(6, 4.5))
plt.plot(df['mu/b'], df['Z1_y3_over_Z1_star'], marker='o',
         label=r'$\frac{Z_1(y_3,\mathcal{F})}{Z_1^*(\mathcal{F})}$',color='#1f77b4')
plt.plot(df['mu/b'], df['Z2_star_over_Z2_y3'], marker='s',
         label=r'$\frac{Z_2^*(\mathcal{F})}{Z_2(y_3,\mathcal{F})}$',color= '#ff7f0e')   
# === 5. 添加图例与标签 ===
plt.xlabel(r'$\frac{\mu}{b}$', fontsize=12)
plt.ylabel(r'Relative Performance', fontsize=12)
plt.title(r'Relative Performance of $y_3$ vs. $\frac{\mu}{b}$ given $\frac{c}{r}=0.3$', fontsize=14)
plt.legend()    
plt.grid(False)
plt.tight_layout()
plt.savefig('cross4_3.pdf', format='pdf', bbox_inches='tight')
plt.show()
