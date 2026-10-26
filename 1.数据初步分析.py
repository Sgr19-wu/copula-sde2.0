import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# ---------------- 学术风全局设置 ----------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'      # 数学符号 Times 风格
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
os.makedirs('images', exist_ok=True)

# ---------------- 数据读取（与你的代码一致） ----------------
df = pd.read_csv(r"D:\降水与径流量\data\dataset.csv", encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"])
X = df["discharge_upstream"].values.astype(float)
Y = df["discharge_downstream"].values.astype(float)

# ==================== 1. 描述统计与正态性检验（原有部分，原样保留） ====================
for name, col in [("上游X_t", "discharge_upstream"), ("下游Y_t", "discharge_downstream")]:
    s = df[col].dropna()
    jb = stats.jarque_bera(s)
    ad = stats.anderson(s, "norm")
    cv5 = dict(zip(ad.significance_level, ad.critical_values))[5]
    print(f"===== {name} =====")
    print(f"样本量={len(s)}  缺失值={df[col].isna().sum()}")
    print(f"均值={s.mean():.4f} 标准差={s.std():.4f} 最小值={s.min():.4f} 最大值={s.max():.4f}")
    print(f"偏度={stats.skew(s, bias=False):.4f}  峰度(超额)={stats.kurtosis(s, fisher=True, bias=False):.4f}  峰度(普通)={stats.kurtosis(s, fisher=False, bias=False):.4f}")
    print(f"JB统计量={jb[0]:.4f}  p值={jb[1]:.3e}  {'拒绝正态' if jb[1] < 0.05 else '不拒绝'}")
    print(f"AD统计量={ad.statistic:.4f}  临界值(5%)={cv5:.4f}  {'拒绝正态' if ad.statistic > cv5 else '不拒绝'}")
    print()

# ==================== 2. 相关系数（更新表 3 用） ====================
r, p_r = stats.pearsonr(X, Y)
rho, p_rho = stats.spearmanr(X, Y)[:2]
tau, p_tau = stats.kendalltau(X, Y)[:2]
print("===== 表3：相关系数（1995-2024） =====")
print(f"Pearson  r   = {r:.4f}  (p={p_r:.3e})")
print(f"Spearman rho = {rho:.4f}  (p={p_rho:.3e})")
print(f"Kendall  tau = {tau:.4f}  (p={p_tau:.3e})")

# ==================== 3. 图1：直方图 + KDE + 正态拟合 ====================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = ['#4C72B0', '#C44E52']          # 学术蓝 / 学术红
titles = ['Upstream $X_t$', 'Downstream $Y_t$']

for ax, s, color, title in zip(axes, [X, Y], colors, titles):
    bins = 80
    ax.hist(s, bins=bins, density=True, color=color, alpha=0.55,
            edgecolor='white', linewidth=0.4, label='Histogram')
    x = np.linspace(s.min(), s.max(), 500)
    kde = gaussian_kde(s)(x)
    ax.plot(x, kde, '-', color='black', lw=1.3, label='KDE')
    ax.plot(x, stats.norm.pdf(x, s.mean(), s.std(ddof=1)), '--',
            color='#2CA02C', lw=1.5, label='Normal fit')
    # 统计量标注框
    skew, kurt = stats.skew(s, bias=False), stats.kurtosis(s, fisher=False, bias=False)
    ax.set_xlabel('Discharge (ft$^3$/s)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    # 学术风：刻度朝内、四周框线
    ax.tick_params(direction='in', top=True, right=True, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('images/fig1.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ==================== 4. 图2：散点图（密度着色 + OLS 拟合线） ====================
fig, ax = plt.subplots(figsize=(7.2, 5.4))
xy = np.vstack([X, Y])
dens = gaussian_kde(xy)(xy)
idx = np.argsort(dens)                    # 按密度排序，低密度点先画
sc = ax.scatter(X[idx], Y[idx], c=dens[idx], s=10, cmap='viridis', alpha=0.65,
                edgecolor='none', rasterized=True)

# OLS 拟合线
coef = np.polyfit(X, Y, 1)
x_line = np.linspace(X.min(), X.max(), 100)
ax.plot(x_line, np.polyval(coef, x_line), '--', color='#D62728', lw=1.6, label='OLS fit')

ax.set_xlabel('Upstream discharge $X_t$ (ft$^3$/s)')
ax.set_ylabel('Downstream discharge $Y_t$ (ft$^3$/s)')
ax.set_title('Scatter Plot of $X_t$ and $Y_t$')

# 相关系数标注
ax.text(0.03, 0.97, f'Pearson $r$ = {r:.4f}\nSpearman $\\rho$ = {rho:.4f}\nKendall $\\tau$ = {tau:.4f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.6'))
ax.legend(loc='lower right', frameon=True, fontsize=10)

cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Kernel density', fontsize=10)

ax.tick_params(direction='in', top=True, right=True, width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()
plt.savefig('images/fig2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n图已保存: images/fig1.png, images/fig2.png")
