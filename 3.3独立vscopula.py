import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, gaussian_kde, rankdata
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

np.random.seed(42)
torch.manual_seed(42)
rng = np.random.default_rng(42)

# 必须和 11.1 训练脚本一致
ALPHA_MIN = 1.05
ALPHA_MAX = 1.99
DT = 1.0

# ============================================================
# 0. Joe Copula（密度公式已修正：θ=1 时退化为 1）
# ============================================================
class JoeCopula:
    def __init__(self, theta):
        self.theta = theta
        if theta < 1:
            raise ValueError("Joe Copula requires theta >= 1")

    def _cond_inv_impl(self, u, w, tol=1e-10, max_iter=100):
        low, high = 0.0, 1.0
        for _ in range(max_iter):
            mid = (low + high) / 2
            num = (1-u)**(self.theta-1) * (1 - (1-mid)**self.theta)
            denom = ((1-u)**self.theta + (1-mid)**self.theta
                     - (1-u)**self.theta*(1-mid)**self.theta)**(1 - 1/self.theta)
            C_uv = num / denom if denom > 0 else 0.0
            if abs(C_uv - w) < tol:
                return mid
            elif C_uv < w:
                low = mid
            else:
                high = mid
        return mid

    def sample(self, n=1, seed=None):
        u = np.random.rand(n)
        w = np.random.rand(n)
        v = np.array([self._cond_inv_impl(ui, wi) for ui, wi in zip(u, w)])
        return u, v

    def pdf(self, u, v):
        th = self.theta
        u = np.clip(np.asarray(u, float), 1e-10, 1-1e-10)
        v = np.clip(np.asarray(v, float), 1e-10, 1-1e-10)
        a = 1.0 - u
        b = 1.0 - v
        D = a**th + b**th - (a**th)*(b**th)
        bracket = (2*th - 1.0)*D + (1.0 - th)
        return (a**(th-1)) * (b**(th-1)) * (D**(1.0/th - 2.0)) * bracket

    def logpdf(self, u, v):
        return np.log(np.clip(self.pdf(u, v), 1e-300, None))


# ============================================================
# 0.5 Lévy NeuralSDE 网络结构（必须和 11.1 完全一致，否则 load_state_dict 报错）
# ============================================================
class NeuralSDE(nn.Module):
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

        target_alpha = 1.55
        p = (target_alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        raw_a0 = np.log(p / (1.0 - p))
        self.alpha_raw = nn.Parameter(
            torch.tensor([raw_a0, raw_a0], dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.zeros(2, dtype=torch.float32))

        final_layer = self.net[-1]
        nn.init.zeros_(final_layer.bias)
        with torch.no_grad():
            final_layer.bias[2:4] = -2.0

    def forward(self, state):
        out = self.net(state)
        drift = out[:, 0:2]
        log_g = torch.clamp(out[:, 2:4], min=-8.0, max=3.0)
        g = torch.exp(log_g)
        alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(self.alpha_raw)
        beta = torch.tanh(self.beta_raw)
        return drift, log_g, g, alpha, beta


# ============================================================
# 1. 加载 Copula 拟合结果 + Lévy 标准化参数 + 原始数据
# ============================================================
# 1.1 Joe Copula 拟合结果
cop_npz = np.load(r"D:\降水与径流量\code\copula_fitting_results_pinn_levy.npz", allow_pickle=True)
eps1 = cop_npz['eps1']
eps2 = cop_npz['eps2']
u_orig = cop_npz['u']
v_orig = cop_npz['v']
joe_theta = float(cop_npz['joe_theta'])
print("Best copula:", str(cop_npz['best_copula']), "theta =", joe_theta)
print("Margins: empirical PIT")

# 经验分位数函数（把 Joe 采样的 (u,v) 映回 Lévy 标准化噪声尺度）
eps1_sorted = np.sort(eps1)
eps2_sorted = np.sort(eps2)
def emp_q1(u):
    u = np.clip(np.asarray(u, float), 1e-10, 1-1e-10)
    idx = np.ceil(u * len(eps1_sorted)).astype(int) - 1
    return eps1_sorted[np.clip(idx, 0, len(eps1_sorted)-1)]
def emp_q2(v):
    v = np.clip(np.asarray(v, float), 1e-10, 1-1e-10)
    idx = np.ceil(v * len(eps2_sorted)).astype(int) - 1
    return eps2_sorted[np.clip(idx, 0, len(eps2_sorted)-1)]

# 1.2 Lévy 训练时保存的标准化参数
sc = np.load(r"D:\降水与径流量\data\standardization_params.npz")
mean_train = sc['mean_train']
std_train = sc['std_train']
print(f"Loaded standardization: mean={mean_train}, std={std_train}")

# 1.3 原始数据 → Lévy 标准化尺度 → 真实增量
df = pd.read_csv(r"D:\降水与径流量\data\dataset.csv", parse_dates=['date'])
raw = df[['discharge_upstream', 'discharge_downstream']].to_numpy(dtype=np.float64)
scaled = (raw - mean_train) / std_train
dX_true = np.diff(scaled[:, 0])
dY_true = np.diff(scaled[:, 1])
T = len(dX_true)


# ============================================================
# 2. Copula 拟合优度（copula-only LRT）
# ============================================================
joe_cop = JoeCopula(theta=joe_theta)

# ============================================================
# 3. 模拟噪声 + Lévy SDE 轨迹传播
# ============================================================
N = len(eps1)

# 3.1 独立噪声：对经验残差各自 bootstrap（天然无相关）
eps1_ind = rng.choice(eps1, size=N, replace=True)
eps2_ind = rng.choice(eps2, size=N, replace=True)

# 3.2 Copula 噪声：Joe 采 (u,v) → 经验分位数映回残差尺度
u_sim, v_sim = joe_cop.sample(n=N, seed=42)
eps1_cop = emp_q1(u_sim)
eps2_cop = emp_q2(v_sim)

# 3.3 加载训练好的 Lévy NeuralSDE
model = NeuralSDE(width=64, depth=3)
model.load_state_dict(torch.load(r"D:\降水与径流量\data\best_neural_sde_model.pth", map_location='cpu'))
model.eval()

# 取出训练好的稳定指数 α₁, α₂
with torch.no_grad():
    alpha_fit = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN)
                 * torch.sigmoid(model.alpha_raw)).numpy()
print(f"\nLoaded Lévy model. alpha1={alpha_fit[0]:.4f}, alpha2={alpha_fit[1]:.4f}")

state_start = np.array([scaled[0, 0], scaled[0, 1]], dtype=np.float32)

def independent_sampler():
    return rng.choice(eps1), rng.choice(eps2)

def copula_sampler():
    us, vs = joe_cop.sample(1)
    return float(emp_q1(us)[0]), float(emp_q2(vs)[0])

def simulate_levy(start_state, sampler, T, dt):
    """Euler 离散 Lévy SDE:
        dZ_i = f_i(Z) dt + g_i(Z) dt^(1/α_i) · ε_i
    其中 ε_i 是已经标准化的 Lévy 新息（均值0、方差≈1）。
    """
    Z_hist = [start_state.copy()]
    Zt = torch.tensor([start_state], dtype=torch.float32)
    for _ in range(T):
        e1, e2 = sampler()
        with torch.no_grad():
            drift, log_g, g, alpha, beta = model(Zt)
        drift = drift.numpy()[0]      # (2,)
        g = g.numpy()[0]             # (2,)
        a = alpha.numpy()            # (2,)
        inc = drift * dt + np.array([
            g[0] * dt ** (1.0 / a[0]) * e1,
            g[1] * dt ** (1.0 / a[1]) * e2
        ])
        nxt = Z_hist[-1] + inc
        Z_hist.append(nxt.copy())
        Zt = torch.tensor([nxt], dtype=torch.float32)
    Z_hist = np.array(Z_hist)
    return np.diff(Z_hist[:, 0]), np.diff(Z_hist[:, 1])

dX_ind, dY_ind = simulate_levy(state_start, independent_sampler, T, DT)
dX_cop, dY_cop = simulate_levy(state_start, copula_sampler, T, DT)

# 绘图用的统一 U,V（独立噪声用秩变换，Copula 噪声直接用采样值）
u_ind = (rankdata(eps1_ind) - 0.5) / N
v_ind = (rankdata(eps2_ind) - 0.5) / N
u_cop = u_sim
v_cop = v_sim


# ============================================================
# 4. 指标计算
# ============================================================
def emp_lambda_u(x, y, q=0.95):
    xt = np.quantile(x, q); yt = np.quantile(y, q)
    return np.mean((x > xt) & (y > yt)) / (1 - q)

def joint_p(x, y, q=0.95):
    return np.mean((x > np.quantile(x, q)) & (y > np.quantile(y, q)))

def edist(X, Y):
    return 2*cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()

print("\n====== Pearson ======")
print(f"Noise:      True={np.corrcoef(eps1,eps2)[0,1]:.4f}, "
      f"Ind={np.corrcoef(eps1_ind,eps2_ind)[0,1]:.4f}, "
      f"Cop={np.corrcoef(eps1_cop,eps2_cop)[0,1]:.4f}")
print(f"Increments: True={np.corrcoef(dX_true,dY_true)[0,1]:.4f}, "
      f"Ind={np.corrcoef(dX_ind,dY_ind)[0,1]:.4f}, "
      f"Cop={np.corrcoef(dX_cop,dY_cop)[0,1]:.4f}")

print("\n====== Kendall tau ======")
t1,_ = stats.kendalltau(eps1, eps2)
t2,_ = stats.kendalltau(eps1_ind, eps2_ind)
t3,_ = stats.kendalltau(eps1_cop, eps2_cop)
t4,_ = stats.kendalltau(dX_true, dY_true)
t5,_ = stats.kendalltau(dX_ind, dY_ind)
t6,_ = stats.kendalltau(dX_cop, dY_cop)
print(f"Noise:      True={t1:.4f}, Ind={t2:.4f}, Cop={t3:.4f}")
print(f"Increments: True={t4:.4f}, Ind={t5:.4f}, Cop={t6:.4f}")

print("\n====== lambda_U (q=0.95) ======")
print(f"Noise:      True={emp_lambda_u(eps1,eps2):.4f}, "
      f"Ind={emp_lambda_u(eps1_ind,eps2_ind):.4f}, "
      f"Cop={emp_lambda_u(eps1_cop,eps2_cop):.4f}")
print(f"Increments: True={emp_lambda_u(dX_true,dY_true):.4f}, "
      f"Ind={emp_lambda_u(dX_ind,dY_ind):.4f}, "
      f"Cop={emp_lambda_u(dX_cop,dY_cop):.4f}")

print("\n====== Joint tail P(>q95,>q95) ======")
print(f"Noise:      True={joint_p(eps1,eps2):.4f}, "
      f"Ind={joint_p(eps1_ind,eps2_ind):.4f}, "
      f"Cop={joint_p(eps1_cop,eps2_cop):.4f}")
print(f"Increments: True={joint_p(dX_true,dY_true):.4f}, "
      f"Ind={joint_p(dX_ind,dY_ind):.4f}, "
      f"Cop={joint_p(dX_cop,dY_cop):.4f}")

print("\n====== Energy distance ======")
ed_ind = edist(np.c_[eps1, eps2], np.c_[eps1_ind, eps2_ind])
ed_cop = edist(np.c_[eps1, eps2], np.c_[eps1_cop, eps2])
ed_inc_ind = edist(np.c_[dX_true, dY_true], np.c_[dX_ind, dY_ind])
ed_inc_cop = edist(np.c_[dX_true, dY_true], np.c_[dX_cop, dY_cop])
print(f"Noise:      Ind={ed_ind:.6f}, Cop={ed_cop:.6f}")
print(f"Increments: Ind={ed_inc_ind:.6f}, Cop={ed_inc_cop:.6f}")


# ============================================================
# 5. 五张图
# ============================================================
# 5.1 噪声散点三联图
xmn = min(eps1.min(), eps1_ind.min(), eps1_cop.min())
xmx = max(eps1.max(), eps1_ind.max(), eps1_cop.max())
ymn = min(eps2.min(), eps2_ind.min(), eps2_cop.min())
ymx = max(eps2.max(), eps2_ind.max(), eps2_cop.max())
ixmn = min(dX_true.min(), dX_ind.min(), dX_cop.min())
ixmx = max(dX_true.max(), dX_ind.max(), dX_cop.max())
iymn = min(dY_true.min(), dY_ind.min(), dY_cop.min())
iymx = max(dY_true.max(), dY_ind.max(), dY_cop.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Standardized Lévy innovation distribution comparison", fontsize=14)
for ax, (x, y, t, c) in zip(axes, [
    (eps1, eps2, "True Lévy noise", "blue"),
    (eps1_ind, eps2_ind, "Independent noise", "green"),
    (eps1_cop, eps2_cop, "Copula noise", "red")]):
    ax.scatter(x, y, s=2, alpha=0.5, color=c)
    ax.set_title(t); ax.set_xlim(xmn, xmx); ax.set_ylim(ymn, ymx)
    ax.set_xlabel(r"$\epsilon_1$")
axes[0].set_ylabel(r"$\epsilon_2$")
plt.tight_layout(); plt.show()

# 5.2 轨迹增量散点三联图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Trajectory increments under Lévy SDE propagation", fontsize=14)
for ax, (x, y, t, c) in zip(axes, [
    (dX_true, dY_true, "True increments", "blue"),
    (dX_ind, dY_ind, "Independent-noise increments", "green"),
    (dX_cop, dY_cop, "Copula-noise increments", "red")]):
    ax.scatter(x, y, s=2, alpha=0.5, color=c)
    ax.set_title(t); ax.set_xlim(ixmn, ixmx); ax.set_ylim(iymn, iymx)
    ax.set_xlabel(r"$\Delta X$")
axes[0].set_ylabel(r"$\Delta Y$")
plt.tight_layout(); plt.show()

# 5.3 2D 联合密度等高线
from matplotlib.lines import Line2D
plt.figure(figsize=(8, 6))
xr = np.linspace(-2, 2, 100)
Xg, Yg = np.meshgrid(xr, xr)
gp = np.vstack([Xg.ravel(), Yg.ravel()])
for data in [(eps1, eps2), (eps1_ind, eps2_ind), (eps1_cop, eps2_cop)]:
    Z = gaussian_kde(np.vstack(data))(gp).reshape(Xg.shape)
    # 先画 contour，不设 label
plt.contour(Xg, Yg, gaussian_kde(np.vstack([eps1, eps2]))(gp).reshape(Xg.shape),
            levels=6, colors='blue', linewidths=1.2, linestyles='-')
plt.contour(Xg, Yg, gaussian_kde(np.vstack([eps1_ind, eps2_ind]))(gp).reshape(Xg.shape),
            levels=6, colors='green', linewidths=1.2, linestyles='--')
plt.contour(Xg, Yg, gaussian_kde(np.vstack([eps1_cop, eps2_cop]))(gp).reshape(Xg.shape),
            levels=6, colors='red', linewidths=1.2, linestyles='-.')
# 用 Line2D 做图例
legend_handles = [
    Line2D([0], [0], color='blue', lw=1.2, ls='-', label='True noise'),
    Line2D([0], [0], color='green', lw=1.2, ls='--', label='Independent noise'),
    Line2D([0], [0], color='red', lw=1.2, ls='-.', label='Copula noise'),
]
plt.legend(handles=legend_handles, loc='upper right')
plt.title("2D joint density contours (Lévy innovations)")
plt.xlabel(r"$\epsilon_1$"); plt.ylabel(r"$\epsilon_2$")
plt.xlim(-2, 2); plt.ylim(-2, 2); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# 5.4 上尾联合事件
thr = 0.95
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Upper-tail joint events (U,V > {thr})", fontsize=14)
for ax, (u, v, t, c) in zip(axes, [
    (u_orig, v_orig, "True", "blue"),
    (u_ind, v_ind, "Independent", "green"),
    (u_cop, v_cop, "Copula", "red")]):
    m = (u > thr) & (v > thr)
    ax.scatter(u[m], v[m], s=10, alpha=0.7, color=c)
    ax.axvline(thr, color="k", ls="--", lw=0.8)
    ax.axhline(thr, color="k", ls="--", lw=0.8)
    ax.set_xlim(thr, 1); ax.set_ylim(thr, 1)
    ax.set_title(f"{t}  (n={m.sum()})")
    ax.set_xlabel("U"); ax.set_ylabel("V"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# 5.5 经验 Copula 密度热力图
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Empirical copula density heatmaps (Lévy)", fontsize=14)
bins = 25
hs = []
for u, v, _ in [(u_orig, v_orig, ""), (u_ind, v_ind, ""), (u_cop, v_cop, "")]:
    h, _, _ = np.histogram2d(u, v, bins=bins, range=[[0, 1], [0, 1]], density=True)
    hs.append(h)
vmax = max(h.max() for h in hs)
for ax, (u, v, t) in zip(axes, [
    (u_orig, v_orig, "True noise"),
    (u_ind, v_ind, "Independent noise"),
    (u_cop, v_cop, "Copula noise")]):
    h, _, _ = np.histogram2d(u, v, bins=bins, range=[[0, 1], [0, 1]], density=True)
    im = ax.imshow(h.T, origin="lower", extent=[0, 1, 0, 1], aspect="equal",
                   cmap="RdBu_r", vmin=0, vmax=vmax)
    ax.set_title(t); ax.set_xlabel("U"); ax.set_ylabel("V")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax)
plt.tight_layout(); plt.show()
