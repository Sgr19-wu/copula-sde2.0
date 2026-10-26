import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import johnsonsu, genhyperbolic, chi2, gaussian_kde
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Random seeds
np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 0. Custom Joe Copula class
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
            denom = ( (1-u)**self.theta + (1-mid)**self.theta - (1-u)**self.theta*(1-mid)**self.theta )**(1 - 1/self.theta)
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
        v = np.array([self._cond_inv_impl(u_i, w_i) for u_i, w_i in zip(u, w)])
        return u, v
    def pdf(self, u, v):
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        theta = self.theta
        eps = 1e-12
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        t1 = (1-u)**theta
        t2 = (1-v)**theta
        inner = t1 + t2 - t1*t2
        c = (theta * (t1/(1-u+eps)) * (t2/(1-v+eps)) * (theta - 1 + inner)) / (inner**2)
        return c
    def logpdf(self, u, v):
        return np.log(self.pdf(u, v))

# ============================================================
# 1. Load data and saved results
# ============================================================
# Residuals and copula fitting results
data_npz = np.load("copula_fitting_results_pinn_brownian.npz", allow_pickle=True)
eps1 = data_npz['eps1']
eps2 = data_npz['eps2']
u_orig = data_npz['u']
v_orig = data_npz['v']
dt = float(data_npz['dt'])
best_copula_name = str(data_npz['best_copula'])
joe_theta = float(data_npz['joe_theta'])
upstream_dist = str(data_npz['upstream_dist'])
downstream_dist = str(data_npz['downstream_dist'])
upstream_params = tuple(data_npz['upstream_params'])      # (a,b,loc,scale)
downstream_params = tuple(data_npz['downstream_params']) # (p,a,b,loc,scale)
print("Upstream distribution:", upstream_dist, "parameters:", upstream_params)
print("Downstream distribution:", downstream_dist, "parameters:", downstream_params)
print("Best copula:", best_copula_name, "theta =", joe_theta)
marginal1 = johnsonsu(*upstream_params)
marginal2 = genhyperbolic(*downstream_params)

# Load raw data (for generating the true increment series)
df = pd.read_csv(r"D:\降水与径流量\data\dataset.csv", parse_dates=['date'])
X_raw = df['discharge_upstream'].values.astype(np.float32)
Y_raw = df['discharge_downstream'].values.astype(np.float32)
# Scalers (must match the ones used in training)
scaler_X = joblib.load("scaler_X.pkl")
scaler_Y = joblib.load("scaler_Y.pkl")
X_norm = scaler_X.transform(X_raw.reshape(-1,1)).flatten()
Y_norm = scaler_Y.transform(Y_raw.reshape(-1,1)).flatten()
dX_true = np.diff(X_norm)
dY_true = np.diff(Y_norm)
states_X = X_norm[:-1]
states_Y = Y_norm[:-1]
T = len(dX_true)

# ============================================================
# 3. Simulated noise and trajectory increments
# ============================================================
joe_cop = JoeCopula(theta=joe_theta)
N = len(eps1)
# 3.1 Independent noise
eps1_ind_sim = marginal1.rvs(size=N)
eps2_ind_sim = marginal2.rvs(size=N)
# 3.2 Copula noise
u_sim, v_sim = joe_cop.sample(n=N, seed=42)
eps1_cop_sim = marginal1.ppf(u_sim)
eps2_cop_sim = marginal2.ppf(v_sim)
# 3.3 Load the PINN model and simulate trajectories
class DriftDiffusionNet(nn.Module):
    def __init__(self, hidden_dim=64, depth=3):
        super().__init__()
        layers = []
        input_dim = 2
        for i in range(depth):
            layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        out = self.net(inp)
        mu1, mu2, log_sigma1, log_sigma2 = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        sigma1 = torch.exp(log_sigma1) + 1e-6
        sigma2 = torch.exp(log_sigma2) + 1e-6
        return mu1, mu2, sigma1, sigma2

model = DriftDiffusionNet(hidden_dim=64, depth=3)
model.load_state_dict(torch.load("sde_dd_model.pth", map_location='cpu'))
model.eval()

# Initial state
X_start = X_norm[0]
Y_start = Y_norm[0]

def independent_noise_sampler():
    return marginal1.rvs(1)[0], marginal2.rvs(1)[0]

def copula_noise_sampler():
    u_s, v_s = joe_cop.sample(1)
    return marginal1.ppf(u_s[0]), marginal2.ppf(v_s[0])

def simulate_trajectory(model, start_X, start_Y, noise_sampler, T, dt):
    X_hist = [start_X]
    Y_hist = [start_Y]
    Xt = torch.tensor([[start_X]], dtype=torch.float32)
    Yt = torch.tensor([[start_Y]], dtype=torch.float32)
    for _ in range(T):
        eps1, eps2 = noise_sampler()
        with torch.no_grad():
            mu1, mu2, sigma1, sigma2 = model(Xt, Yt)
        dX = mu1.item() * dt + sigma1.item() * np.sqrt(dt) * eps1
        dY = mu2.item() * dt + sigma2.item() * np.sqrt(dt) * eps2
        next_X = Xt.item() + dX
        next_Y = Yt.item() + dY
        X_hist.append(next_X)
        Y_hist.append(next_Y)
        Xt = torch.tensor([[next_X]], dtype=torch.float32)
        Yt = torch.tensor([[next_Y]], dtype=torch.float32)
    return np.array(X_hist), np.array(Y_hist)

X_ind, Y_ind = simulate_trajectory(model, X_start, Y_start, independent_noise_sampler, T, dt)
X_cop, Y_cop = simulate_trajectory(model, X_start, Y_start, copula_noise_sampler, T, dt)
dX_ind = np.diff(X_ind)
dY_ind = np.diff(Y_ind)
dX_cop = np.diff(X_cop)
dY_cop = np.diff(Y_cop)

# Uniform scores for independent and copula noise (used in tail-event and heatmap plots)
u_ind = marginal1.cdf(eps1_ind_sim)
v_ind = marginal2.cdf(eps2_ind_sim)
u_cop = marginal1.cdf(eps1_cop_sim)   # or directly use u_sim
v_cop = marginal2.cdf(eps2_cop_sim)   # or directly use v_sim

# ============================================================
# 4. Pearson, Kendall's tau, and upper-tail dependence lambda_U
# ============================================================
# Empirical upper-tail dependence (quantile threshold q=0.95)
def empirical_upper_tail_dependence(x, y, q=0.95):
    x_th = np.quantile(x, q)
    y_th = np.quantile(y, q)
    p_both = np.mean((x > x_th) & (y > y_th))
    return p_both / (1 - q)

# --- Pearson ---
pearson_true_noise = np.corrcoef(eps1, eps2)[0,1]
pearson_ind_noise = np.corrcoef(eps1_ind_sim, eps2_ind_sim)[0,1]
pearson_cop_noise = np.corrcoef(eps1_cop_sim, eps2_cop_sim)[0,1]
pearson_true_inc = np.corrcoef(dX_true, dY_true)[0,1]
pearson_ind_inc = np.corrcoef(dX_ind, dY_ind)[0,1]
pearson_cop_inc = np.corrcoef(dX_cop, dY_cop)[0,1]
print("\n====== Pearson correlation ======")
print(f"Noise residuals: True={pearson_true_noise:.4f}, Independent={pearson_ind_noise:.4f}, Copula={pearson_cop_noise:.4f}")
print(f"Trajectory increments: True={pearson_true_inc:.4f}, Independent={pearson_ind_inc:.4f}, Copula={pearson_cop_inc:.4f}")

# --- Kendall's tau ---
tau_true_noise, _ = stats.kendalltau(eps1, eps2)
tau_ind_noise, _ = stats.kendalltau(eps1_ind_sim, eps2_ind_sim)
tau_cop_noise, _ = stats.kendalltau(eps1_cop_sim, eps2_cop_sim)
tau_true_inc, _ = stats.kendalltau(dX_true, dY_true)
tau_ind_inc, _ = stats.kendalltau(dX_ind, dY_ind)
tau_cop_inc, _ = stats.kendalltau(dX_cop, dY_cop)
print("\n====== Kendall's tau ======")
print(f"Noise residuals: True={tau_true_noise:.4f}, Independent={tau_ind_noise:.4f}, Copula={tau_cop_noise:.4f}")
print(f"Trajectory increments: True={tau_true_inc:.4f}, Independent={tau_ind_inc:.4f}, Copula={tau_cop_inc:.4f}")

# --- Upper-tail dependence lambda_U ---
utd_true_noise = empirical_upper_tail_dependence(eps1, eps2)
utd_ind_noise = empirical_upper_tail_dependence(eps1_ind_sim, eps2_ind_sim)
utd_cop_noise = empirical_upper_tail_dependence(eps1_cop_sim, eps2_cop_sim)
utd_true_inc = empirical_upper_tail_dependence(dX_true, dY_true)
utd_ind_inc = empirical_upper_tail_dependence(dX_ind, dY_ind)
utd_cop_inc = empirical_upper_tail_dependence(dX_cop, dY_cop)
print("\n====== Upper-tail dependence lambda_U (q=0.95) ======")
print(f"Noise residuals: True={utd_true_noise:.4f}, Independent={utd_ind_noise:.4f}, Copula={utd_cop_noise:.4f}")
print(f"Trajectory increments: True={utd_true_inc:.4f}, Independent={utd_ind_inc:.4f}, Copula={utd_cop_inc:.4f}")

# Joint tail event probability P(eps1 > q95, eps2 > q95)
q = 0.95
# --- Noise residuals ---
q1_true = np.quantile(eps1, q)
q2_true = np.quantile(eps2, q)
p_true_noise = np.mean((eps1 > q1_true) & (eps2 > q2_true))
q1_ind = np.quantile(eps1_ind_sim, q)
q2_ind = np.quantile(eps2_ind_sim, q)
p_ind_noise = np.mean((eps1_ind_sim > q1_ind) & (eps2_ind_sim > q2_ind))
q1_cop = np.quantile(eps1_cop_sim, q)
q2_cop = np.quantile(eps2_cop_sim, q)
p_cop_noise = np.mean((eps1_cop_sim > q1_cop) & (eps2_cop_sim > q2_cop))
# --- Trajectory increments ---
q1_true_inc = np.quantile(dX_true, q)
q2_true_inc = np.quantile(dY_true, q)
p_true_inc = np.mean((dX_true > q1_true_inc) & (dY_true > q2_true_inc))
q1_ind_inc = np.quantile(dX_ind, q)
q2_ind_inc = np.quantile(dY_ind, q)
p_ind_inc = np.mean((dX_ind > q1_ind_inc) & (dY_ind > q2_ind_inc))
q1_cop_inc = np.quantile(dX_cop, q)
q2_cop_inc = np.quantile(dY_cop, q)
p_cop_inc = np.mean((dX_cop > q1_cop_inc) & (dY_cop > q2_cop_inc))
print("\n====== Upper-tail joint event probability P(eps1>q95, eps2>q95) ======")
print(f"Noise residuals: True={p_true_noise:.4f}, Independent={p_ind_noise:.4f}, Copula={p_cop_noise:.4f}")
print(f"Trajectory increments: True={p_true_inc:.4f}, Independent={p_ind_inc:.4f}, Copula={p_cop_inc:.4f}")

# ============================================================
# 5. Energy distance (printed only)
# ============================================================
def energy_distance(X, Y):
    d_xy = cdist(X, Y, 'euclidean').mean()
    d_xx = cdist(X, X, 'euclidean').mean()
    d_yy = cdist(Y, Y, 'euclidean').mean()
    return 2*d_xy - d_xx - d_yy

true_points = np.column_stack([eps1, eps2])
ind_points = np.column_stack([eps1_ind_sim, eps2_ind_sim])
cop_points = np.column_stack([eps1_cop_sim, eps2_cop_sim])
ed_ind = energy_distance(true_points, ind_points)
ed_cop = energy_distance(true_points, cop_points)
true_inc_points = np.column_stack([dX_true, dY_true])
ind_inc_points = np.column_stack([dX_ind, dY_ind])
cop_inc_points = np.column_stack([dX_cop, dY_cop])
ed_inc_ind = energy_distance(true_inc_points, ind_inc_points)
ed_inc_cop = energy_distance(true_inc_points, cop_inc_points)
print("\n====== Energy distance (reference only) ======")
print(f"Noise distribution: Independent={ed_ind:.6f}, Copula={ed_cop:.6f}")
print(f"Increment distribution: Independent={ed_inc_ind:.6f}, Copula={ed_inc_cop:.6f}")

# ============================================================
# 6. Visualization
# ============================================================
# Unified axis limits (noise residuals)
x_min_noise = min(eps1.min(), eps1_ind_sim.min(), eps1_cop_sim.min())
x_max_noise = max(eps1.max(), eps1_ind_sim.max(), eps1_cop_sim.max())
y_min_noise = min(eps2.min(), eps2_ind_sim.min(), eps2_cop_sim.min())
y_max_noise = max(eps2.max(), eps2_ind_sim.max(), eps2_cop_sim.max())
# Unified axis limits (increments)
x_min_inc = min(dX_true.min(), dX_ind.min(), dX_cop.min())
x_max_inc = max(dX_true.max(), dX_ind.max(), dX_cop.max())
y_min_inc = min(dY_true.min(), dY_ind.min(), dY_cop.min())
y_max_inc = max(dY_true.max(), dY_ind.max(), dY_cop.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Standardized residual (noise) distribution comparison", fontsize=14)
axes[0].scatter(eps1, eps2, s=2, alpha=0.5, color='blue')
axes[0].set_title("True noise")
axes[0].set_xlim(x_min_noise, x_max_noise)
axes[0].set_ylim(y_min_noise, y_max_noise)
axes[0].set_xlabel("epsilon_1")
axes[0].set_ylabel("epsilon_2")
axes[1].scatter(eps1_ind_sim, eps2_ind_sim, s=2, alpha=0.5, color='green')
axes[1].set_title("Independent noise simulation")
axes[1].set_xlim(x_min_noise, x_max_noise)
axes[1].set_ylim(y_min_noise, y_max_noise)
axes[1].set_xlabel("epsilon_1")
axes[2].scatter(eps1_cop_sim, eps2_cop_sim, s=2, alpha=0.5, color='red')
axes[2].set_title("Copula noise simulation")
axes[2].set_xlim(x_min_noise, x_max_noise)
axes[2].set_ylim(y_min_noise, y_max_noise)
axes[2].set_xlabel("epsilon_1")
plt.tight_layout()
plt.show()

# Increment scatter plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Trajectory increment (Delta X, Delta Y) distribution comparison", fontsize=14)
axes[0].scatter(dX_true, dY_true, s=2, alpha=0.5, color='blue')
axes[0].set_title("True increments")
axes[0].set_xlim(x_min_inc, x_max_inc)
axes[0].set_ylim(y_min_inc, y_max_inc)
axes[0].set_xlabel("Delta X")
axes[0].set_ylabel("Delta Y")
axes[1].scatter(dX_ind, dY_ind, s=2, alpha=0.5, color='green')
axes[1].set_title("Independent-noise-driven increments")
axes[1].set_xlim(x_min_inc, x_max_inc)
axes[1].set_ylim(y_min_inc, y_max_inc)
axes[1].set_xlabel("Delta X")
axes[2].scatter(dX_cop, dY_cop, s=2, alpha=0.5, color='red')
axes[2].set_title("Copula-noise-driven increments")
axes[2].set_xlim(x_min_inc, x_max_inc)
axes[2].set_ylim(y_min_inc, y_max_inc)
axes[2].set_xlabel("Delta X")
plt.tight_layout()
plt.show()

# 2D joint density contours (noise only, three lines on the same axes)
plt.figure(figsize=(8, 6))
# Fixed display range
x_lim = (-2.0, 2.0)
y_lim = (-2.0, 2.0)
# Grid (fixed range -2 to 2)
x_range = np.linspace(x_lim[0], x_lim[1], 100)
y_range = np.linspace(y_lim[0], y_lim[1], 100)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
grid_points = np.vstack([X_grid.ravel(), Y_grid.ravel()])
# True noise density
kde_true = gaussian_kde(np.vstack([eps1, eps2]))
Z_true = kde_true(grid_points).reshape(X_grid.shape)
# Independent noise density
kde_ind = gaussian_kde(np.vstack([eps1_ind_sim, eps2_ind_sim]))
Z_ind = kde_ind(grid_points).reshape(X_grid.shape)
# Copula noise density
kde_cop = gaussian_kde(np.vstack([eps1_cop_sim, eps2_cop_sim]))
Z_cop = kde_cop(grid_points).reshape(X_grid.shape)
# Contours in three colors
contour_true = plt.contour(X_grid, Y_grid, Z_true, levels=6, colors='blue', linewidths=1.2, linestyles='-')
contour_ind = plt.contour(X_grid, Y_grid, Z_ind, levels=6, colors='green', linewidths=1.2, linestyles='--')
contour_cop = plt.contour(X_grid, Y_grid, Z_cop, levels=6, colors='red', linewidths=1.2, linestyles='-.')
# Legend
h1, _ = contour_true.legend_elements()
h2, _ = contour_ind.legend_elements()
h3, _ = contour_cop.legend_elements()
plt.legend([h1[0], h2[0], h3[0]], ['True noise', 'Independent noise', 'Copula noise'], loc='upper right')
plt.title('2D joint density contours (noise)')
plt.xlabel('epsilon_1')
plt.ylabel('epsilon_2')
plt.xlim(x_lim)
plt.ylim(y_lim)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============ Tail joint-event plot ============
threshold = 0.95
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Upper-tail joint events (U > {threshold}, V > {threshold})', fontsize=14)
datasets = [
    (u_orig, v_orig, 'True noise', 'blue'),
    (u_ind, v_ind, 'Independent noise', 'green'),
    (u_cop, v_cop, 'Copula noise', 'red')
]
for ax, (u, v, title, color) in zip(axes, datasets):
    mask = (u > threshold) & (v > threshold)
    ax.scatter(u[mask], v[mask], s=10, alpha=0.7, color=color)
    # Threshold lines
    ax.axvline(threshold, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(threshold, color='black', linestyle='--', linewidth=0.8)
    ax.set_xlim(threshold, 1.0)
    ax.set_ylim(threshold, 1.0)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_title(f'{title}  (count: {mask.sum()})')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============ Empirical copula heatmap ============
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Empirical copula density heatmaps', fontsize=14)
datasets_hist = [
    (u_orig, v_orig, 'True noise'),
    (u_ind, v_ind, 'Independent noise'),
    (u_cop, v_cop, 'Copula noise')
]
bins = 25
# Pre-compute all histograms to obtain a common density maximum
hists = []
for u, v, _ in datasets_hist:
    hist, _, _ = np.histogram2d(u, v, bins=bins, range=[[0,1],[0,1]], density=True)
    hists.append(hist)
vmax = max(h.max() for h in hists)
vmin = 0.0
for ax, (u, v, title) in zip(axes, datasets_hist):
    hist, xedges, yedges = np.histogram2d(u, v, bins=bins, range=[[0,1],[0,1]], density=True)
    im = ax.imshow(hist.T, origin='lower', extent=[0,1,0,1], aspect='equal',
                   cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Individual colorbars per subplot to avoid layout overlap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax=cax)
plt.tight_layout()
plt.show()
