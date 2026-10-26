import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import stats
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.stats import rankdata
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Academic font configuration
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==================== 1. Load Lévy residual data ====================
data = np.load("residuals.npz")   # now holds Lévy standardized innovations
eps1 = data["eps1"]   # upstream standardized Lévy innovation
eps2 = data["eps2"]   # downstream standardized Lévy innovation
dt = float(data["dt"])
n = len(eps1)
print(f"Lévy innovation sample size: {n}")
print(f"eps1 range: [{eps1.min():.4f}, {eps1.max():.4f}]")
print(f"eps2 range: [{eps2.min():.4f}, {eps2.max():.4f}]")
print(f"eps1 mean: {eps1.mean():.6f}, variance: {eps1.var():.6f}")
print(f"eps2 mean: {eps2.mean():.6f}, variance: {eps2.var():.6f}")
print(f"Kendall's tau: {stats.kendalltau(eps1, eps2)[0]:.4f}")
print(f"Spearman's rho: {stats.spearmanr(eps1, eps2)[0]:.4f}")

# ==================== 2. Empirical PIT (rank-based pseudo-observations) ====================
# No fitted marginal; use empirical CDF: u_(i) = (rank_i - 0.5) / n
u = (rankdata(eps1) - 0.5) / n
v = (rankdata(eps2) - 0.5) / n
print(f"\nEmpirical pseudo-observation U range: [{u.min():.6f}, {u.max():.6f}]")
print(f"Empirical pseudo-observation V range: [{v.min():.6f}, {v.max():.6f}]")

# ==================== 3. Copula densities and log-likelihoods ====================
# 3.1 Gaussian Copula
def gaussian_copula_pdf(u, v, rho):
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    det = 1 - rho**2
    exponent = -(x**2 - 2*rho*x*y + y**2) / (2*det) + (x**2 + y**2)/2
    return np.exp(exponent) / np.sqrt(det)

def gaussian_copula_loglik(rho, u, v):
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = gaussian_copula_pdf(u, v, rho)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 3.2 Student-t Copula
def student_t_copula_pdf(u, v, rho, nu):
    x = stats.t.ppf(u, nu)
    y = stats.t.ppf(v, nu)
    term1 = gamma((nu+2)/2) * gamma(nu/2) / (gamma((nu+1)/2)**2)
    term2 = 1 / np.sqrt(1 - rho**2)
    term3 = (1 + (x**2 + y**2 - 2*rho*x*y) / (nu*(1-rho**2)))**(-((nu+2)/2))
    term4 = (1 + x**2/nu)**(-(nu+1)/2) * (1 + y**2/nu)**(-(nu+1)/2)
    return term1 * term2 * term3 / term4

def student_t_copula_loglik(params, u, v):
    rho, nu = params
    nu = max(nu, 2.1)
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = student_t_copula_pdf(u, v, rho, nu)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 3.3 Clayton Copula
def clayton_copula_pdf(u, v, theta):
    if theta <= 0:
        return np.zeros_like(u)
    term1 = (1 + theta) * (u*v)**(-1-theta)
    term2 = (u**(-theta) + v**(-theta) - 1)**(-(2+1/theta))
    return term1 * term2

def clayton_copula_loglik(theta, u, v):
    if theta <= 1e-6:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = clayton_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 3.4 Gumbel Copula
def gumbel_copula_pdf(u, v, theta):
    if theta < 1:
        return np.zeros_like(u)
    A = (-np.log(u))**theta + (-np.log(v))**theta
    C = np.exp(-A**(1/theta))
    term1 = C / (u * v)
    term2 = A**(-2 + 2/theta)
    term3 = (np.log(u) * np.log(v))**(theta-1)
    term4 = 1 + (theta - 1) * A**(-1/theta)
    return term1 * term2 * term3 * term4

def gumbel_copula_loglik(theta, u, v):
    if theta < 1:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = gumbel_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# 3.5 Joe Copula
def joe_copula_pdf(u, v, theta):
    if theta < 1:
        return np.zeros_like(u)
    a = 1 - u
    b = 1 - v
    A = a ** (theta - 1)
    B = b ** (theta - 1)
    D = a**theta + b**theta - a**theta * b**theta
    E = D**(1/theta - 2)
    F = D + theta - 1
    return A * B * E * F

def joe_copula_loglik(theta, u, v):
    if theta < 1:
        return 1e10
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    pdf_vals = joe_copula_pdf(u, v, theta)
    pdf_vals = np.clip(pdf_vals, 1e-10, None)
    return -np.sum(np.log(pdf_vals))

# ==================== 4. Tail dependence coefficients ====================
def upper_tail_dependence(copula_type, params):
    if copula_type == 'Gaussian':
        return 0.0
    elif copula_type == 'Student-t':
        rho = params['rho']
        nu = params['nu']
        lambda_u = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return lambda_u
    elif copula_type == 'Clayton':
        return 0.0
    elif copula_type == 'Gumbel':
        theta = params['theta']
        lambda_u = 2 - 2**(1/theta)
        return lambda_u
    elif copula_type == 'Joe':
        theta = params['theta']
        lambda_u = 2 - 2**(1/theta)
        return lambda_u
    return None

def lower_tail_dependence(copula_type, params):
    if copula_type == 'Gaussian':
        return 0.0
    elif copula_type == 'Student-t':
        rho = params['rho']
        nu = params['nu']
        lambda_l = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return lambda_l
    elif copula_type == 'Clayton':
        theta = params['theta']
        if theta > 0:
            lambda_l = 2**(-1/theta)
        else:
            lambda_l = 0.0
        return lambda_l
    elif copula_type == 'Gumbel':
        return 0.0
    elif copula_type == 'Joe':
        return 0.0
    return None

# ==================== 5. Model fitting ====================
results = {}
print("\n" + "="*60)
print("1. Gaussian Copula fit:")
res_gauss = minimize(gaussian_copula_loglik, x0=[0.5], args=(u, v),
                     bounds=[(-0.99, 0.99)], method='L-BFGS-B')
rho_gauss = res_gauss.x[0]
loglik_gauss = -res_gauss.fun
aic_gauss = 2 * 1 - 2 * loglik_gauss
bic_gauss = np.log(n) * 1 - 2 * loglik_gauss
results['Gaussian'] = {
    'params': {'rho': rho_gauss},
    'loglik': loglik_gauss,
    'AIC': aic_gauss,
    'BIC': bic_gauss,
    'n_params': 1
}
print(f"   Parameter: rho = {rho_gauss:.4f}")
print(f"   Log-likelihood: {loglik_gauss:.2f}")
print(f"   AIC: {aic_gauss:.2f}, BIC: {bic_gauss:.2f}")

print("\n2. Student-t Copula fit:")
res_t = minimize(student_t_copula_loglik, x0=[0.5, 5], args=(u, v),
                 bounds=[(-0.99, 0.99), (2.1, 30)], method='L-BFGS-B')
rho_t, nu_t = res_t.x
loglik_t = -res_t.fun
aic_t = 2 * 2 - 2 * loglik_t
bic_t = np.log(n) * 2 - 2 * loglik_t
results['Student-t'] = {
    'params': {'rho': rho_t, 'nu': nu_t},
    'loglik': loglik_t,
    'AIC': aic_t,
    'BIC': bic_t,
    'n_params': 2
}
print(f"   Parameter: rho = {rho_t:.4f}, nu = {nu_t:.4f}")
print(f"   Log-likelihood: {loglik_t:.2f}")
print(f"   AIC: {aic_t:.2f}, BIC: {bic_t:.2f}")

print("\n3. Clayton Copula fit:")
res_clayton = minimize(clayton_copula_loglik, x0=[1.0], args=(u, v),
                       bounds=[(0.01, 20)], method='L-BFGS-B')
theta_clayton = res_clayton.x[0]
loglik_clayton = -res_clayton.fun
aic_clayton = 2 * 1 - 2 * loglik_clayton
bic_clayton = np.log(n) * 1 - 2 * loglik_clayton
results['Clayton'] = {
    'params': {'theta': theta_clayton},
    'loglik': loglik_clayton,
    'AIC': aic_clayton,
    'BIC': bic_clayton,
    'n_params': 1
}
print(f"   Parameter: theta = {theta_clayton:.4f}")
print(f"   Log-likelihood: {loglik_clayton:.2f}")
print(f"   AIC: {aic_clayton:.2f}, BIC: {bic_clayton:.2f}")

print("\n4. Gumbel Copula fit:")
res_gumbel = minimize(gumbel_copula_loglik, x0=[1.5], args=(u, v),
                     bounds=[(1.01, 20)], method='L-BFGS-B')
theta_gumbel = res_gumbel.x[0]
loglik_gumbel = -res_gumbel.fun
aic_gumbel = 2 * 1 - 2 * loglik_gumbel
bic_gumbel = np.log(n) * 1 - 2 * loglik_gumbel
results['Gumbel'] = {
    'params': {'theta': theta_gumbel},
    'loglik': loglik_gumbel,
    'AIC': aic_gumbel,
    'BIC': bic_gumbel,
    'n_params': 1
}
print(f"   Parameter: theta = {theta_gumbel:.4f}")
print(f"   Log-likelihood: {loglik_gumbel:.2f}")
print(f"   AIC: {aic_gumbel:.2f}, BIC: {bic_gumbel:.2f}")

print("\n5. Joe Copula fit:")
res_joe = minimize(joe_copula_loglik, x0=[2.0], args=(u, v),
                   bounds=[(1.01, 20)], method='L-BFGS-B')
theta_joe = res_joe.x[0]
loglik_joe = -res_joe.fun
aic_joe = 2 * 1 - 2 * loglik_joe
bic_joe = np.log(n) * 1 - 2 * loglik_joe
results['Joe'] = {
    'params': {'theta': theta_joe},
    'loglik': loglik_joe,
    'AIC': aic_joe,
    'BIC': bic_joe,
    'n_params': 1
}
print(f"   Parameter: theta = {theta_joe:.4f}")
print(f"   Log-likelihood: {loglik_joe:.2f}")
print(f"   AIC: {aic_joe:.2f}, BIC: {bic_joe:.2f}")

# ==================== 6. Tail dependence table ====================
print("\n" + "="*60)
print("Tail dependence analysis:")
print(f"{'Copula':<12} {'Upper tail λ_U':<15} {'Lower tail λ_L':<15} {'Interpretation'}")
for model in results.keys():
    lambda_u = upper_tail_dependence(model, results[model]['params'])
    lambda_l = lower_tail_dependence(model, results[model]['params'])
    results[model]['upper_tail'] = lambda_u
    results[model]['lower_tail'] = lambda_l
    lu_str = f"{lambda_u:.6f}" if lambda_u is not None else "N/A"
    ll_str = f"{lambda_l:.6f}" if lambda_l is not None else "N/A"
    explanation = ""
    if lambda_u is not None and lambda_l is not None:
        if lambda_u > 0.05 and lambda_l > 0.05:
            explanation = "Both tails"
        elif lambda_u > 0.05:
            explanation = "Upper tail"
        elif lambda_l > 0.05:
            explanation = "Lower tail"
        else:
            explanation = "No tail dependence"
    print(f"    {model:<12} {lu_str:<15} {ll_str:<15} {explanation}")

# ==================== 7. Model comparison ====================
print("\n" + "="*60)
print("Model comparison summary:")
comparison_df = pd.DataFrame({
    'Copula': list(results.keys()),
    'Log-Likelihood': [results[k]['loglik'] for k in results.keys()],
    'AIC': [results[k]['AIC'] for k in results.keys()],
    'BIC': [results[k]['BIC'] for k in results.keys()],
    'Parameters': [results[k]['n_params'] for k in results.keys()],
    'Upper_Tail': [results[k]['upper_tail'] for k in results.keys()],
    'Lower_Tail': [results[k]['lower_tail'] for k in results.keys()]
})
comparison_df = comparison_df.sort_values('AIC')
print(comparison_df.to_string(index=False))
best_model = comparison_df.iloc[0]['Copula']
print(f"\nBest model (by AIC): {best_model}")
print(f"Best model parameters: {results[best_model]['params']}")
if results[best_model]['upper_tail'] is not None:
    print(f"Best model upper-tail λ_U: {results[best_model]['upper_tail']:.6f}")
if results[best_model]['lower_tail'] is not None:
    print(f"Best model lower-tail λ_L: {results[best_model]['lower_tail']:.6f}")

# ==================== 8. Save results ====================
np.savez("copula_fitting_results_pinn_levy.npz",
         u=u, v=v,
         eps1=eps1, eps2=eps2,
         best_copula=best_model,
         gaussian_rho=rho_gauss,
         student_rho=rho_t, student_nu=nu_t,
         clayton_theta=theta_clayton,
         gumbel_theta=theta_gumbel,
         joe_theta=theta_joe,
         dt=dt,
         upstream_dist='empirical', downstream_dist='empirical')
print(f"\nSaved results to: copula_fitting_results_pinn_levy.npz")

# ==================== 9. Visualization (grid density and plots) ====================
u_grid = np.linspace(0.01, 0.99, 100)
v_grid = np.linspace(0.01, 0.99, 100)
U_grid, V_grid = np.meshgrid(u_grid, v_grid)
grid_ravel = (U_grid.ravel(), V_grid.ravel())
Z_gauss = gaussian_copula_pdf(*grid_ravel, rho_gauss).reshape(100, 100)
Z_t = student_t_copula_pdf(*grid_ravel, rho_t, nu_t).reshape(100, 100)
Z_clayton = clayton_copula_pdf(*grid_ravel, theta_clayton).reshape(100, 100)
Z_gumbel = gumbel_copula_pdf(*grid_ravel, theta_gumbel).reshape(100, 100)
Z_joe = joe_copula_pdf(*grid_ravel, theta_joe).reshape(100, 100)

vmin = 0
vmax = 5
Z_gauss = np.clip(Z_gauss, vmin, vmax)
Z_t = np.clip(Z_t, vmin, vmax)
Z_clayton = np.clip(Z_clayton, vmin, vmax)
Z_gumbel = np.clip(Z_gumbel, vmin, vmax)
Z_joe = np.clip(Z_joe, vmin, vmax)
levels = np.linspace(vmin, vmax, 20)
norm = Normalize(vmin=vmin, vmax=vmax)
UNIFIED_SCATTER_COLOR = 'red'
CONTOUR_CMAP = 'turbo'
CONTOUR_LINE_COLOR = 'black'
CONTOUR_LINE_ALPHA = 0.9

fig = plt.figure(figsize=(18, 12))
ax1 = plt.subplot(2, 3, 1)
plt.scatter(u, v, alpha=0.6, s=12, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.2)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title('Pseudo-observations scatter', fontsize=12, fontweight='bold', pad=10, fontfamily='Times New Roman')
plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
ax1.tick_params(width=1.2)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

ax2 = plt.subplot(2, 3, 2)
plt.contourf(U_grid, V_grid, Z_gauss, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_gauss, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Gaussian Copula\nρ={rho_gauss:.3f}\nλ_U={results["Gaussian"]["upper_tail"]:.3f}, λ_L={results["Gaussian"]["lower_tail"]:.3f}',
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax2.tick_params(width=1.2)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

ax3 = plt.subplot(2, 3, 3)
plt.contourf(U_grid, V_grid, Z_t, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_t, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Student-t Copula\nρ={rho_t:.3f}, ν={nu_t:.2f}\nλ_U={results["Student-t"]["upper_tail"]:.3f}, λ_L={results["Student-t"]["lower_tail"]:.3f}',
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax3.tick_params(width=1.2)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontname('Times New Roman')

ax4 = plt.subplot(2, 3, 4)
plt.contourf(U_grid, V_grid, Z_clayton, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_clayton, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Clayton Copula\nθ={theta_clayton:.3f}\nλ_U={results["Clayton"]["upper_tail"]:.3f}, λ_L={results["Clayton"]["lower_tail"]:.3f}',
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax4.tick_params(width=1.2)
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontname('Times New Roman')

ax5 = plt.subplot(2, 3, 5)
plt.contourf(U_grid, V_grid, Z_gumbel, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_gumbel, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Gumbel Copula\nθ={theta_gumbel:.3f}\nλ_U={results["Gumbel"]["upper_tail"]:.3f}, λ_L={results["Gumbel"]["lower_tail"]:.3f}',
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax5.tick_params(width=1.2)
for label in ax5.get_xticklabels() + ax5.get_yticklabels():
    label.set_fontname('Times New Roman')

ax6 = plt.subplot(2, 3, 6)
plt.contourf(U_grid, V_grid, Z_joe, levels=levels, cmap=CONTOUR_CMAP, alpha=0.9, norm=norm)
plt.contour(U_grid, V_grid, Z_joe, levels=levels, colors=CONTOUR_LINE_COLOR, alpha=CONTOUR_LINE_ALPHA, linewidths=0.5)
plt.scatter(u, v, alpha=0.7, s=5, c=UNIFIED_SCATTER_COLOR, edgecolor='black', linewidth=0.1)
plt.xlabel('U', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('V', fontsize=11, fontweight='bold', fontfamily='Times New Roman')
plt.title(f'Joe Copula\nθ={theta_joe:.3f}\nλ_U={results["Joe"]["upper_tail"]:.3f}, λ_L={results["Joe"]["lower_tail"]:.3f}',
          fontsize=11, fontweight='bold', pad=8, fontfamily='Times New Roman')
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.8)
ax6.tick_params(width=1.2)
for label in ax6.get_xticklabels() + ax6.get_yticklabels():
    label.set_fontname('Times New Roman')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(plt.contourf(U_grid, V_grid, Z_gauss, levels=levels, cmap=CONTOUR_CMAP), cax=cbar_ax)
cbar.set_label('Probability density', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
cbar.ax.tick_params(width=1.2, labelsize=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')
cbar.set_ticks(np.arange(vmin, vmax+1, 1))

plt.suptitle('Lévy-noise copula density functions and tail dependence', fontsize=14, fontweight='bold', y=0.99, fontfamily='Times New Roman')
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig('copula_fitting__pinn_levy_contour.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Tail dependence bar chart and map
fig3 = plt.figure(figsize=(12, 6))
ax1 = plt.subplot(1, 2, 1)
models_list = list(results.keys())
upper_tails = [results[m]['upper_tail'] if results[m]['upper_tail'] is not None else 0 for m in models_list]
lower_tails = [results[m]['lower_tail'] if results[m]['lower_tail'] is not None else 0 for m in models_list]
x_pos = np.arange(len(models_list))
width = 0.35
plt.bar(x_pos - width/2, upper_tails, width, label='Upper tail λ_U',
        color='#e74c3c', edgecolor='black', linewidth=1.2)
plt.bar(x_pos + width/2, lower_tails, width, label='Lower tail λ_L',
        color='#3498db', edgecolor='black', linewidth=1.2)
plt.xlabel('Copula model', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Tail dependence coefficient', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
plt.title('Tail dependence coefficients', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
plt.xticks(x_pos, models_list, rotation=45, ha='right', fontfamily='Times New Roman')
plt.legend(fontsize=11, loc='upper left', prop={'family':'Times New Roman', 'size':11})
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Significance threshold')
for i, (u_val, l_val) in enumerate(zip(upper_tails, lower_tails)):
    if u_val > 0.005:
        plt.text(i - width/2, u_val + 0.01, f'{u_val:.3f}', ha='center', fontsize=9, fontweight='bold')
    if l_val > 0.005:
        plt.text(i + width/2, l_val + 0.01, f'{l_val:.3f}', ha='center', fontsize=9, fontweight='bold')
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

ax2 = plt.subplot(1, 2, 2)
colors_scatter = ['#2ecc71' if m == best_model else '#95a5a6' for m in models_list]
sizes = [400 if m == best_model else 200 for m in models_list]
plt.scatter(upper_tails, lower_tails, c=colors_scatter, s=sizes,
            edgecolors='black', linewidth=2.5, alpha=0.8, zorder=3)
for i, model in enumerate(models_list):
    offset_x = 0.008 if model != best_model else 0.015
    offset_y = 0.008 if model != best_model else 0.015
    fontsize = 12 if model == best_model else 10
    fontweight = 'bold' if model == best_model else 'normal'
    plt.annotate(model, (upper_tails[i] + offset_x, lower_tails[i] + offset_y),
                fontsize=fontsize, fontweight=fontweight, fontfamily='Times New Roman')
plt.xlabel('Upper-tail dependence λ_U', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Lower-tail dependence λ_L', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
plt.title('Tail dependence map', fontsize=13, fontweight='bold', fontfamily='Times New Roman')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
plt.axvline(x=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
max_x = max(upper_tails) * 1.1
max_y = max(lower_tails) * 1.1
plt.text(max_x*0.7, max_y*0.9, 'Both-tail region', fontsize=10, alpha=0.6, fontfamily='Times New Roman',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
plt.text(max_x*0.7, max_y*0.1, 'Upper-tail region', fontsize=10, alpha=0.6, fontfamily='Times New Roman',
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
plt.text(max_x*0.1, max_y*0.9, 'Lower-tail region', fontsize=10, alpha=0.6, fontfamily='Times New Roman',
         bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontname('Times New Roman')

plt.suptitle('Lévy-noise copula tail dependence analysis', fontsize=14, fontweight='bold', y=0.98, fontfamily='Times New Roman')
plt.tight_layout()
plt.savefig('copula_tail_dependence_pinn_levy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*60)
print("Analysis completed! Output files:")
print("  - copula_fitting_results_pinn_levy.npz")
print("  - copula_fitting__pinn_levy_contour.png")
print("  - copula_tail_dependence_pinn_levy.png")
