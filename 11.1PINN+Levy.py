import os
import copy
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import levy_stable, kendalltau, spearmanr, kstest


# ============================================================
# 1. Configuration
# ============================================================

DATA_PATH = r"D:\降水与径流量\data\dataset.csv"

DATE_COL = "date"
UP_COL = "discharge_upstream"
DOWN_COL = "discharge_downstream"

SEED = 2026
TRAIN_RATIO = 0.80

BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 5.0

# Numerical Fourier integration settings
FOURIER_T_MAX = 25.0
FOURIER_N_GRID = 513

# Main-model alpha bounds.
# The theoretical stable range is 0 < alpha <= 2.
# Here we use (1.05, 1.99) to avoid the alpha=1 singularity
# in this Fourier formula and the exact alpha=2 boundary.
ALPHA_MIN = 1.05
ALPHA_MAX = 1.99

DT = 1.0  # Daily observations

OUTPUT_DIR = os.path.dirname(DATA_PATH) or "."

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Avoid excessive CPU thread overhead.
torch.set_num_threads(
    max(1, min(8, os.cpu_count() or 1))
)

print("=" * 70)
print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())
print("=" * 70)


# ============================================================
# 2. Load and preprocess data
# ============================================================

df = pd.read_csv(DATA_PATH)

required_columns = [DATE_COL, UP_COL, DOWN_COL]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

df[DATE_COL] = pd.to_datetime(
    df[DATE_COL], errors="coerce"
)

df = (
    df.dropna(subset=required_columns)
      .sort_values(DATE_COL)
      .drop_duplicates(subset=DATE_COL, keep="last")
      .reset_index(drop=True)
)

if len(df) < 100:
    raise ValueError(
        f"Too few valid observations: {len(df)}"
    )

# Flow observations in original units.
raw = df[[UP_COL, DOWN_COL]].to_numpy(
    dtype=np.float64
)

n_obs = len(raw)
n_increments = n_obs - 1

n_train = int(TRAIN_RATIO * n_increments)

# Fit standardization parameters using training-period
# observations only. No validation leakage.
training_observations = raw[:n_train + 1]

mean_train = training_observations.mean(axis=0)
std_train = training_observations.std(
    axis=0, ddof=0
)

std_train[std_train < 1e-12] = 1.0

scaled = (raw - mean_train) / std_train

# Euler increments and left-endpoint states.
states_all = scaled[:-1]
increments_all = scaled[1:] - scaled[:-1]

X_train = states_all[:n_train]
dZ_train = increments_all[:n_train]

X_val = states_all[n_train:]
dZ_val = increments_all[n_train:]

print("\nData summary")
print("Observations:", n_obs)
print("Increments:", n_increments)
print("Training increments:", len(X_train))
print("Validation increments:", len(X_val))
print("Training means:", mean_train)
print("Training SDs:", std_train)


def to_tensor(x):
    return torch.as_tensor(
        x, dtype=torch.float32, device=DEVICE
    )


train_dataset = TensorDataset(
    to_tensor(X_train),
    to_tensor(dZ_train)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

X_val_t = to_tensor(X_val)
dZ_val_t = to_tensor(dZ_val)


# ============================================================
# 3. Neural network for f1, f2, g1, g2
# ============================================================

class NeuralSDE(nn.Module):
    """
    Input:
        [X_t, Y_t]

    Network outputs:
        f1, f2, log(g1), log(g2)

    Trainable global parameters:
        alpha1, alpha2, beta1, beta2
    """

    def __init__(self, width=64, depth=3):
        super().__init__()

        layers = [
            nn.Linear(2, width),
            nn.Tanh()
        ]

        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.Tanh()
            ])

        layers.append(nn.Linear(width, 4))

        self.net = nn.Sequential(*layers)

        # Initialize alpha around 1.55.
        target_alpha = 1.55

        p = (
            (target_alpha - ALPHA_MIN)
            / (ALPHA_MAX - ALPHA_MIN)
        )

        raw_alpha_initial = np.log(
            p / (1.0 - p)
        )

        # Separate alpha parameters.
        self.alpha_raw = nn.Parameter(
            torch.tensor(
                [raw_alpha_initial, raw_alpha_initial],
                dtype=torch.float32
            )
        )

        # Separate beta parameters.
        self.beta_raw = nn.Parameter(
            torch.zeros(2, dtype=torch.float32)
        )

        # Initialize the network's scale outputs.
        final_layer = self.net[-1]

        nn.init.zeros_(final_layer.bias)

        with torch.no_grad():
            final_layer.bias[2:4] = -2.0

    def forward(self, state):

        output = self.net(state)

        drift = output[:, 0:2]

        # Restrict the numerical range of log(g).
        log_g = torch.clamp(
            output[:, 2:4],
            min=-8.0,
            max=3.0
        )

        # Positive state-dependent scales.
        g = torch.exp(log_g)

        # alpha_i in (ALPHA_MIN, ALPHA_MAX).
        alpha = (
            ALPHA_MIN
            + (ALPHA_MAX - ALPHA_MIN)
            * torch.sigmoid(self.alpha_raw)
        )

        # beta_i in (-1, 1).
        beta = torch.tanh(self.beta_raw)

        return drift, log_g, g, alpha, beta


# ============================================================
# 4. Differentiable alpha-stable density
# ============================================================

class StableFourierDensity(nn.Module):
    """
    Approximate the unit-scale alpha-stable density by
    Fourier inversion using the S1 parameterization.

    For alpha != 1:

    p(x) = 1/pi * integral_0^infinity
           exp(-t^alpha)
           cos(x*t - beta*t^alpha*tan(pi*alpha/2)) dt

    IMPORTANT:
    This is a numerical approximation. The implementation
    must be checked against SciPy before trusting estimates.
    """

    def __init__(
        self,
        t_max=FOURIER_T_MAX,
        n_grid=FOURIER_N_GRID
    ):
        super().__init__()

        t = torch.linspace(
            0.0,
            t_max,
            n_grid,
            dtype=torch.float32,
            device=DEVICE
        )

        self.register_buffer("t", t)

    def forward(self, x, alpha, beta):

        # x has shape [batch, 2].
        # Integration grid has shape [1, 1, n_grid].
        t = self.t.view(1, 1, -1)

        a = alpha.view(1, 2, 1)
        b = beta.view(1, 2, 1)

        xx = x.unsqueeze(-1)

        # t^alpha; handle t=0 explicitly.
        t_alpha = torch.pow(
            t.clamp_min(1e-12), a
        )

        t_alpha = torch.where(
            t == 0.0,
            torch.zeros_like(t_alpha),
            t_alpha
        )

        tan_term = torch.tan(
            0.5 * torch.pi * a
        )

        phase = (
            xx * t
            - b * t_alpha * tan_term
        )

        integrand = (
            torch.exp(-t_alpha)
            * torch.cos(phase)
        )

        integral = torch.trapezoid(
            integrand,
            self.t,
            dim=-1
        ) / torch.pi

        # Numerical safeguard only.
        # If many densities are clamped here, the approximation
        # is not adequate and must be improved.
        return integral.clamp_min(1e-12)


# ============================================================
# 5. Marginal stable negative log-likelihood
# ============================================================

def marginal_nll(
    model,
    density,
    state,
    increment,
    dt=DT
):
    """
    Euler model:
        dZ_i = f_i(Z) dt + g_i(Z) dL_i

    Residual:
        r_i = dZ_i - f_i(Z) dt

    Stable increment scale:
        s_i = g_i(Z) * dt^(1/alpha_i)

    Standardized innovation:
        z_i = r_i / s_i
    """

    drift, log_g, g, alpha, beta = model(state)

    residual = increment - drift * dt

    log_dt = torch.log(
        torch.as_tensor(
            dt, dtype=state.dtype, device=state.device
        )
    )

    log_scale = (
        log_g
        + log_dt / alpha.view(1, 2)
    )

    scale = torch.exp(log_scale)

    z = residual / scale

    pdf = density(z, alpha, beta)

    # log p(residual) = log p(z) - log(scale)
    nll_each = -torch.log(pdf) + log_scale

    return nll_each.mean()


@torch.no_grad()
def evaluate_nll(model, density, state, increment):

    model.eval()

    loss = marginal_nll(
        model, density, state, increment
    )

    return float(loss.cpu())


# ============================================================
# 6. Validate Fourier density against SciPy
# ============================================================

def check_fourier_density():

    print("\n" + "=" * 70)
    print("Fourier density check against SciPy")
    print("=" * 70)

    levy_stable.parameterization = "S1"

    test_cases = [
        (1.30, -0.40),
        (1.50, 0.00),
        (1.75, 0.45),
        (1.90, -0.25)
    ]

    x_grid = np.array(
        [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
    )

    density = StableFourierDensity().to(DEVICE)

    errors = []

    for alpha_value, beta_value in test_cases:

        # Repeat values for both margins; compare first column.
        x_tensor = torch.tensor(
            np.column_stack([x_grid, x_grid]),
            dtype=torch.float32,
            device=DEVICE
        )

        alpha_tensor = torch.tensor(
            [alpha_value, alpha_value],
            dtype=torch.float32,
            device=DEVICE
        )

        beta_tensor = torch.tensor(
            [beta_value, beta_value],
            dtype=torch.float32,
            device=DEVICE
        )

        with torch.no_grad():
            p_torch = density(
                x_tensor,
                alpha_tensor,
                beta_tensor
            )[:, 0].cpu().numpy()

        p_scipy = levy_stable.pdf(
            x_grid,
            alpha_value,
            beta_value,
            loc=0.0,
            scale=1.0
        )

        relative_error = (
            np.abs(p_torch - p_scipy)
            / np.maximum(p_scipy, 1e-8)
        )

        max_error = float(relative_error.max())

        errors.append(max_error)

        print(
            f"alpha={alpha_value:.2f}, "
            f"beta={beta_value:+.2f}, "
            f"max relative error={max_error:.4g}"
        )

    if max(errors) > 0.15:

        warnings.warn(
            "The Fourier density differs from SciPy by more "
            "than 15% on the test grid. Treat fitted parameters "
            "as provisional. Improve the integration accuracy "
            "or replace the density implementation before "
            "using the estimates in a paper."
        )

    return errors


fourier_errors = check_fourier_density()


# ============================================================
# 7. Train the neural SDE model
# ============================================================

model = NeuralSDE(
    width=64,
    depth=3
).to(DEVICE)

density = StableFourierDensity().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

best_val_loss = float("inf")
best_state = None
best_epoch = 0
wait = 0
train_loss_history = []
val_loss_history = []
print("\n" + "=" * 70)
print("Training")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    batch_losses = []

    for xb, db in train_loader:

        optimizer.zero_grad(set_to_none=True)

        loss = marginal_nll(
            model,
            density,
            xb,
            db,
            dt=DT
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite loss encountered. Check the "
                "density calculation and parameter constraints."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_losses.append(
            float(loss.detach().cpu())
        )

    train_loss = float(np.mean(batch_losses))

    val_loss = evaluate_nll(
        model,
        density,
        X_val_t,
        dZ_val_t
    )

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)


    with torch.no_grad():

        alpha_current = (
            ALPHA_MIN
            + (ALPHA_MAX - ALPHA_MIN)
            * torch.sigmoid(model.alpha_raw)
        ).cpu().numpy()

        beta_current = torch.tanh(
            model.beta_raw
        ).cpu().numpy()

    if epoch == 1 or epoch % 5 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"train NLL={train_loss:.6f} | "
            f"val NLL={val_loss:.6f} | "
            f"alpha=({alpha_current[0]:.4f}, "
            f"{alpha_current[1]:.4f}) | "
            f"beta=({beta_current[0]:+.4f}, "
            f"{beta_current[1]:+.4f})"
        )

    # Early stopping based on validation NLL.
    if val_loss < best_val_loss - 1e-5:

        best_val_loss = val_loss

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch
        wait = 0

    else:

        wait += 1

        if wait >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}"
            )

            break

if best_state is None:
    raise RuntimeError(
        "No valid model checkpoint was obtained."
    )

model.load_state_dict(best_state)
model.eval()


# ============================================================
# 8. Estimate the proper SDE noise terms
# ============================================================

with torch.no_grad():

    (
        drift_val_t,
        log_g_val_t,
        g_val_t,
        alpha_t,
        beta_t
    ) = model(X_val_t)

    # r_i = Delta Z_i - f_i * dt
    residual_val_t = (
        dZ_val_t - drift_val_t * DT
    )

    # Stable increment scale:
    # g_i * dt^(1 / alpha_i)
    stable_dt_scale = (
        DT ** (1.0 / alpha_t.view(1, 2))
    )

    # Proper estimated Levy increments:
    # Delta L_i = residual_i / g_i
    levy_increment_val_t = (
        residual_val_t / g_val_t
    )

    # Standardized stable innovations:
    # epsilon_i = residual_i /
    #             [g_i * dt^(1 / alpha_i)]
    standardized_noise_val_t = (
        residual_val_t
        / (g_val_t * stable_dt_scale)
    )

alpha_hat = alpha_t.cpu().numpy()
beta_hat = beta_t.cpu().numpy()

drift_val = drift_val_t.cpu().numpy()
g_val = g_val_t.cpu().numpy()
residual_val = residual_val_t.cpu().numpy()

levy_increment_val = (
    levy_increment_val_t.cpu().numpy()
)

standardized_noise_val = (
    standardized_noise_val_t.cpu().numpy()
)

print("\n" + "=" * 70)
print("Estimated parameters")
print("=" * 70)

print(f"Upstream alpha   = {alpha_hat[0]:.6f}")
print(f"Upstream beta    = {beta_hat[0]:.6f}")
print(f"Downstream alpha = {alpha_hat[1]:.6f}")
print(f"Downstream beta  = {beta_hat[1]:.6f}")

print(
    f"Best validation composite marginal NLL "
    f"= {best_val_loss:.6f}"
)

print("\nInterpretation:")
print("  Levy increment = (increment - drift * dt) / g")
print("  Standardized noise = (increment - drift * dt)")
print("                     / (g * dt**(1/alpha))")

# ============================================================
# 9. Save the best model, all-sample noise, and scale parameters
# ============================================================

# 1. Use all observations to estimate the SDE noise
#    states_all: standardized states, length n_obs - 1
#    increments_all: standardized increments, length n_obs - 1

X_all_t = torch.as_tensor(
    states_all, dtype=torch.float32, device=DEVICE
)
dZ_all_t = torch.as_tensor(
    increments_all, dtype=torch.float32, device=DEVICE
)

model.eval()

with torch.no_grad():
    drift_all_t, log_g_all_t, g_all_t, alpha_all_t, beta_all_t = model(X_all_t)

    # Residual increments: dZ - f(Z) * dt
    residual_all_t = dZ_all_t - drift_all_t * DT

    # Estimated Levy increments: Delta L = residual / g
    levy_increment_all_t = residual_all_t / g_all_t

    # Standardized stable innovations:
    # epsilon = residual / (g * dt^(1/alpha))
    stable_dt_scale_all_t = DT ** (1.0 / alpha_all_t.view(1, 2))

    standardized_noise_all_t = (
        residual_all_t / (g_all_t * stable_dt_scale_all_t)
    )

# Convert to NumPy
drift_all = drift_all_t.cpu().numpy()
g_all = g_all_t.cpu().numpy()
residual_all = residual_all_t.cpu().numpy()
levy_increment_all = levy_increment_all_t.cpu().numpy()
standardized_noise_all = standardized_noise_all_t.cpu().numpy()

alpha_all = alpha_all_t.cpu().numpy()
beta_all = beta_all_t.cpu().numpy()

# 2. Save all-sample noise to CSV, including dates
#    Each increment corresponds to the transition date from t to t+1.
noise_dates = df[DATE_COL].iloc[1:].reset_index(drop=True)

noise_df = pd.DataFrame({
    "date": noise_dates,
    "levy_increment_upstream": levy_increment_all[:, 0],
    "levy_increment_downstream": levy_increment_all[:, 1],
    "standardized_noise_upstream": standardized_noise_all[:, 0],
    "standardized_noise_downstream": standardized_noise_all[:, 1],
    "residual_upstream": residual_all[:, 0],
    "residual_downstream": residual_all[:, 1],
    "drift_upstream": drift_all[:, 0],
    "drift_downstream": drift_all[:, 1],
    "diffusion_upstream": g_all[:, 0],
    "diffusion_downstream": g_all[:, 1],
})

noise_csv_path = os.path.join(OUTPUT_DIR, "all_sample_levy_noise.csv")
noise_df.to_csv(noise_csv_path, index=False, encoding="utf-8-sig")

# 3. Save all-sample noise and model parameters to NPZ
noise_npz_path = os.path.join(OUTPUT_DIR, "all_sample_levy_noise.npz")

np.savez(
    noise_npz_path,
    dates=noise_dates.astype(str).to_numpy(),
    levy_increment_upstream=levy_increment_all[:, 0],
    levy_increment_downstream=levy_increment_all[:, 1],
    standardized_noise_upstream=standardized_noise_all[:, 0],
    standardized_noise_downstream=standardized_noise_all[:, 1],
    residual_upstream=residual_all[:, 0],
    residual_downstream=residual_all[:, 1],
    drift_upstream=drift_all[:, 0],
    drift_downstream=drift_all[:, 1],
    diffusion_upstream=g_all[:, 0],
    diffusion_downstream=g_all[:, 1],
    alpha=alpha_all,
    beta=beta_all,
    dt=DT,
    mean_train=mean_train,
    std_train=std_train,
)

# 4. Save the best trained model
model_path = os.path.join(OUTPUT_DIR, "best_neural_sde_model.pth")
torch.save(model.state_dict(), model_path)

# 5. Save standardization parameters for inverse transformation
scaler_params_path = os.path.join(OUTPUT_DIR, "standardization_params.npz")
np.savez(
    scaler_params_path,
    mean_train=mean_train,
    std_train=std_train,
)

# 6. Save the full model configuration and estimated stable parameters
model_params_path = os.path.join(OUTPUT_DIR, "neural_sde_parameters.npz")
np.savez(
    model_params_path,
    alpha=alpha_hat,
    beta=beta_hat,
    best_epoch=best_epoch,
    best_val_loss=best_val_loss,
    dt=DT,
    mean_train=mean_train,
    std_train=std_train,
    train_ratio=TRAIN_RATIO,
)

# 7. Print output locations and basic diagnostics
print("\n" + "=" * 70)
print("All-sample noise and model saved")
print("=" * 70)
print("Noise CSV:", noise_csv_path)
print("Noise NPZ:", noise_npz_path)
print("Model:", model_path)
print("Standardization parameters:", scaler_params_path)
print("Model parameters:", model_params_path)
print("Number of saved noise pairs:", len(standardized_noise_all))

print("\nAll-sample standardized noise statistics:")
print(
    "Upstream: mean={:.4f}, variance={:.4f}".format(
        standardized_noise_all[:, 0].mean(),
        standardized_noise_all[:, 0].var()
    )
)
print(
    "Downstream: mean={:.4f}, variance={:.4f}".format(
        standardized_noise_all[:, 1].mean(),
        standardized_noise_all[:, 1].var()
    )
)

# ============================================================
# 9. Save validation noise and state-dependent estimates
# ============================================================

validation_dates = df[DATE_COL].iloc[
    n_train + 1:
].to_numpy()

results = pd.DataFrame({
    "date": validation_dates,

    "state_upstream_scaled": X_val[:, 0],
    "state_downstream_scaled": X_val[:, 1],

    "increment_upstream_scaled": dZ_val[:, 0],
    "increment_downstream_scaled": dZ_val[:, 1],

    "drift_upstream_hat": drift_val[:, 0],
    "drift_downstream_hat": drift_val[:, 1],

    "g_upstream_hat": g_val[:, 0],
    "g_downstream_hat": g_val[:, 1],

    "residual_upstream": residual_val[:, 0],
    "residual_downstream": residual_val[:, 1],

    "levy_increment_upstream_hat":
        levy_increment_val[:, 0],

    "levy_increment_downstream_hat":
        levy_increment_val[:, 1],

    "standardized_noise_upstream_hat":
        standardized_noise_val[:, 0],

    "standardized_noise_downstream_hat":
        standardized_noise_val[:, 1]
})

noise_path = os.path.join(
    OUTPUT_DIR,
    "neural_sde_levy_noise_validation.csv"
)

results.to_csv(
    noise_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", noise_path)

# ============================================================
# 11. Plot training and validation loss curves
# ============================================================
import matplotlib.pyplot as plt

# Academic-style figure settings.
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
})

epochs = np.arange(1, len(train_loss_history) + 1)
best_val_value = val_loss_history[best_epoch - 1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    epochs, train_loss_history,
    label="Train NLL",
    color="#1f77b4",
    linewidth=1.2
)
ax.plot(
    epochs, val_loss_history,
    label="Validation NLL",
    color="#d62728",
    linewidth=1.2
)

# Mark the early-stopping best epoch.
ax.axvline(
    best_epoch,
    linestyle="--",
    color="gray",
    linewidth=1.0,
    label=f"Best epoch = {best_epoch}"
)
ax.scatter(
    best_epoch, best_val_value,
    color="black", s=20, zorder=5
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Negative log-likelihood")
ax.set_title("Training and validation loss curves")
ax.legend(frameon=True)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "neural_sde_loss_curve.png"
)
fig.savefig(loss_plot_path, bbox_inches="tight")
print("Saved:", loss_plot_path)
plt.show()

# ============================================================
# 10. PIT diagnostics
# ============================================================

# PIT uses the fitted marginal stable CDF.
# The marginal CDF is evaluated using SciPy's S1 parameterization.

levy_stable.parameterization = "S1"

u = np.zeros_like(
    standardized_noise_val,
    dtype=np.float64
)

for j in range(2):

    u[:, j] = levy_stable.cdf(
        standardized_noise_val[:, j],
        alpha_hat[j],
        beta_hat[j],
        loc=0.0,
        scale=1.0
    )

u = np.clip(
    u,
    1e-8,
    1.0 - 1e-8
)

series_names = [
    "upstream",
    "downstream"
]

pit_rows = []

print("\n" + "=" * 70)
print("PIT diagnostics")
print("=" * 70)

for j, name in enumerate(series_names):

    uj = u[:, j]

    ks_result = kstest(
        uj,
        "uniform"
    )

    row = {
        "series": name,
        "alpha_hat": alpha_hat[j],
        "beta_hat": beta_hat[j],

        "noise_mean": np.mean(
            standardized_noise_val[:, j]
        ),

        "noise_sd": np.std(
            standardized_noise_val[:, j],
            ddof=1
        ),

        "PIT_mean": np.mean(uj),

        "PIT_variance": np.var(
            uj, ddof=1
        ),

        "PIT_KS_D": ks_result.statistic,

        "PIT_KS_pvalue": ks_result.pvalue
    }

    pit_rows.append(row)

    print(
        f"{name}: "
        f"PIT mean={row['PIT_mean']:.4f}, "
        f"PIT variance={row['PIT_variance']:.4f}, "
        f"KS D={row['PIT_KS_D']:.4f}, "
        f"KS p={row['PIT_KS_pvalue']:.4g}"
    )

pit_path = os.path.join(
    OUTPUT_DIR,
    "neural_sde_pit_diagnostics.csv"
)

pd.DataFrame(pit_rows).to_csv(
    pit_path,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:", pit_path)


# ============================================================
# 11. Dependence diagnostics on validation PITs
# ============================================================

tau, tau_p = kendalltau(
    u[:, 0], u[:, 1]
)

rho_s, rho_s_p = spearmanr(
    u[:, 0], u[:, 1]
)

print("\n" + "=" * 70)
print("Dependence diagnostics")
print("=" * 70)

print(
    f"Kendall tau = {tau:.6f}, "
    f"p-value = {tau_p:.4g}"
)

print(
    f"Spearman rho = {rho_s:.6f}, "
    f"p-value = {rho_s_p:.4g}"
)


# ============================================================
# 12. Gaussian copula
# ============================================================

def gaussian_copula_nll(theta, uu):

    rho = np.tanh(theta[0])

    x = stats.norm.ppf(uu[:, 0])
    y = stats.norm.ppf(uu[:, 1])

    one_minus_rho2 = max(
        1.0 - rho * rho,
        1e-10
    )

    log_c = (
        -0.5 * np.log(one_minus_rho2)
        - (
            rho * rho * (x*x + y*y)
            - 2.0 * rho * x * y
        ) / (2.0 * one_minus_rho2)
    )

    return -float(np.sum(log_c))


rho_start = np.clip(
    np.sin(np.pi * tau / 2.0),
    -0.95,
    0.95
)

gaussian_result = minimize(
    gaussian_copula_nll,
    x0=np.array([
        np.arctanh(rho_start)
    ]),
    args=(u,),
    method="L-BFGS-B",
    bounds=[(-4.0, 4.0)]
)

rho_gaussian = float(
    np.tanh(gaussian_result.x[0])
)

ll_gaussian = -float(
    gaussian_result.fun
)

aic_gaussian = (
    2.0 * 1 - 2.0 * ll_gaussian
)


# ============================================================
# 13. Student-t copula
# ============================================================

def student_t_copula_nll(theta, uu):

    rho = float(
        np.tanh(theta[0])
    )

    # Enforce degrees of freedom nu > 2.
    nu = float(
        2.0 + np.exp(
            np.clip(theta[1], -3.0, 4.0)
        )
    )

    x = stats.t.ppf(
        uu[:, 0], df=nu
    )

    y = stats.t.ppf(
        uu[:, 1], df=nu
    )

    points = np.column_stack([
        x, y
    ])

    correlation_matrix = np.array([
        [1.0, rho],
        [rho, 1.0]
    ])

    joint_logpdf = stats.multivariate_t.logpdf(
        points,
        loc=np.zeros(2),
        shape=correlation_matrix,
        df=nu
    )

    marginal_logpdf = (
        stats.t.logpdf(x, df=nu)
        + stats.t.logpdf(y, df=nu)
    )

    log_c = (
        joint_logpdf - marginal_logpdf
    )

    if not np.all(np.isfinite(log_c)):
        return 1e100

    return -float(
        np.sum(log_c)
    )


student_t_result = minimize(
    student_t_copula_nll,

    x0=np.array([
        np.arctanh(rho_start),
        np.log(8.0 - 2.0)
    ]),

    args=(u,),

    method="L-BFGS-B",

    bounds=[
        (-4.0, 4.0),
        (-3.0, 4.0)
    ],

    options={
        "maxiter": 500
    }
)

rho_student_t = float(
    np.tanh(student_t_result.x[0])
)

nu_student_t = float(
    2.0 + np.exp(
        np.clip(student_t_result.x[1], -3.0, 4.0)
    )
)

ll_student_t = -float(
    student_t_result.fun
)

aic_student_t = (
    2.0 * 2 - 2.0 * ll_student_t
)


# ============================================================
# 14. Compare copulas and save
# ============================================================

copula_results = pd.DataFrame([
    {
        "copula": "Gaussian",
        "rho": rho_gaussian,
        "nu": np.nan,
        "log_likelihood": ll_gaussian,
        "AIC": aic_gaussian,
        "converged": gaussian_result.success
    },
    {
        "copula": "Student-t",
        "rho": rho_student_t,
        "nu": nu_student_t,
        "log_likelihood": ll_student_t,
        "AIC": aic_student_t,
        "converged": student_t_result.success
    }
])

copula_path = os.path.join(
    OUTPUT_DIR,
    "neural_sde_copula_validation.csv"
)

copula_results.to_csv(
    copula_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("Copula fitting results")
print("=" * 70)

print(
    copula_results.to_string(
        index=False
    )
)

print("\nSaved:", copula_path)


# ============================================================
# 15. Final reminders
# ============================================================

print("\n" + "=" * 70)
print("Run completed")
print("=" * 70)

print("Estimated parameters:")
print(f"  alpha1 = {alpha_hat[0]:.6f}")
print(f"  beta1  = {beta_hat[0]:.6f}")
print(f"  alpha2 = {alpha_hat[1]:.6f}")
print(f"  beta2  = {beta_hat[1]:.6f}")

print("\nInterpretation checklist:")
print("1. Check the Fourier density errors before trusting estimates.")
print("2. Levy increments are residuals divided by g.")
print("3. Standardized innovations also divide by dt**(1/alpha).")
print("4. Copulas are fitted to PIT-transformed validation innovations.")
print("5. Poor PIT calibration weakens the interpretation of copula results.")
print("6. A significant dependence statistic alone does not validate a copula.")
print("7. This model uses a composite marginal likelihood, not a joint")
print("   copula likelihood during neural-network training.")
print("8. The KS p-values do not account for temporal dependence.")







import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 16. Diagnostic suite for the fitted neural SDE
# Append this section AFTER your original Section 15.
# ============================================================

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Use fitted objects from the original script.
levy_stable.parameterization = "S1"

diag_states = np.asarray(X_val, dtype=float)
diag_increments = np.asarray(dZ_val, dtype=float)
diag_f = np.asarray(drift_val, dtype=float)
diag_g = np.asarray(g_val, dtype=float)
diag_eps = np.asarray(standardized_noise_val, dtype=float)
diag_u = np.asarray(u, dtype=float)

diag_log_g = np.log(np.maximum(diag_g, 1e-12))
diag_n = len(diag_increments)

if not (
    diag_states.shape == diag_increments.shape
    == diag_f.shape == diag_g.shape == diag_eps.shape
    == diag_u.shape
):
    raise ValueError(
        "Diagnostic arrays have inconsistent shapes. "
        "Check X_val, dZ_val, drift_val, g_val, "
        "standardized_noise_val and u."
    )

if not np.all(np.isfinite(diag_u)):
    raise ValueError("PIT array contains non-finite values.")

diag_u = np.clip(diag_u, 1e-8, 1 - 1e-8)

names = ["Upstream", "Downstream"]


# ------------------------------------------------------------
# 16.1 PIT calibration: histogram, ECDF, time series
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PIT MARGINAL CALIBRATION")
print("=" * 70)

pit_rows = []

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for j, name in enumerate(names):
    uj = diag_u[:, j]
    ks = stats.kstest(uj, "uniform")

    pit_rows.append({
        "margin": name,
        "alpha": alpha_hat[j],
        "beta": beta_hat[j],
        "PIT_mean": np.mean(uj),
        "PIT_variance": np.var(uj, ddof=1),
        "KS_D": ks.statistic,
        "KS_pvalue_iid_assumption": ks.pvalue
    })

    print(
        f"{name}: mean={np.mean(uj):.4f}, "
        f"variance={np.var(uj, ddof=1):.4f}, "
        f"KS D={ks.statistic:.4f}, "
        f"KS p={ks.pvalue:.4e}"
    )

    axes[j, 0].hist(
        uj, bins=20, range=(0, 1), density=True,
        edgecolor="black", alpha=0.75
    )
    axes[j, 0].axhline(1.0, linestyle="--")
    axes[j, 0].set_title(f"{name}: PIT histogram")
    axes[j, 0].set_xlabel("PIT")
    axes[j, 0].set_ylabel("Density")

    sorted_u = np.sort(uj)
    empirical_cdf = np.arange(1, len(uj) + 1) / len(uj)

    axes[j, 1].plot(sorted_u, empirical_cdf)
    axes[j, 1].plot([0, 1], [0, 1], "--")
    axes[j, 1].set_title(f"{name}: empirical PIT CDF")
    axes[j, 1].set_xlabel("u")
    axes[j, 1].set_ylabel("Empirical CDF")

    axes[j, 2].plot(uj, linewidth=0.7)
    axes[j, 2].axhline(0.5, linestyle="--")
    axes[j, 2].set_ylim(0, 1)
    axes[j, 2].set_title(f"{name}: PIT over time")
    axes[j, 2].set_xlabel("Validation time index")
    axes[j, 2].set_ylabel("PIT")

plt.tight_layout()
plt.show()

pd.DataFrame(pit_rows).to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_pit.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16.2 Stable innovation QQ plots
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for j, name in enumerate(names):
    eps = diag_eps[:, j]
    m = min(len(eps), 1500)
    idx = np.linspace(0, len(eps) - 1, m).astype(int)
    sample_sorted = np.sort(eps[idx])

    probs = (np.arange(1, m + 1) - 0.5) / m
    theoretical = levy_stable.ppf(
        probs, alpha_hat[j], beta_hat[j],
        loc=0, scale=1
    )

    axes[j].scatter(theoretical, sample_sorted, s=8, alpha=0.5)

    bounds = np.concatenate([theoretical, sample_sorted])
    lo, hi = np.nanpercentile(bounds, [2, 98])
    axes[j].plot([lo, hi], [lo, hi], "--")

    axes[j].set_title(f"{name}: stable innovation QQ plot")
    axes[j].set_xlabel("Theoretical quantiles")
    axes[j].set_ylabel("Empirical quantiles")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 16.3 Innovation autocorrelation and Ljung-Box tests
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("INNOVATION TEMPORAL DEPENDENCE")
print("=" * 70)

max_lag = min(30, max(5, diag_n // 10))

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
lb_rows = []

for j, name in enumerate(names):
    eps = diag_eps[:, j]

    plot_acf(
        eps, lags=max_lag, zero=False,
        ax=axes[j, 0]
    )
    axes[j, 0].set_title(f"{name}: innovation ACF")

    # Squared-innovation ACF is an auxiliary diagnostic.
    # For alpha-stable noise with alpha < 2, second moments
    # need not exist, so interpret this plot cautiously.
    plot_acf(
        eps ** 2, lags=max_lag, zero=False,
        ax=axes[j, 1]
    )
    axes[j, 1].set_title(f"{name}: squared-innovation ACF")

    lb_eps = acorr_ljungbox(
        eps, lags=[10, 20], return_df=True
    )
    lb_sq = acorr_ljungbox(
        eps ** 2, lags=[10, 20], return_df=True
    )

    print(f"\n{name}: innovation Ljung-Box")
    print(lb_eps)
    print(f"{name}: squared-innovation Ljung-Box")
    print(lb_sq)

    for lag in [10, 20]:
        lb_rows.append({
            "margin": name,
            "series": "innovation",
            "lag": lag,
            "statistic": lb_eps.loc[lag, "lb_stat"],
            "pvalue_iid_assumption": lb_eps.loc[lag, "lb_pvalue"]
        })
        lb_rows.append({
            "margin": name,
            "series": "squared_innovation",
            "lag": lag,
            "statistic": lb_sq.loc[lag, "lb_stat"],
            "pvalue_iid_assumption": lb_sq.loc[lag, "lb_pvalue"]
        })

plt.tight_layout()
plt.show()

pd.DataFrame(lb_rows).to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_ljung_box.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16.4 State-dependent f and g diagnostics
# ------------------------------------------------------------

def diag_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fg_rows = []
scale_rows = []

for j, name in enumerate(names):
    state = diag_states[:, j]
    fj = diag_f[:, j]
    gj = diag_g[:, j]

    fg_rows.append({
        "margin": name,
        "f_mean": np.mean(fj),
        "f_sd": np.std(fj, ddof=1),
        "g_mean": np.mean(gj),
        "g_sd": np.std(gj, ddof=1),
        "g_cv": np.std(gj, ddof=1) /
                (np.mean(gj) + 1e-12),
        "corr_f_g": diag_corr(fj, gj),
        "corr_state_f": diag_corr(state, fj),
        "corr_state_g": diag_corr(state, gj)
    })

    q = np.quantile(gj, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    scale_rows.append({
        "margin": name,
        "g_min": q[0],
        "g_q01": q[1],
        "g_q05": q[2],
        "g_q25": q[3],
        "g_median": q[4],
        "g_q75": q[5],
        "g_q95": q[6],
        "g_q99": q[7],
        "g_max": q[8]
    })

    axes[j, 0].scatter(state, fj, s=8, alpha=0.3)
    axes[j, 0].set_title(f"{name}: f versus own state")
    axes[j, 0].set_xlabel("Standardized state")
    axes[j, 0].set_ylabel("f")

    axes[j, 1].scatter(state, gj, s=8, alpha=0.3)
    axes[j, 1].set_title(f"{name}: g versus own state")
    axes[j, 1].set_xlabel("Standardized state")
    axes[j, 1].set_ylabel("g")

    axes[j, 2].scatter(fj, gj, s=8, alpha=0.3)
    axes[j, 2].set_title(f"{name}: f versus g")
    axes[j, 2].set_xlabel("f")
    axes[j, 2].set_ylabel("g")

plt.tight_layout()
plt.show()

fg_df = pd.DataFrame(fg_rows)
scale_df = pd.DataFrame(scale_rows)

print("\nState-dependent f/g diagnostics:")
print(fg_df.to_string(index=False))

print("\nScale quantiles:")
print(scale_df.to_string(index=False))

fg_df.to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_fg_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)
scale_df.to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_g_quantiles.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16.5 State-bin scale support
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
bin_rows = []

for j, name in enumerate(names):
    frame = pd.DataFrame({
        "state": diag_states[:, j],
        "g": diag_g[:, j],
        "innovation": diag_eps[:, j]
    })

    frame["bin"] = pd.qcut(
        frame["state"], q=10,
        labels=False, duplicates="drop"
    )

    summary = frame.groupby("bin", observed=True).agg(
        n=("state", "size"),
        state_mean=("state", "mean"),
        g_mean=("g", "mean"),
        g_sd=("g", "std"),
        innovation_mean=("innovation", "mean"),
        innovation_sd=("innovation", "std")
    ).reset_index()

    summary["margin"] = name
    bin_rows.append(summary)

    # Keep all bins here; quantile bins generally have similar counts.
    axes[j, 0].plot(
        summary["state_mean"], summary["g_mean"], marker="o"
    )
    axes[j, 0].set_title(f"{name}: predicted g by state bin")
    axes[j, 0].set_xlabel("Mean standardized state")
    axes[j, 0].set_ylabel("Mean g")

    axes[j, 1].plot(
        summary["state_mean"], summary["innovation_sd"], marker="o"
    )
    axes[j, 1].set_title(f"{name}: innovation SD by state bin")
    axes[j, 1].set_xlabel("Mean standardized state")
    axes[j, 1].set_ylabel("Innovation SD")

plt.tight_layout()
plt.show()

pd.concat(bin_rows, ignore_index=True).to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_state_bins.csv"),
    index=False,
    encoding="utf-8-sig"
)


# ------------------------------------------------------------
# 16.6 Local likelihood sensitivity for f and log(g)
# ------------------------------------------------------------

# IMPORTANT:
# This perturbs f and log(g) by CONSTANT offsets over the validation
# sample. It is a local sensitivity diagnostic, not a formal proof
# of neural-network parameter/function identifiability.
#
# The fitted training model is not guaranteed to maximize the
# validation likelihood. Therefore, curvature at (0, 0) must not
# be interpreted as a formal Fisher information matrix.

def stable_validation_loglik(dz, f, log_g, alpha, beta, dt=1.0):
    scale = np.exp(np.clip(log_g, -8, 3)) * dt ** (1.0 / alpha)
    residual = dz - f * dt

    lp = levy_stable.logpdf(
        residual,
        alpha=alpha,
        beta=beta,
        loc=0.0,
        scale=scale
    )

    lp = np.asarray(lp)
    if not np.all(np.isfinite(lp)):
        return -np.inf

    return float(np.sum(lp))


profile_rows = []
rng_profile = np.random.default_rng(SEED)
profile_n = min(diag_n, 1000)

profile_idx = np.sort(
    rng_profile.choice(diag_n, size=profile_n, replace=False)
)

for j, name in enumerate(names):
    dz = diag_increments[profile_idx, j]
    fj = diag_f[profile_idx, j]
    lgj = diag_log_g[profile_idx, j]

    # Perturbation scales are numerical choices, not confidence limits.
    f_step = max(0.25 * np.std(fj), 0.02)
    lg_step = 0.25

    f_grid = np.linspace(-2 * f_step, 2 * f_step, 9)
    lg_grid = np.linspace(-2 * lg_step, 2 * lg_step, 9)

    ll_surface = np.empty((len(lg_grid), len(f_grid)))

    for a, delta_lg in enumerate(lg_grid):
        for b, delta_f in enumerate(f_grid):
            ll_surface[a, b] = stable_validation_loglik(
                dz=dz,
                f=fj + delta_f,
                log_g=lgj + delta_lg,
                alpha=alpha_hat[j],
                beta=beta_hat[j],
                dt=DT
            )

    finite = np.isfinite(ll_surface)
    if not finite.any():
        print(f"{name}: profile likelihood failed; skipping.")
        continue

    ll_rel = np.full_like(ll_surface, np.nan)
    ll_rel[finite] = ll_surface[finite] - np.max(ll_surface[finite])

    plt.figure(figsize=(7, 5))
    contour = plt.contourf(
        f_grid, lg_grid, ll_rel, levels=20
    )
    plt.colorbar(contour, label="Relative validation log-likelihood")
    plt.scatter([0], [0], marker="x", s=100, label="Original fitted function")
    plt.xlabel("Constant offset added to f")
    plt.ylabel("Constant offset added to log(g)")
    plt.title(f"{name}: local likelihood sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Report only the maximum location and likelihood loss at the
    # original fitted function. Do not label this formal identifiability.
    max_idx = np.unravel_index(
        np.nanargmax(ll_rel), ll_rel.shape
    )

    zero_f_idx = int(np.argmin(np.abs(f_grid)))
    zero_g_idx = int(np.argmin(np.abs(lg_grid)))
    loss_at_zero = ll_rel[zero_g_idx, zero_f_idx]

    profile_rows.append({
        "margin": name,
        "best_delta_f_in_grid": f_grid[max_idx[1]],
        "best_delta_log_g_in_grid": lg_grid[max_idx[0]],
        "relative_ll_at_nearest_zero": loss_at_zero,
        "profile_grid_max_relative_ll": np.nanmax(ll_rel)
    })

profile_df = pd.DataFrame(profile_rows)

print("\nLocal likelihood sensitivity summary:")
if len(profile_df):
    print(profile_df.to_string(index=False))
    profile_df.to_csv(
        os.path.join(OUTPUT_DIR, "diagnostic_fg_profile.csv"),
        index=False,
        encoding="utf-8-sig"
    )


# ------------------------------------------------------------
# 16.7 Block bootstrap for rank dependence in PITs
# ------------------------------------------------------------

# The bootstrap retains short-range time ordering. The interval
# remains conditional on the fitted marginal models; it does not
# account for parameter-estimation uncertainty from retraining.

BOOTSTRAP_REPS = 500
BLOCK_LENGTH = 30
rng_boot = np.random.default_rng(SEED + 1)

def circular_block_indices(n, block_length, rng):
    sampled = []

    while len(sampled) < n:
        start = rng.integers(0, n)
        block = (start + np.arange(block_length)) % n
        sampled.extend(block.tolist())

    return np.asarray(sampled[:n], dtype=int)

tau_observed = kendalltau(diag_u[:, 0], diag_u[:, 1]).statistic
rho_observed = spearmanr(diag_u[:, 0], diag_u[:, 1]).statistic

tau_boot = np.empty(BOOTSTRAP_REPS)
rho_boot = np.empty(BOOTSTRAP_REPS)

for b in range(BOOTSTRAP_REPS):
    idx = circular_block_indices(
        diag_n, BLOCK_LENGTH, rng_boot
    )

    tau_boot[b] = kendalltau(
        diag_u[idx, 0], diag_u[idx, 1]
    ).statistic

    rho_boot[b] = spearmanr(
        diag_u[idx, 0], diag_u[idx, 1]
    ).statistic

bootstrap_summary = pd.DataFrame([
    {
        "statistic": "Kendall_tau",
        "estimate": tau_observed,
        "CI_2.5%": np.nanquantile(tau_boot, 0.025),
        "CI_97.5%": np.nanquantile(tau_boot, 0.975),
        "block_length": BLOCK_LENGTH,
        "bootstrap_reps": BOOTSTRAP_REPS
    },
    {
        "statistic": "Spearman_rho",
        "estimate": rho_observed,
        "CI_2.5%": np.nanquantile(rho_boot, 0.025),
        "CI_97.5%": np.nanquantile(rho_boot, 0.975),
        "block_length": BLOCK_LENGTH,
        "bootstrap_reps": BOOTSTRAP_REPS
    }
])

print("\nBlock-bootstrap dependence intervals:")
print(bootstrap_summary.to_string(index=False))

bootstrap_summary.to_csv(
    os.path.join(OUTPUT_DIR, "diagnostic_dependence_bootstrap.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nDiagnostic suite completed.")
print("Output directory:", OUTPUT_DIR)



