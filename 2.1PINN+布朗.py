import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------
# 1. 数据加载与预处理
# ------------------------------
df = pd.read_csv(r"D:\降水与径流量\data\dataset.csv", parse_dates=['date'])
X = df['discharge_upstream'].values.astype(np.float32)
Y = df['discharge_downstream'].values.astype(np.float32)
dt = 1.0 

# --- 数据标准化（使训练稳定）---
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_norm = scaler_X.fit_transform(X.reshape(-1,1)).flatten()
Y_norm = scaler_Y.fit_transform(Y.reshape(-1,1)).flatten()

# 计算差分（对应于 dX, dY）
dX = np.diff(X_norm)
dY = np.diff(Y_norm)
# 状态使用前 N-1 个点
X_state = X_norm[:-1]
Y_state = Y_norm[:-1]

# 转换为 PyTorch tensor
X_state_t = torch.tensor(X_state, dtype=torch.float32).unsqueeze(1)   # (N,1)
Y_state_t = torch.tensor(Y_state, dtype=torch.float32).unsqueeze(1)
dX_t = torch.tensor(dX, dtype=torch.float32).unsqueeze(1)
dY_t = torch.tensor(dY, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_state_t, Y_state_t, dX_t, dY_t)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# ------------------------------
# 2. 定义神经网络模型 (同时输出 f1,f2,g1,g2)
# ------------------------------
class DriftDiffusionNet(nn.Module):
    def __init__(self, hidden_dim=64, depth=3):
        super().__init__()
        layers = []
        input_dim = 2  # X, Y
        for i in range(depth):
            layers.append(nn.Linear(input_dim if i==0 else hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 4))  # 输出: mu1, mu2, log_sigma1, log_sigma2
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)   # (batch,2)
        out = self.net(inp)              # (batch,4)
        mu1, mu2, log_sigma1, log_sigma2 = out[:,0:1], out[:,1:2], out[:,2:3], out[:,3:4]
        # 保证 sigma 为正
        sigma1 = torch.exp(log_sigma1) + 1e-6
        sigma2 = torch.exp(log_sigma2) + 1e-6
        return mu1, mu2, sigma1, sigma2

model = DriftDiffusionNet(hidden_dim=64, depth=3)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

# ------------------------------
# 3. 训练循环（负对数似然）
# ------------------------------
def nll_loss(mu1, mu2, sigma1, sigma2, dX, dY, dt):
    # 计算增量条件下的预测均值与方差
    mean_X = mu1 * dt
    mean_Y = mu2 * dt
    var_X = (sigma1**2) * dt
    var_Y = (sigma2**2) * dt
    
    # 高斯负对数似然
    loss_X = 0.5 * ((dX - mean_X)**2 / var_X + torch.log(2*np.pi*var_X))
    loss_Y = 0.5 * ((dY - mean_Y)**2 / var_Y + torch.log(2*np.pi*var_Y))
    return (loss_X + loss_Y).mean()

num_epochs = 500
train_losses = []
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for Xb, Yb, dXb, dYb in dataloader:
        optimizer.zero_grad()
        mu1, mu2, sigma1, sigma2 = model(Xb, Yb)
        loss = nll_loss(mu1, mu2, sigma1, sigma2, dXb, dYb, dt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(Xb)
    epoch_loss /= len(dataset)
    train_losses.append(epoch_loss)
    scheduler.step(epoch_loss)
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

# 绘制损失曲线
plt.figure(figsize=(8,4))
plt.plot(train_losses,'b-',color="#1f77b4")
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training Loss')
plt.grid(True)
plt.show()

# ------------------------------
# 4. 提取标准化残差（供 Copula 建模）
# ------------------------------
model.eval()
with torch.no_grad():
    mu1, mu2, sigma1, sigma2 = model(X_state_t, Y_state_t)
    # 计算标准化残差 (epsilon)
    eps1 = (dX_t - mu1 * dt) / (sigma1 * np.sqrt(dt))
    eps2 = (dY_t - mu2 * dt) / (sigma2 * np.sqrt(dt))
    eps1 = eps1.numpy().flatten()
    eps2 = eps2.numpy().flatten()

# 检查残差的基本统计
print("残差 eps1 均值: {:.4f}, 方差: {:.4f}".format(eps1.mean(), eps1.var()))
print("残差 eps2 均值: {:.4f}, 方差: {:.4f}".format(eps2.mean(), eps2.var()))

# 可视化残差散点图（可初步观察相依结构）
plt.figure(figsize=(6,5))
plt.scatter(eps1, eps2, s=1, alpha=0.5)
plt.xlabel('Standardized Residual (Upstream)')
plt.ylabel('Standardized Residual (Downstream)')
plt.title('Residual Pairs for Copula Modeling')
plt.grid(True)
plt.show()

# ------------------------------
# 5. 保存模型与残差（后续 Copula 分析用）
# ------------------------------
torch.save(model.state_dict(), "sde_dd_model.pth")
np.savez("residuals.npz", eps1=eps1, eps2=eps2, dt=dt)

# 另外保存标准化器的参数，以便反变换回原始尺度（如果需要）
import joblib
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_Y, "scaler_Y.pkl")

print("模型及残差已保存。下一步可进行Copula边缘拟合与相依结构估计。")


