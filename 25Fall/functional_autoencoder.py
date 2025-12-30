import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import BSpline, splrep
import matplotlib.pyplot as plt

class FunctionalAutoencoder(nn.Module):
    def __init__(self, t_grid, num_basis=20, hidden_dim=32, hidden_dim2=16, latent_dim=8, degree=3):
        super(FunctionalAutoencoder, self).__init__()
        self.t_grid = t_grid  # 网格点, shape (M,)
        self.M = len(t_grid)
        self.dt = t_grid[1] - t_grid[0]  # 假设均匀网格
        self.num_basis = num_basis
        self.degree = degree
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        
        # 生成B-spline基函数 (均匀、端点clamped的结点向量)
        self.knots = self._make_clamped_uniform_knots(num_basis=self.num_basis, degree=self.degree)
        # 将基矩阵注册为buffer，避免参与梯度但随设备移动
        basis_matrix = self._compute_basis_matrix()  # shape (num_basis, M)
        self.register_buffer('basis_matrix', basis_matrix)
        
        # 编码器权重系数: 为每个隐藏节点k，有num_basis系数
        # w_k(t) = sum_l c_kl * B_l(t)
        # 小尺度初始化，避免前期数值过大
        self.encoder_coeffs = nn.Parameter(torch.randn(hidden_dim, num_basis) * 0.01)
        
        # 编码器多层结构
        # 积分 → hidden_dim → hidden_dim2 → latent_dim
        self.fc_encoder_1 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc_encoder_2 = nn.Linear(hidden_dim2, latent_dim)
        
        # 解码器多层结构
        # latent_dim → hidden_dim2 → hidden_dim → 基函数系数
        self.fc_decoder_1 = nn.Linear(latent_dim, hidden_dim2)
        self.fc_decoder_2 = nn.Linear(hidden_dim2, hidden_dim)
        
        # 解码器权重系数: 为每个隐藏节点k，有num_basis系数
        # v_k(t) = sum_l d_kl * B_l(t)
        self.decoder_coeffs = nn.Parameter(torch.randn(hidden_dim, num_basis) * 0.01)
        
        # 激活函数
        self.activation = nn.ReLU()

    def _make_clamped_uniform_knots(self, num_basis, degree):
        """构造均匀且端点重复(degree+1次)的clamped结点向量。
        对于BSpline(t, c, k): 需要 len(t) = n + k + 1，其中 n = len(c) = num_basis。
        端点重复 k+1 次，内部结点个数为 n - k - 1（若>0）。
        """
        n = num_basis
        k = degree
        if n <= k:
            raise ValueError("num_basis 必须大于 degree")
        num_internal = n - k - 1
        if num_internal > 0:
            internal = np.linspace(0.0, 1.0, num_internal + 2)[1:-1]  # 去掉端点
        else:
            internal = np.array([])
        t = np.concatenate([
            np.zeros(k + 1),
            internal,
            np.ones(k + 1)
        ])
        return t

    def _compute_basis_matrix(self):
        """计算B-spline基在网格上的值, shape (num_basis, M)"""
        basis_matrix = np.zeros((self.num_basis, self.M))
        for l in range(self.num_basis):
            # extrapolate=True 以防由于数值误差落在区间外导致 NaN
            spline = BSpline(self.knots, np.eye(self.num_basis)[l], k=self.degree, extrapolate=True)
            basis_matrix[l] = spline(self.t_grid)
        return torch.tensor(basis_matrix, dtype=torch.float32)
    
    def forward(self, x):
        """x: 输入函数数据, shape (batch_size, M)"""
        # 编码器
        # 1. 函数积分
        w_grid = self.encoder_coeffs @ self.basis_matrix
        integrals = torch.einsum('bm,km->bk', x, w_grid) * self.dt
        
        # 2. 通过多层全连接网络
        hidden1 = self.activation(integrals)
        hidden2 = self.activation(self.fc_encoder_1(hidden1))
        latent = self.fc_encoder_2(hidden2)  # 潜变量通常不加激活
        
        # 解码器
        # 1. 通过多层全连接网络
        hidden2_dec = self.activation(self.fc_decoder_1(latent))
        hidden1_dec = self.activation(self.fc_decoder_2(hidden2_dec))
        
        # 2. 使用输出作为系数重构函数
        v_grid = self.decoder_coeffs @ self.basis_matrix
        recon = torch.einsum('bk,km->bm', hidden1_dec, v_grid)
        
        return recon, latent

# 示例训练函数
def train_fae(data, t_grid, epochs=500, lr=0.005, batch_size=32, 
              num_basis=20, hidden_dim=64, hidden_dim2=32, latent_dim=16, degree=3):
    model = FunctionalAutoencoder(t_grid, num_basis=num_basis, hidden_dim=hidden_dim, 
                                  hidden_dim2=hidden_dim2, latent_dim=latent_dim, degree=degree)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum') 
    dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\n--- Training Original FAE ---")
    final_loss = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch[0]  # (batch_size, M)
            recon, _ = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 归一化方式改为按样本数，与VAE/GMVAE保持一致
        avg_loss = total_loss / len(loader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            # 调整打印格式以匹配新的损失尺度
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        if epoch == epochs - 1:
            final_loss = avg_loss

    return model, final_loss

def visualize(model, data, t_grid, num_examples=5):
    """绘制重构曲线与潜变量散点。
    - 重构: 选取若干样本，画原始与重构对比
    - 潜变量: 画前两维的散点
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        recon_all, latent_all = model(data_tensor)
        recon_all = recon_all.cpu().numpy()
        latent_all = latent_all.cpu().numpy()

    # 重构可视化
    examples = min(num_examples, data.shape[0])
    plt.figure(figsize=(10, 6))
    for i in range(examples):
        plt.subplot(examples, 1, i + 1)
        plt.plot(t_grid, data[i], label='Original', linewidth=1.5)
        plt.plot(t_grid, recon_all[i], label='Reconstruction', linestyle='--', linewidth=1.5)
        if i == 0:
            plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.tight_layout()
    plt.suptitle('Reconstruction Examples', y=1.02)
    plt.tight_layout()
    plt.savefig('reconstruction_examples.png', dpi=150)

    # 潜变量可视化（前两维）
    plt.figure(figsize=(6, 5))
    if latent_all.shape[1] >= 2:
        plt.scatter(latent_all[:, 0], latent_all[:, 1], s=20, alpha=0.8)
        plt.xlabel('Latent dim 1')
        plt.ylabel('Latent dim 2')
        plt.title('Latent Space (first two dims)')
    else:
        plt.scatter(np.arange(latent_all.shape[0]), latent_all[:, 0], s=20, alpha=0.8)
        plt.xlabel('Sample index')
        plt.ylabel('Latent dim 1')
        plt.title('Latent Space (single dim)')
    plt.tight_layout()
    plt.savefig('latent_scatter.png', dpi=150)

    # 可选：展示
    try:
        plt.show()
    except Exception:
        pass

def generate_complex_data(n_samples=100, n_points=100):
    """生成更复杂的函数型数据，由多个随机组件构成"""
    t = np.linspace(0, 1, n_points)
    data = np.zeros((n_samples, n_points))
    for i in range(n_samples):
        # 1. 随机频率和相位的正弦波
        freq = np.random.uniform(1, 5)
        phase = np.random.uniform(0, np.pi)
        c1 = np.sin(2 * np.pi * freq * t + phase)
        
        # 2. 随机二次趋势
        a = np.random.uniform(-2, 2)
        b = np.random.uniform(-2, 2)
        c2 = a * (t - 0.5)**2 + b * t
        
        # 3. 随机位置和宽度的高斯峰
        mean = np.random.uniform(0.2, 0.8)
        std = np.random.uniform(0.05, 0.15)
        height = np.random.uniform(0.5, 1.5)
        c3 = height * np.exp(-((t - mean)**2) / (2 * std**2))
        
        # 随机加权组合
        w1 = np.random.uniform(0.5, 1.5)
        w2 = np.random.uniform(0.2, 0.8)
        w3 = np.random.uniform(0.5, 1.2)
        curve = w1 * c1 + w2 * c2 + w3 * c3
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, n_points)
        data[i, :] = curve + noise
        
    return t, data

# 示例使用
if __name__ == "__main__":
    # 使用更复杂的数据
    t_grid, data = generate_complex_data(n_samples=200, n_points=100)
    
    model, _ = train_fae(data, t_grid) # The returned loss is not needed here
    # 测试
    test_x = torch.tensor(data[0:1], dtype=torch.float32)
    recon, latent = model(test_x)
    print("Latent representation:", latent)
    # 可视化
    visualize(model, data, t_grid, num_examples=5)