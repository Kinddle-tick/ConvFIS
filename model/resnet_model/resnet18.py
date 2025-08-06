import torch  
import torch.nn as nn  
import torch.optim as optim # 引入优化器  
import torchvision.models as models  
import numpy as np  
import matplotlib.pyplot as plt  

class Resnet18Track(nn.Module):  
    """  
    使用 ResNet-18 进行时序预测的模型。  

    输入维度: [B, T_in, C_in] (Batch, Input Time Steps, Input State Dim)  
    输出维度: [B, T_out, C_out] (Batch, Output Time Steps, Output State Dim)  

    将时间步 T_in 作为 ResNet 的输入通道，将 C_in 映射到虚拟的 HxW 空间。  
    """  
    def __init__(self, state_dim_in=9, state_dim_out=9,  
                 in_time_dim=1, out_time_dim=1,  
                 device="cuda:0"):  
        super().__init__()  

        self.state_dim_in = state_dim_in  
        self.state_dim_out = state_dim_out  
        self.in_time_dim = in_time_dim  
        self.out_time_dim = out_time_dim  
        self.device = device  

        # 1. 输入投影层：将 C_in 映射到 H*W (e.g., 32*32=1024)  
        self.h = 2  
        self.w = 2  
        self.input_proj_dim = self.h * self.w  
        self.input_proj = nn.Linear(state_dim_in, self.input_proj_dim)  

        # 2. 加载 ResNet-18 (无预训练权重)  
        resnet = models.resnet18(weights=None)  

        # 3. 修改 ResNet 的第一个卷积层以接受 T_in 个通道  
        original_conv1 = resnet.conv1  
        self.conv1 = nn.Conv2d(  
            in_time_dim,  
            original_conv1.out_channels,  
            kernel_size=original_conv1.kernel_size,  
            stride=original_conv1.stride,  
            padding=original_conv1.padding,  
            bias=False  
        )  
        resnet.conv1 = self.conv1  

        # 4. 获取 ResNet 特征提取部分  
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])  
        resnet_output_dim = resnet.fc.in_features # 512  

        # 5. 输出投影层：将 ResNet 特征映射到 T_out * C_out  
        self.output_proj = nn.Linear(resnet_output_dim, out_time_dim * state_dim_out)  
        
        # 损失函数  
        self.loss_model = nn.MSELoss(reduction="none")  
        
        self.to(device)  
        
    def forward(self, samples, labels=None, *args_, flg_train=True, **kwargs):  
        B = samples.shape[0]  
        # --- 检查维度 ---  
        assert samples.shape[1] == self.in_time_dim, f"Input time dimension mismatch: expected {self.in_time_dim}, got {samples.shape[1]}"  
        assert samples.shape[2] == self.state_dim_in, f"Input state dimension mismatch: expected {self.state_dim_in}, got {samples.shape[2]}"  

        # --- 核心计算 ---  
        x = self.input_proj(samples) # -> [B, T_in, H*W]  
        x = x.view(B, self.in_time_dim, self.h, self.w) # -> [B, T_in, H, W]  

        features = self.resnet_features(x) # -> [B, resnet_output_dim, 1, 1]  
        features_flat = torch.flatten(features, 1) # -> [B, resnet_output_dim]  
        predictions_flat = self.output_proj(features_flat) # -> [B, T_out * C_out]  
        predictions = predictions_flat.view(B, self.out_time_dim, self.state_dim_out) # -> [B, T_out, C_out]  

        # --- 计算损失并返回 ---  
        if labels is not None:  
            labels = labels.to(self.device) # 确保标签在同一设备  
            loss = self.loss(predictions, labels) # 使用 self.loss_fn 计算损失  
            return predictions, loss  
        else:  
            return predictions
          
    def loss(self, pred, target):
        return self.loss_model(pred[...,0:2],target[...,0:2]).mean((-1,-2))

    def configure_optimizers(self, lr=5e-4, weight_decay=0.1, betas=(0.9, 0.999)):  
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)  
        return optimizer  


# === 测试用例与训练 ===  
if __name__ == '__main__':  
    # 1. 定义参数  
    batch_size = 16        # 稍微增大 batch size 以获得更稳定的梯度  
    in_time_dim = 10       # 输入时间步 T_in  
    out_time_dim = 50      # 输出时间步 T_out  
    state_dim = 2          # 状态维度 C_in 和 C_out (sin, cos)  
    total_time = in_time_dim + out_time_dim  
    num_epochs = 100       # 训练轮数  
    learning_rate = 1e-3   # 学习率  

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    print(f"Using device: {device}")  

    # 2. 生成模拟数据  
    all_samples_np = []  
    all_labels_np = []  
    time_vector = np.arange(total_time) * 0.1  

    for i in range(batch_size):  
        phase_shift = np.random.rand() * np.pi * 2  
        t = time_vector + phase_shift  
        sin_wave = np.sin(t)  
        cos_wave = np.cos(t)  
        sequence = np.stack([sin_wave, cos_wave], axis=-1)  
        sample = sequence[:in_time_dim, :]  
        label = sequence[in_time_dim:total_time, :]  
        all_samples_np.append(sample)  
        all_labels_np.append(label)  

    samples_np = np.stack(all_samples_np, axis=0)  
    labels_np = np.stack(all_labels_np, axis=0)  
    samples = torch.tensor(samples_np, dtype=torch.float32).to(device)  
    labels = torch.tensor(labels_np, dtype=torch.float32).to(device)  

    print(f"Generated samples shape: {samples.shape}") # [B, 10, 2]  
    print(f"Generated labels shape: {labels.shape}")   # [B, 50, 2]  

    # 3. 实例化模型  
    model = Resnet18Track(  
        state_dim_in=state_dim,  
        state_dim_out=state_dim,  
        in_time_dim=in_time_dim,  
        out_time_dim=out_time_dim,  
        device=device  
    )  

    # 4. 定义优化器  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  

    # 5. 训练循环  
    print("\n--- Starting Training ---")  
    for epoch in range(num_epochs):  
        model.train() # 设置为训练模式  

        # 前向传播  
        predictions_train, loss = model(samples, labels) # 获取预测和损失  
        loss = loss.mean()
        # 反向传播和优化  
        optimizer.zero_grad() # 清空梯度  
        loss.backward()       # 计算梯度  
        optimizer.step()      # 更新权重  

        # 打印损失 (每 10 轮)  
        if (epoch + 1) % 10 == 0:  
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')  

    print("--- Training Finished ---")  

    # 6. 评估训练后的模型  
    model.eval() # 设置为评估模式  
    with torch.no_grad(): # 推理时不需要计算梯度  
        predictions_final = model(samples) # 只获取预测结果  

    print("\n--- Evaluation After Training ---")  
    print(f"Final predictions shape: {predictions_final.shape}") # 应为 [B, 50, 2]  

    # 计算最终的 MSE 损失（用于比较）  
    final_loss = model.loss(predictions_final, labels).mean()  
    print(f"Final MSE Loss on training data: {final_loss.item():.6f}")  


    # 7. 可视化第一个样本的结果 (训练后)  
    plt.figure(figsize=(12, 6))  

    # 获取第一个样本的数据 (移回 CPU 并转为 numpy)  
    sample_0_in = samples[0].cpu().numpy()  
    label_0_out = labels[0].cpu().numpy()  
    pred_0_out_final = predictions_final[0].cpu().numpy() # 使用训练后的预测  

    # 时间轴  
    time_in_axis = np.arange(in_time_dim)  
    time_out_axis = np.arange(in_time_dim, total_time)  

    # 绘制 Sin 波  
    plt.subplot(1, 2, 1)  
    plt.plot(time_in_axis, sample_0_in[:, 0], 'bo-', label='Input Sin')  
    plt.plot(time_out_axis, label_0_out[:, 0], 'g-', linewidth=2, label='True Future Sin')  
    plt.plot(time_out_axis, pred_0_out_final[:, 0], 'r--', label='Predicted Future Sin (Trained)')  
    plt.title('Sine Wave Prediction (After Training)')  
    plt.xlabel('Time Steps')  
    plt.ylabel('Value')  
    plt.legend()  
    plt.grid(True)  

    # 绘制 Cos 波  
    plt.subplot(1, 2, 2)  
    plt.plot(time_in_axis, sample_0_in[:, 1], 'bo-', label='Input Cos')  
    plt.plot(time_out_axis, label_0_out[:, 1], 'g-', linewidth=2, label='True Future Cos')  
    plt.plot(time_out_axis, pred_0_out_final[:, 1], 'r--', label='Predicted Future Cos (Trained)')  
    plt.title('Cosine Wave Prediction (After Training)')  
    plt.xlabel('Time Steps')  
    plt.ylabel('Value')  
    plt.legend()  
    plt.grid(True)  

    plt.tight_layout()  
    plt.show()  