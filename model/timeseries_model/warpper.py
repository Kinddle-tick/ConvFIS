import torch
import torch.nn as nn


class _TFmodel_warped(nn.Module):
    def __init__(self, tfmodel: nn.Module):
        super().__init__()
        self.tfmodel = tfmodel

    def forward(self, samples, labels=None, *args_, flg_train=True, **kwargs):
        raise NotImplementedError

    def loss(self, pred, target):
        raise NotImplementedError

    def configure_optimizers(self, lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95)):
        raise NotImplementedError


class TFmodel_warped(_TFmodel_warped):
    """
    [...,in_time_dim, state_dim] -> [...,out_time_dim, state_dim]

    """
    def __init__(self, tfmodel: nn.Module):
        super().__init__(tfmodel)
        self.loss_model = torch.nn.MSELoss(reduction="none")
        self.input_len = tfmodel.seq_len
        self.pred_len = tfmodel.pred_len
    
    def forward(self, samples, labels=None, *args_, flg_train=True, **kwargs):
        model = self.tfmodel
        batch_size = samples.shape[0]  
        target_seq = labels

        x_enc = samples
        
        # decoder input
        dec_inp = torch.zeros_like(labels).float()
        dec_inp = torch.cat([samples, dec_inp], dim=1).float().to(samples.device)
        x_dec=dec_inp
        
        diff_predictions = model.forward(x_enc, None, x_dec, None, mask=None)

        full_predictions = samples[...,-1:,:] + diff_predictions[..., -self.pred_len:,:]
        
        return full_predictions[..., -self.pred_len:,:], self.loss(full_predictions, target_seq)  
    
        
    def loss(self, pred, target):
        return self.loss_model(pred[...,0:2],target[...,0:2]).mean((-1,-2))
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.tfmodel.parameters())
    
if __name__ == "__main__":
    import torch
    from collections import namedtuple  
    import numpy as np  
    import matplotlib.pyplot as plt  
    from model import iTransformer
    from model.timeseries_model.warpper import TFmodel_warped
    
    
    # 创建一个命名元组，或者也可使用简单的类来存储配置参数。  
    Config = namedtuple('Config', [  
        'seq_len', 'pred_len', 'output_attention', 'use_norm', 'd_model',  
        'embed', 'freq', 'dropout', 'class_strategy', 'factor', 'n_heads',  
        'e_layers', 'd_ff', 'activation'  
    ])  

    # 实例化配置对象，使用命令行作为默认值。  
    configs = Config(  
        seq_len=50,  
        pred_len=50,  
        output_attention=False,  # 假定一个默认值，如果脚本中没有提供这个的话  
        use_norm=False,  # 假定一个默认值，如果脚本中没有提供这个的话  
        d_model=512,  
        embed='linear',  # 假定一个可能的默认值  
        freq='h',  # 假定一个可能的默认值  
        dropout=0.1,  # 假定一个可能的默认值  
        class_strategy='simple',  # 假定一个可能的默认值  
        factor=5,  # 假定一个可能的默认值  
        n_heads=8,  # 假定一个可能的默认值  
        e_layers=3,  
        d_ff=512,  
        activation='relu'  # 假定一个可能的默认值  
    )  
    
    
    # Generate sine and cosine time series  
    def generate_time_series(batch_size, seq_length):  
        x = np.linspace(0, 2 * np.pi, seq_length)  
        sin_wave = np.sin(x)  
        cos_wave = np.cos(x)  
        series = np.vstack([sin_wave, cos_wave]).T  
        return series  

    model = TFmodel_warped(iTransformer(configs))

    # Training setup  
    optimizer = model.configure_optimizers()  

    # Data preparation  
    seq_length = 100  
    batch_size = 16  
    epochs = 100  

    # Create synthetic data  
    x_data = np.array([generate_time_series(batch_size, seq_length) for _ in range(batch_size)])  
    y_data = x_data[...,50:,:]  # Target is same as input in this mock setup  
    x_data = x_data[...,:50,:]
    
    # Convert to torch tensors  
    x_tensor = torch.tensor(x_data, dtype=torch.float32)  
    y_tensor = torch.tensor(y_data, dtype=torch.float32)  

    # Training loop  
    for epoch in range(epochs):  
        model.train()  
        optimizer.zero_grad()  
        
        # Forward pass: using all data as both input (samples) and target (labels)  
        prediction, loss = model(x_tensor, labels=y_tensor, mode="train")  
        
        # Backward pass and optimization  
        loss.mean().backward()  
        optimizer.step()  

        if epoch % 10 == 0:  
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.mean().item():.4f}")  

    # Testing phase  
    model.eval()  
    with torch.no_grad():  
        prediction, _ = model(x_tensor, labels=y_tensor, mode="test")  

    # Plot results for visualization  
    plt.figure(figsize=(12, 6))  
    plt.plot(y_tensor[0, :, 0], label='True Sine', linestyle='--')  
    plt.plot(prediction[0, :, 0].numpy(), label='Predicted Sine')  
    plt.plot(y_tensor[0, :, 1], label='True Cosine', linestyle='--')  
    plt.plot(prediction[0, :, 1].numpy(), label='Predicted Cosine')  
    plt.legend()  
    plt.show()  