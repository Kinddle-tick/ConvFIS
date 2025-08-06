import torch  
import numpy as np  
import matplotlib.pyplot as plt  
from transformers import GPT2Config, GPT2Model  
from transformers.pytorch_utils import Conv1D
import math  

class GPTTrack(torch.nn.Module):  
    """  
    输入输出维度  
    [...,in_time_dim, state_dim] -> [...,out_time_dim, state_dim]  
    """  
    def __init__(self, state_dim_in=9, state_dim_out=9,  
                 in_time_dim=1, out_time_dim=1,  
                 num_layers=6, n_head=8, n_embd=256,   
                 dropout=0.1, device="cuda:0"):  
        super().__init__()  
        self.state_dim_in = state_dim_in  
        self.state_dim_out = state_dim_out  
        self.in_time_dim = in_time_dim  
        self.out_time_dim = out_time_dim  
        self.n_embd = n_embd  
        self.device = device  
        
        # 创建GPT2配置  
        self.config = GPT2Config(  
            vocab_size=1,  # 不使用词嵌入  
            n_positions=in_time_dim + out_time_dim,  
            n_ctx=in_time_dim + out_time_dim,  
            n_embd=n_embd,  
            n_layer=num_layers,  
            n_head=n_head,  
            resid_pdrop=dropout,  
            attn_pdrop=dropout,  
            embd_pdrop=dropout,  
        )  
        
        # 输入投影层  
        self.input_proj = torch.nn.Linear(state_dim_in, n_embd)  
        
        # GPT2模型  
        self.transformer = GPT2Model(self.config)  
        
        # 输出层  
        self.output_head = torch.nn.Linear(n_embd, state_dim_out)  
        
        # 损失函数  
        self.loss_model = torch.nn.MSELoss(reduction="none")  
        
        # 将模型移至指定设备  
        self.to(device)  
    
    # def loss(self, pred, target):  
    #     # # 与示例中类似的损失函数  
    #     # if pred.shape[-1] >= 4:  
    #     #     # 如果维度足够，采用与示例类似的加权损失  
    #     #     raw_loss = (torch.sum(self.loss_model(pred[..., 0:2], target[..., 0:2]), dim=-1)  
    #     #             + torch.sum(self.loss_model(pred[..., 2:4], target[..., 2:4]) * 1e-5, dim=-1))  
    #     # else:  
    #         # 普通MSE损失  
    #     raw_loss = torch.sum(self.loss_model(pred, target), dim=-1)  
        
    #     loss = torch.sum(raw_loss, dim=-1)  
    #     return loss  
    
    def loss(self, pred, target):
        pred_euclidean_dists = torch.norm(torch.diff(pred[..., 0:2],dim=-2), dim=-1)  # 结果形状为 (batch_size, timestep - 1)
        target_euclidean_dists = torch.norm(torch.diff(target[..., 0:2],dim=-2), dim=-1)  # 结果形状为 (batch_size, timestep - 1)
        loss = (self.loss_model(pred[..., 0:2], target[..., 0:2]).mean((-1,-2))
        
                + self.loss_model(pred[..., 2:4], target[..., 2:4]).mean((-1,-2)) * 1e-3
                + torch.pow(pred_euclidean_dists-target_euclidean_dists,2).mean((-1,))* 1e-1
                #+ torch.sum(torch.pow(self.smooth_loss_model(pred[..., 0:2])-self.smooth_loss_model2(target[..., 0:2]),2))
        )
        return loss
    
    def forward(self, samples, labels=None, *args_, flg_train=True, **kwargs):  
        batch_size = samples.shape[0]  
        
        if kwargs["mode"]=="test":
            flg_train = False
        
        if flg_train and labels is not None:
            # 训练阶段 - Teacher Forcing  
            output_len = labels.shape[-2]
            # 准备目标序列（错开一位） 
            combined_seq = torch.cat([samples, labels], dim=-2)
            init_seq = combined_seq[..., :-1,:]
            target_seq = combined_seq[..., 1: ,:]  
            # 输入嵌入  
            init_embeds = self.input_proj(init_seq) 
            
            # 通过transformer  
            outputs = self.transformer(inputs_embeds=init_embeds)  
            output_logits = outputs.last_hidden_state  
            
            # # 添加最后一个位置的预测  
            full_predictions = self.output_head(output_logits)
            full_predictions[..., -output_len:,:]=full_predictions[..., -output_len:,:]+samples[...,-1:,:]
            return full_predictions[..., -output_len:,:], self.loss(full_predictions, target_seq)  
        
        else:  
            # 预测阶段 - 自回归生成
            # 输入嵌入  
            input_embeds = self.input_proj(samples)   
            predictions = torch.zeros(batch_size, self.out_time_dim, self.state_dim_out, device=self.device)  
            current_embeds = input_embeds  
            
            for i in range(self.out_time_dim):  
                # 通过transformer  
                outputs = self.transformer(inputs_embeds=current_embeds)  
                last_logits = outputs.last_hidden_state[...,-1,:]  
                
                # 预测下一个值  
                next_pred = self.output_head(last_logits)
                next_pred = samples[...,-1,:]+next_pred
                predictions[..., i, :] = next_pred  
                
                # 如果不是最后一步，添加到输入序列  
                if i < self.out_time_dim - 1:  
                    next_embed = self.input_proj(next_pred.unsqueeze(1))  
                    current_embeds = torch.cat([current_embeds, next_embed], dim=1)  
            
            if labels is not None:  
                return predictions, self.loss(predictions, labels)  
            return predictions  

    def configure_optimizers(self, lr=5e-5, weight_decay=0.1,betas=(0.9,0.95)):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        
        # for mn, m in self.named_modules():
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn
        #         print(f"Module name: {mn}, Module type: {type(m)}, Param name: {pn}")
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Conv1D)  # 包含 Conv1D
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # 使用 f-string 更安全

                # 规则 1：所有偏置（bias）不衰减
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                # 规则 2：白名单模块的权重（weight）衰减
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                # 规则 3：黑名单模块的权重（weight）不衰减
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

            # special case the position embedding parameter in the root GPT module as not decayed
            no_decay.add('transformer.wpe.weight')
            no_decay.add('transformer.wte.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

# 测试代码 - 生成正弦函数数据并训练模型  
def test_sine_prediction(device="cuda:0"):  
    # 设置随机种子以保证可复现性  
    torch.manual_seed(42)  
    np.random.seed(42)  
    
    # 参数设置  
    seq_length = 100  # 序列总长度  
    in_seq_len = 50   # 输入序列长度  
    out_seq_len = 50  # 输出序列长度  
    state_dim = 1     # 特征维度 (正弦函数只有一个值)  
    batch_size = 32  
    epochs = 100  
    
    # 生成正弦函数数据  
    def generate_sine_data(samples, seq_len, freq=0.1, phase_shift=0):  
        x = np.arange(0, seq_len)  
        data = np.sin(2 * np.pi * freq * x + phase_shift)  
        result = np.zeros((samples, seq_len, 1))  
        for i in range(samples):  
            # 随机相位偏移使数据更多样化  
            shift = np.random.uniform(0, 2*np.pi)  
            result[i, :, 0] = np.sin(2 * np.pi * freq * x + shift)  
        return result  
    
    # 创建训练数据  
    train_data = generate_sine_data(500, seq_length, freq=0.05)  
    
    # 创建验证数据  
    val_data = generate_sine_data(100, seq_length, freq=0.05)  
    
    # 创建数据加载器  
    train_inputs = torch.FloatTensor(train_data[:, :in_seq_len, :]).to(device)  
    train_targets = torch.FloatTensor(train_data[:, in_seq_len:, :]).to(device)  
    val_inputs = torch.FloatTensor(val_data[:, :in_seq_len, :]).to(device)  
    val_targets = torch.FloatTensor(val_data[:, in_seq_len:, :]).to(device)  
    
    # 初始化模型  
    model = GPTTrack(  
        state_dim_in=state_dim,  
        state_dim_out=state_dim,  
        in_time_dim=in_seq_len,  
        out_time_dim=out_seq_len,  
        num_layers=4,  
        n_head=4,  
        n_embd=128,  
        device=device  
    )  
    
    # 优化器  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)  
    
    # 训练循环  
    train_losses = []  
    val_losses = []  
    best_val_loss = float('inf')  
    
    print(f"开始训练，设备: {device}")  
    for epoch in range(epochs):  
        # 训练模式  
        model.train()  
        for i in range(0, len(train_inputs), batch_size):  
            batch_inputs = train_inputs[i:i+batch_size]  
            batch_targets = train_targets[i:i+batch_size]  
            
            optimizer.zero_grad()  
            _, loss = model(batch_inputs, batch_targets, flg_train=True,mode = "train")  
            loss_mean = loss.mean()  
            loss_mean.backward()  
            optimizer.step()  
            
        # 验证模式  
        model.eval()  
        with torch.no_grad():  
            val_preds, val_loss = model(val_inputs, val_targets, flg_train=True,mode = "val")  
            val_loss_mean = val_loss.mean().item()  
            
            # 计算训练损失  
            train_preds, train_loss = model(train_inputs[:100], train_targets[:100], flg_train=True,mode = "val")  
            train_loss_mean = train_loss.mean().item()  
            
        # 保存损失  
        train_losses.append(train_loss_mean)  
        val_losses.append(val_loss_mean)  
        
        # 更新学习率  
        scheduler.step(val_loss_mean)  
        
        # 打印进度  
        if (epoch + 1) % 10 == 0:  
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss_mean:.6f}, Val Loss: {val_loss_mean:.6f}")  
        
        # 保存最佳模型  
        if val_loss_mean < best_val_loss:  
            best_val_loss = val_loss_mean  
            torch.save(model.state_dict(), f"gpt_track_best_model_{device.replace(':', '_')}.pt")  
    
    # 加载最佳模型  
    model.load_state_dict(torch.load(f"gpt_track_best_model_{device.replace(':', '_')}.pt"))  
    
    # 绘制损失曲线  
    plt.figure(figsize=(10, 5))  
    plt.plot(train_losses, label='训练损失')  
    plt.plot(val_losses, label='验证损失')  
    plt.xlabel('Epoch')  
    plt.ylabel('损失')  
    plt.title('训练和验证损失')  
    plt.legend()  
    plt.savefig(f'gpt_track_loss_{device.replace(":", "_")}.png')  
    
    # 测试预测效果  
    model.eval()  
    with torch.no_grad():  
        # 自回归预测  
        test_input = val_inputs[0:1]  # 取第一个样本  
        predictions = model(test_input,mode="test")  
        
        # 转为numpy，方便绘图  
        input_np = test_input.cpu().numpy()[0, :, 0]  
        true_output_np = val_targets[0].cpu().numpy()[:, 0]  
        pred_output_np = predictions.cpu().numpy()[0, :, 0]  
        
        # 绘制结果  
        plt.figure(figsize=(12, 6))  
        t_input = np.arange(in_seq_len)  
        t_output = np.arange(in_seq_len, in_seq_len + out_seq_len)  
        
        plt.plot(t_input, input_np, 'b-', label='输入序列')  
        plt.plot(t_output, true_output_np, 'g-', label='真实序列')  
        plt.plot(t_output, pred_output_np, 'r--', label='预测序列')  
        
        plt.grid(True)  
        plt.xlabel('时间步')  
        plt.ylabel('数值')  
        plt.title('GPT时间序列预测 - 正弦函数')  
        plt.legend()  
        plt.savefig(f'gpt_track_prediction_{device.replace(":", "_")}.png')  
        plt.show()  
        
        print("预测评估 - MSE:", np.mean((true_output_np - pred_output_np)**2))  
        
    return model, train_losses, val_losses  

# 运行测试，指定GPU设备  
if __name__ == "__main__":  
    # 可以指定任何可用的GPU设备  
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  
    model, train_losses, val_losses = test_sine_prediction(device)  