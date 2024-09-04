import torch
import torch.nn as nn
from fairseq.criterions.l0 import L0

class AdaptiveFeatureSelection(nn.Module):
    def __init__(self, input_dim, training=True, dropout=0.1, enable_afs_t=False, enable_afs_f=False):
        super().__init__()
        # torch.set_default_dtype(torch.float16)
        self.input_dim = input_dim
        self.enable_afs_t = enable_afs_t
        self.enable_afs_f = enable_afs_f
        self.dropout = nn.Dropout(dropout)
        self.training = training
        self.beta = nn.Parameter(torch.tensor(2.0 / 3.0), requires_grad=False)
        self.gamma = nn.Parameter(torch.tensor(-0.1), requires_grad=False)
        self.zeta = nn.Parameter(torch.tensor(1.1), requires_grad=False)
        self.epsilon = nn.Parameter(torch.tensor(1e-8), requires_grad=False)

        if enable_afs_t:
            self.source_pruning = nn.Linear(input_dim, 1)
        
        if enable_afs_f:
            self.neuron_pruning = nn.Parameter(torch.zeros(1, 1, input_dim))


    def forward(self, x, mask=None):
        l0_norm = 0.0

        if self.enable_afs_t:
            source_pruning = self.source_pruning(x)
            if self.training:
                x, weight_noise = self.var_train((x, source_pruning))
                l0_norm += self.l0_norm(source_pruning).sum()
            else:
                x, weight_noise = self.var_eval(x, source_pruning)
                # Do anyway to log in fairseq
                l0_norm += self.l0_norm(source_pruning).sum()
        if self.enable_afs_f:
            if self.training:
                x, neuron_noise = self.var_train((x, self.neuron_pruning))
                l0_norm += self.l0_norm(self.neuron_pruning).sum()
            else:
                x, neuron_noise = self.var_eval(x, self.neuron_pruning)
                l0_norm += self.l0_norm(self.neuron_pruning).sum()

        x = self.dropout(x)
        return x, l0_norm
    
    def hard_concrete_sample(self, log_alpha):
        device = log_alpha.device
        # self.beta = self.beta.to(device) 
        random_noise = nn.Parameter(torch.rand(log_alpha.shape)).to(device)

        # Add small constant to the noise before taking the log to avoid NaNs
        gate_inputs = torch.log(random_noise + self.epsilon) - torch.log(1.0 - random_noise + self.epsilon)
        # gate_inputs = gate_inputs.to(device)
        gate_inputs = torch.sigmoid((gate_inputs + log_alpha) / self.beta)

        # Stretch the values
        stretched_values = gate_inputs * (self.zeta - self.gamma) + self.gamma

        # Clip the values to be in the range [0, 1]
        return torch.clamp(stretched_values, min=0.0, max=1.0)
    
    def hard_concrete_mean(self, log_alpha):
        gamma = self.gamma
        zeta = self.zeta
        stretched_values = torch.sigmoid(log_alpha) * (zeta - gamma) + gamma
        return torch.clamp(stretched_values, min=0.0, max=1.0)
    
    def l0_norm(self, log_alpha):
        beta = self.beta
        gamma = self.gamma
        zeta = self.zeta
        
        # Value of the CDF of the hard-concrete distribution evaluated at 0
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma / zeta))
        return reg_per_weight
    
    def var_train(self, weight_parameters):
        
        # theta is source_memory, log_alpha is linear transform of source_memory (source_pruning)
        theta, log_alpha = weight_parameters
        
        # Sample the z values from the hard-concrete distribution
        weight_noise = self.hard_concrete_sample(log_alpha)
        
        weights = theta * weight_noise
        
        return weights, weight_noise
    
    def var_eval(self,
        weight_parameters, pruning):
        theta = weight_parameters
        log_alpha = pruning
        # Use the mean of the learned hard-concrete distribution as the
        # deterministic weight noise at evaluation time
        weight_noise = self.hard_concrete_mean(log_alpha)
        
        # weight_noise = tf.Print(weight_noise, [weight_noise[0, :, 0]], message="mean activation", summarize=2512)
        weights = theta * weight_noise
        return weights, weight_noise
    
    def sparsify_inputs(self, asr_output):
        with torch.no_grad():
            asr_output = asr_output['encoder_out'][0]
            
            
            neuron_pruning = self.neuron_pruning
            source_memory, neuron_mask = self.var_eval(asr_output, neuron_pruning)
            
            
            source_pruning = self.source_pruning(asr_output)
            
            # Cut more aggresively
            # source_pruning -= 1.3 * (torch.mean(source_pruning))
            
            # Cut less aggresively
            # source_pruning += 1.3 * (torch.mean(source_pruning))
            
            source_memory, l0_mask = self.var_eval(source_memory, source_pruning)
            
            
            
            x, l0_mask, new_l0_mask, sorted_indices = self.temporal_selection(source_memory, l0_mask)
            
            
            return x, self.infer_out_seq_lengths(x), new_l0_mask, sorted_indices
            
    def temporal_selection(self, x, l0_mask):
        
        # x.shape: [seq_len, batch_size, features] = [88, 56, 512]
        seq_len, batch_size, hidden_size = x.size()
        
        # Sum across features, then find max k for any item in batch
        k_value = max(torch.sum(l0_mask, dim=(0, 2)).max().int().item(), 1)
        
        
        l0_squeeze = l0_mask.squeeze(-1)
        
        
        # Get top k indices for each item in batch
        top_k_indices = torch.topk(l0_mask.squeeze(-1), k_value, dim=0).indices  # [k, batch_size]
        
        sorted_indices, _ = torch.sort(top_k_indices, dim=0)
        
        # Select top k timesteps for each item in batch
        x_selected = x[sorted_indices, torch.arange(batch_size).unsqueeze(0), :]  # [k, batch_size, hidden_size]
        l0_mask_selected = l0_mask[sorted_indices, torch.arange(batch_size).unsqueeze(0), :]  # [k, batch_size, 1]
        
        # Create a new l0_mask with the same shape as the original, but with 0s for non-selected indices
        new_l0_mask = torch.zeros_like(l0_mask)
        
        # Use scatter to place the selected mask values in their correct positions
        new_l0_mask.scatter_(0, sorted_indices.unsqueeze(-1), l0_mask_selected)
        
        # print(f"Percentage features kept: {(k_value/seq_len) * 100}%")
        
        return x_selected, l0_mask_selected, new_l0_mask, sorted_indices
             
    def infer_out_seq_lengths(self, x, epsilon=1e-6):
        # x shape: [k, batch_size, features]
        norms = x.norm(dim=-1)  # [k, batch_size]
        mask = norms > epsilon
        seq_nums = mask.cumsum(0) * mask
        lengths = seq_nums.max(dim=0).values
        
        return lengths
            
