import torch
import torch.nn as nn
from fairseq.criterions.l0 import L0

class AdaptiveFeatureSelection(nn.Module):
    def __init__(self, input_dim, training=True, dropout=0.1, enable_afs_t=True, enable_afs_f=True):
        super().__init__()
        self.input_dim = input_dim
        self.enable_afs_t = enable_afs_t
        self.enable_afs_f = enable_afs_f
        self.dropout = nn.Dropout(dropout)
        self.training = training
        self.beta = torch.tensor(2.0 / 3.0)
        self.gamma = torch.tensor(-0.1)
        self.zeta = torch.tensor(1.1)
        self.epsilon = torch.tensor(1e-8)

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
                x, weight_noise = self.var_eval((x, source_pruning))

        if self.enable_afs_f:
            if self.training:
                x, neuron_noise = self.var_train((x, self.neuron_pruning))
                l0_norm += self.l0_norm(self.neuron_pruning).sum()
            else:
                x, neuron_noise = self.var_eval((x, self.neuron_pruning))

        x = self.dropout(x)
        return x, l0_norm
    
    def hard_concrete_sample(self, log_alpha):
        device = log_alpha.device
        # self.beta = self.beta.to(device) 
        random_noise = torch.rand(log_alpha.shape)

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
            # neuron pruning
            asr_output = asr_output['encoder_out'][0]
            neuron_pruning = self.neuron_pruning
            source_memory, neuron_mask = self.var_eval(asr_output, neuron_pruning)
            
            # temporal pruning
            source_pruning = self.source_pruning(asr_output)
            source_memory, l0_mask = self.var_eval(source_memory, source_pruning)
            
            #TODO: Check dimensions of stuff here
            x, l0_mask = self.temporal_selection(source_memory, l0_mask)
            
            return x, self.infer_out_seq_lengths(x)
            
    def temporal_selection(self, x, l0_mask):
        
        # Possible dimension casting of things here
        
        # determine k_value 
        k_value = torch.max(torch.sum(l0_mask, dim=1)).int().item()
        k_value = max(k_value, 1)
        
        # get top_k elements
        top_k_indices = torch.topk(l0_mask, k_value, dim=1).indices.squeeze(-1)
        
        # Gather selected features
        batch_size, seq_len, hidden_size = x.size()
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k_value)
        x_selected = x[batch_indices, top_k_indices]
        l0_mask_selected = l0_mask[batch_indices, top_k_indices]
        
        return x_selected, l0_mask_selected
             
    def infer_out_seq_lengths(self, x, epsilon=1e-6):
        # Assume x is of shape T x B x (C x D)
        # Compute the norm along the feature dimension
        norms = x.norm(dim=-1)  # T x B
        
        # Find the last non-zero (or above epsilon) element for each sequence
        mask = norms > epsilon  # T x B
        
        # Use cumsum to create a sequence of increasing numbers, masked
        seq_nums = mask.cumsum(0) * mask  # T x B
        
        # Get the maximum sequence number for each batch item
        lengths = seq_nums.max(dim=0).values
        
        return lengths
            
