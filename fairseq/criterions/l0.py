import torch

beta = 2.0 / 3.0
gamma = -0.1
zeta = 1.1
epsilon = 1e-8
    
class L0():
    
    def __init__(self) -> None:
        self.beta = torch.tensor(2.0 / 3.0)
        self.gamma = torch.tensor(-0.1)
        self.zeta = torch.tensor(1.1)
        self.epsilon = torch.tensor(1e-8)
    
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
        
    def l0_regularization_loss(self, l0_norm_loss, step,
                            reg_scalar=1.0,
                            start_reg_ramp_up=0,
                            end_reg_ramp_up=1000,
                            warm_up=True):

        # Ensure step is a float tensor
        step = torch.tensor(step, dtype=torch.float32)

        # Calculate the current step for ramp-up
        current_step_reg = torch.max(torch.tensor(0.0), step - start_reg_ramp_up)

        # Calculate the fraction of ramp-up completed
        fraction_ramp_up_completed = torch.min(current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), torch.tensor(1.0))

        if warm_up:
            # Regularizer intensifies over the course of ramp-up
            reg_scalar = fraction_ramp_up_completed * reg_scalar

        l0_norm_loss = reg_scalar * l0_norm_loss
        return l0_norm_loss

    def var_eval(self,
        weight_parameters):
        theta, log_alpha = weight_parameters

        # Use the mean of the learned hard-concrete distribution as the
        # deterministic weight noise at evaluation time
        weight_noise = self.hard_concrete_mean(log_alpha)
        # weight_noise = tf.Print(weight_noise, [weight_noise[0, :, 0]], message="mean activation", summarize=2512)
        weights = theta * weight_noise
        return weights, weight_noise

