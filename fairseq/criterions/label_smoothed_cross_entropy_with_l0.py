import math
import torch
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.criterions.l0 import L0
from fairseq import utils
from fairseq.logging import metrics

@register_criterion("label_smoothed_cross_entropy_with_l0")
class AsrAfsLoss(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, 
                 l0_norm_reg_scalar=1.0, l0_norm_start_reg_ramp_up=0, 
                 l0_norm_end_reg_ramp_up=100000, l0_norm_warm_up=True):
        super().__init__(task, sentence_avg, label_smoothing)
        self.l0_norm_reg_scalar = l0_norm_reg_scalar
        self.l0_norm_start_reg_ramp_up = l0_norm_start_reg_ramp_up
        self.l0_norm_end_reg_ramp_up = l0_norm_end_reg_ramp_up
        self.l0_norm_warm_up = l0_norm_warm_up
        self.beta = torch.tensor(2.0 / 3.0)
        self.gamma = torch.tensor(-0.1)
        self.zeta = torch.tensor(1.1)
        self.epsilon = torch.tensor(1e-8)
    

    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--l0-norm-reg-scalar', type=float, default=1.0)
        parser.add_argument('--l0-norm-start-reg-ramp-up', type=int, default=0)
        parser.add_argument('--l0-norm-end-reg-ramp-up', type=int, default=100000)
        parser.add_argument('--l0-norm-warm-up', type=bool, default=True)

    def forward(self, model, sample, reduce=True):
        net_output, l0_norm = model(**sample['net_input'])
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)
        
        # Apply L0 regularization
        num_updates = model.num_updates
        l0_reg = self.l0_regularization_loss(
            l0_norm,
            step=num_updates,
            reg_scalar=self.l0_norm_reg_scalar,
            start_reg_ramp_up=self.l0_norm_start_reg_ramp_up,
            end_reg_ramp_up=self.l0_norm_end_reg_ramp_up,
            warm_up=self.l0_norm_warm_up
        )

        total_loss = loss + l0_reg

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'l0_reg': utils.item(l0_reg.data) if reduce else l0_reg.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return total_loss, sample_size, logging_output
    
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
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        loss_sum = sum(log.get("l0_reg", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "l0_reg", loss_sum / sample_size / math.log(2), sample_size, round=3
        )