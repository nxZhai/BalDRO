import math

import torch
import torch.nn.functional as F

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll, compute_forget_group, compute_retain_group


class SimNPO(GradDiff):
    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        self.log({"forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class DrSimNPO(SimNPO):
    def __init__(
        self,
        beta_dv_forget,
        beta_dv_retain,
        forget_dro=True,
        retain_dro=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta_dv_forget = beta_dv_forget
        self.beta_dv_retain = beta_dv_retain
        self.forget_dro = forget_dro
        self.retain_dro = retain_dro

    def _dv_aggregate(self, per_sample_loss, beta_dv):
        # Positive DV aggregation emphasizes samples with larger per-sample losses.
        return beta_dv * (
            torch.logsumexp(per_sample_loss / beta_dv, dim=0)
            - math.log(per_sample_loss.numel())
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_labels = forget_inputs["labels"]
        # Match compute_batch_nll: count only valid target tokens after shifting.
        loss_mask = forget_labels[..., 1:] != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1).clamp_min(1) - self.delta

        forget_loss = -F.logsigmoid(self.beta * forget_loss) * 2 / self.beta

        if self.forget_dro:
            # Larger SimNPO losses are harder to forget, so use positive DV.
            forget_loss = self._dv_aggregate(forget_loss, self.beta_dv_forget)
        else:
            forget_loss = forget_loss.mean()
        self.log({"forget_loss": forget_loss.item()})

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        if self.retain_dro:
            assert self.retain_loss_type == "NLL", (
                "DRO only supports NLL retain loss currently."
            )
            retain_loss, _ = compute_batch_nll(model, retain_inputs)
            # Normalize each retain sequence by its valid target-token count.
            retain_mask = retain_inputs["labels"][..., 1:] != -100
            retain_loss = retain_loss / retain_mask.sum(-1).clamp_min(1)
            retain_loss = self._dv_aggregate(retain_loss, self.beta_dv_retain)
        else:
            retain_loss = self.compute_retain_loss(
                model=model, retain_inputs=retain_inputs
            )
        self.log({"retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupSimNPO(SimNPO):
    def __init__(self, sampling_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampling_ratio = sampling_ratio

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_inputs = inputs["forget"]

        forget_inputs, k, top_idx = compute_forget_group(
            model, forget_inputs, self.sampling_ratio
        )

        forget_labels = forget_inputs["labels"][..., 1:].contiguous()
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        retain_inputs = inputs["retain"]

        retain_inputs = compute_retain_group(retain_inputs, k)

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        self.log({"forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
