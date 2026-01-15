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
        sigma_forget,
        sigma_retain,
        forget_dro=True,
        retain_dro=False,
        log_ori_loss=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.sigma_forget = sigma_forget
        self.sigma_retain = sigma_retain
        self.forget_dro = forget_dro
        self.retain_dro = retain_dro
        self.log_ori_loss = log_ori_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / loss_mask.sum(-1) - self.delta

        forget_loss = -F.logsigmoid(self.beta * forget_loss) * 2 / self.beta
        
        if self.forget_dro:
            if self.log_ori_loss:
                self.log(
                    {"forget_loss_ori": forget_loss.clone().detach().mean().item()}
                )
            forget_loss = -self.sigma_forget * torch.log(
                torch.mean(torch.exp(-forget_loss / self.sigma_forget))
            )
        else:
            forget_loss = forget_loss.mean()
        self.log({"forget_loss": forget_loss.item()})

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        assert (
            self.retain_loss_type == "NLL"
        ), "DRO only supports NLL retain loss currently."
        retain_loss, _ = compute_batch_nll(model, retain_inputs)
        if self.retain_dro:
            if self.log_ori_loss:
                self.log(
                    {
                        "retain_loss_ori": retain_loss.clone().detach().mean().item(),
                    }
                )
            retain_loss = -self.sigma_retain * torch.log(
                torch.mean(torch.exp(-retain_loss / self.sigma_retain))
            )
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
