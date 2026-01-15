import torch
from torch import nn

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import (
    compute_batch_nll,
    compute_forget_group,
    compute_retain_group,
    compute_satimp_loss,
)


class SatImp(GradDiff):
    def __init__(self, beta1=5.0, beta2=1.0, gamma=1.0, alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.alpha = alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
        )

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


class DrSatImp(SatImp):
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

    def compute_satimp_loss(self, model, inputs, beta1, beta2):
        outputs = model(**inputs)
        labels = inputs["labels"]
        labels = labels.to(outputs.logits.device)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        weight_sat = ((-lm_loss).exp().detach()) ** beta1
        weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
        forget_loss = -((weight_sat * weight_imp) * lm_loss)[
            shift_labels.view(-1) != -100
        ]

        return forget_loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_loss, forget_outputs = self.compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
        )

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
            retain_loss = retain_loss.mean()
        self.log({"retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss


class GroupSatImp(SatImp):
    def __init__(self, sampling_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampling_ratio = sampling_ratio

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        # * =====> compute Group
        forget_inputs, k, top_idx = compute_forget_group(
            model, forget_inputs, self.sampling_ratio
        )

        forget_loss, forget_outputs = compute_satimp_loss(
            model=model, inputs=forget_inputs, beta1=self.beta1, beta2=self.beta2
        )

        retain_inputs = inputs["retain"]

        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        retain_inputs = compute_retain_group(retain_inputs, k)

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # FEAT: log retain_loss and forget_loss
        self.log({"forget_loss": forget_loss.item(), "retain_loss": retain_loss.item()})

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
