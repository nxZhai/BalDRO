import os
from collections.abc import MutableMapping
from dataclasses import asdict, is_dataclass

import hydra
import mlflow
import wandb
from omegaconf import DictConfig, OmegaConf

from data import get_collators, get_data
from evals import get_evaluators
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything


def flatten(dictionary, parent_key="", separator="_"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def to_plain_dict(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, DictConfig):
        return OmegaConf.to_container(x, resolve=True)
    return x


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """

    wandb_config = {
        **to_plain_dict(cfg),
    }

    wandb_config_flat = flatten(wandb_config)
    if cfg.trainer.args.report_to == "wandb":
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "open-unlearning"),
            name=cfg.task_name,
            config=wandb_config_flat,
        )
        training(cfg)
    elif cfg.trainer.args.report_to == "mlflow":
        mlflow.set_experiment(os.getenv("WANDB_PROJECT", "open-unlearning"))

        with mlflow.start_run(run_name=cfg.task_name):
            mlflow.log_params(wandb_config_flat)
            training(cfg)


def training(cfg):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        if cfg.do_save:
            trainer.save_state()
            trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":

    main()
