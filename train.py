import os
import warnings

from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb

from absl import app, flags
from ml_collections import config_flags
from tqdm import tqdm
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from ucell.frm import FRMWrapper

flags.DEFINE_string("resume", None, "resume a stopped run")
flags.DEFINE_string("dir", "checkpoints", "checkpointing location")
flags.DEFINE_string("init", None, "initalization weights")

_CONFIG = config_flags.DEFINE_config_file("config", "config.py")

try:
    num_nodes = int(os.environ['SLURM_NNODES'])
except KeyError:
    warnings.warn("Not running within a Slurm job. Assuming one node")
    num_nodes = 1

fabric = Fabric(precision="bf16-mixed", num_nodes=num_nodes)

@dataclass
class TrainState:
    model: Any
    optimizer: torch.optim.Optimizer|None
    carry: Any
    step: int
    metrics: "AverageMetric"
    ema_model: AveragedModel

class AverageMetric:
    def __init__(self):
        self.count = 0
        self.logs = {}
    
    def update(self, data:dict):
        if "count" in data:
            self.count += data.pop("count")

        for k,v in data.items():
            if not k in self.logs:
                self.logs[k] = v
            else:
                self.logs[k] += v
    
    def compute(self):
        return {k: v/self.count for k, v in self.logs.items()}

    def compute_and_reset(self):
        metrics = self.compute()
        self.count = 0
        self.logs = {}
        return metrics

    def __repr__(self):
        metrics = self.compute()
        return ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])


def create_dataloader(config, split: str):
    import data as data_module

    dataloader = data_module.get_dataloader(
        config, split
    )
    if dataloader is None:
        return None
    else:
        return fabric.setup_dataloaders(dataloader)


def cp_dir(logger:WandbLogger):
    if fabric.is_global_zero:
        run_name = str(logger.experiment.name)
    else:
        run_name = "_"

    run_name = fabric.broadcast(run_name, src=0)
    run_name = run_name.split("-")

    return Path(logger.save_dir) / "-".join(run_name[-1:] + run_name[:-1])


def save_state(train_state, logger:WandbLogger):
    cp = dict(
        step = train_state.step,
        model = train_state.model,
        ema_model = train_state.ema_model,
        optimizer = train_state.optimizer,        
    )

    save_path = cp_dir(logger) / f"cp_{train_state.step:08d}.pt"

    fabric.save(save_path, cp)


def resume_state(train_state, logger:WandbLogger):
    cp_file = sorted(list(cp_dir(logger).glob("cp*")))[-1]
    
    fabric.print(f"Loading checkpoint from {cp_file}")
 
    cp = fabric.load(cp_file)

    train_state.step = cp['step']
    # when LoRA adapters have been injected the checkpoint may not contain
    # their parameters (in case we are resuming a base model).  load with
    # strict=False in that situation to avoid missing-key errors.
    config = _CONFIG.value
    strict = not (hasattr(config, "lora") and config.lora.rank > 0)
    train_state.model.load_state_dict(cp['model'], strict=strict)
    train_state.ema_model.load_state_dict(cp['ema_model'])
    train_state.optimizer.load_state_dict(cp['optimizer'])

    return train_state


def setup_logging(config):
    logger = WandbLogger(
        project=config.name, 
        id=flags.FLAGS.resume, 
        save_dir=flags.FLAGS.dir,
        config=config.to_dict(),
    )

    fabric.print(f"Checkpoints saved in {cp_dir(logger)}")

    return logger


def setup(config):
    logger = setup_logging(config)

    with fabric.init_module():
        model = FRMWrapper(config)

    if flags.FLAGS.init is not None:
        model.load_checkpoint(flags.FLAGS.init)

    # if LoRA is requested, inject adapters before freezing any parameters
    if hasattr(config, "lora") and config.lora.rank > 0:
        from ucell.lora import inject_lora, mark_only_lora_trainable

        inject_lora(
            model,
            config.lora.rank,
            config.lora.alpha,
            config.lora.dropout,
            config.lora.target_modules,
        )
        # freeze everything except LoRA weights
        mark_only_lora_trainable(model)
        fabric.print(
            f"LoRA enabled (r={config.lora.rank}, alpha={config.lora.alpha}) "
            f"on modules {config.lora.target_modules or 'ALL'}"
        )

    if config.train_emb_only:
        if hasattr(config, "lora") and config.lora.rank > 0:
            fabric.print("WARNING: train_emb_only and LoRA are both set; "
                         "LoRA adapters will still be trained but non-task embeddings will be frozen.")
        for name, p in model.named_parameters():
            if not "task_emb" in name:
                p.requires_grad = False

    ema_model = AveragedModel(model, avg_fn=get_ema_avg_fn(config.ema_decay))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.opt.lr,
        weight_decay=config.opt.weight_decay,
        betas=(config.opt.beta1, config.opt.beta2)    
    )

    model, optimizer = fabric.setup(torch.compile(model), optimizer)

    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        carry=None,
        step=0,
        metrics=AverageMetric(),
        ema_model=ema_model,
    )

    if flags.FLAGS.resume:
        train_state = resume_state(train_state, logger)

    return train_state, logger


def train_batch(config, train_state, batch)->TrainState:
    train_state.step += 1

    # Init carry if it is None
    if train_state.carry is None:
        train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    outputs = train_state.model(carry=train_state.carry, batch=batch)
    train_state.carry, losses, metrics = outputs['carry'], outputs['losses'], outputs['metrics']

    loss = losses['det_loss'].mean() * config.det_loss_weight + losses['l2_loss'].mean()

    # loss.backward()
    fabric.backward(loss)

    # Apply optimizer
    optim = train_state.optimizer
    optim.step()
    optim.zero_grad()

    # Reduce metrics
    train_state.metrics.update(metrics)

    train_state.ema_model.update_parameters(train_state.model._original_module)

    return train_state


def evaluate(model, dataloader):
    model = model.eval()
    cur_metrics = AverageMetric()

    fabric.print("Evaluating ...")

    for batch in dataloader:
        with torch.device(fabric.device):
            carry = model.initial_carry(batch)  # type: ignore

        all_halted = False
        with torch.no_grad():
            while not all_halted:
                output = model(carry, batch)
                carry = output['carry']
                all_halted = output['carry'].halted.all()
        
        cur_metrics.update(output['metrics'])

    model.train()

    metrics = {k: fabric.all_reduce(v) for k, v in cur_metrics.compute().items()}

    fabric.print("Evaluation ... done")

    return metrics

def set_lr(train_state, dataset_size):
    config = _CONFIG.value

    factor = 1.

    if config.opt.cosine_annealing:
        total_steps = config.n_iters * dataset_size * config.halt_max_steps * fabric.world_size
        factor = 0.5 * (1 + np.cos(np.pi * train_state.step / total_steps))

    if train_state.step < config.opt.warmup_steps and flags.FLAGS.resume is None:
        factor *= train_state.step / config.opt.warmup_steps

    for gr in train_state.optimizer.param_groups:
        gr['lr'] = config.opt.lr * factor

    return config.opt.lr * factor


def run(_):
    config = _CONFIG.value
    print(config)

    fabric.seed_everything(config.seed)

    fabric.launch()

    train_state, logger = setup(config)

    train_data = create_dataloader(config, 'train')
    val_data = create_dataloader(config, 'validation')
    if val_data is None:
        warnings.warn("Cannot find validation dataset. Will skip validation")

    for i in range(config.n_iters):                
        train_state.metrics = AverageMetric()

        if fabric.is_global_zero:
            progress_bar = tqdm(total=len(train_data), desc=f"Iteration {i+1}")

        for k, batch in enumerate(train_data):
            cur_lr = set_lr(train_state, len(train_data))

            all_halted = False
            while not all_halted:
                train_state = train_batch(config, train_state, batch)
                all_halted = train_state.carry.halted.all()

            if fabric.is_global_zero:
                progress_bar.update(1)

            if (k+1) % 10 == 0:
                metrics = {k: fabric.all_reduce(v) for k, v in train_state.metrics.compute().items()}
                logger.log_metrics(dict(train=metrics, lr=cur_lr), step=train_state.step)

        save_state(train_state, logger)

        if val_data is not None:
            eval_metrics = evaluate(train_state.model, val_data)
            fabric.print(f"Eval metrics: {eval_metrics}")

            logger.log_metrics(dict(eval=eval_metrics), step=train_state.step)
            logger.save()
            

    if fabric.is_global_zero:
        cp_file = cp_dir(logger) / "final_ema.pt"
        torch.save(train_state.ema_model.module, cp_file)

        artifact = wandb.Artifact(name="model_"+ logger.experiment.name , type="model")
        artifact.add_file(cp_file, name="final_ema.pt")
        aliases = ["latest"]
        logger.experiment.log_model(artifact, aliases=aliases)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    app.run(run)
