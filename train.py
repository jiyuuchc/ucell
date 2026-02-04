from absl import app, flags
from typing import Any
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import wandb
# from adam_atan2 import AdamATan2
from ml_collections import config_flags
from tqdm import tqdm
from ucell.frm import FRMWrapper
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

flags.DEFINE_string("resume", None, "resume a stopped run")
flags.DEFINE_string("dir", "checkpoints", "checkpointing location")
flags.DEFINE_string("init", None, "initalization weights")

_CONFIG = config_flags.DEFINE_config_file("config", "config.py")
fabric = Fabric(precision="bf16-mixed")

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
    data_fn = getattr(data_module, config.dataset)

    dataloader = data_fn(
        config, split
    )
    return fabric.setup_dataloaders(dataloader)


def cp_dir(logger:WandbLogger):
    run_name = str(logger.experiment.name)
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
    if fabric.is_global_zero:
        cp_file = sorted(list(cp_dir(logger).glob("*.pt")))[-1]
    else:
        cp_file = None
    
    cp = fabric.load(cp_file)

    train_state.step = cp['step']
    train_state.model.load_state_dict(cp['model'])
    train_state.ema_model.load_state_dict(cp['ema_model'])
    train_state.optimizer.load_state_dict(cp['optimizer'])

    return train_state


def setup_logging(config):
    logger = WandbLogger(
        project=config.name, 
        id=flags.FLAGS.resume, 
        dir=flags.FLAGS.dir,
        config=config.to_dict(),
    )

    return logger


def setup(config):
    logger = setup_logging(config)

    model = FRMWrapper(config)
    if flags.FLAGS.init is not None:
        cp = torch.load(flags.FLAGS.init, weights_only=True)
        if 'model' in cp:
            cp = cp['model']
        model.load_state_dict(cp, strict=False)

    ema_model = AveragedModel(model, avg_fn=get_ema_avg_fn(config.ema_decay))
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    train_state.step += fabric.world_size

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


def run(_):
    config = _CONFIG.value

    train_state, logger = setup(config)

    train_data = create_dataloader(config, 'train')
    test_data = create_dataloader(config, 'test')

    for i in range(config.n_iters):
        train_state.metrics = AverageMetric()

        if fabric.is_global_zero:
            progress_bar = tqdm(total=len(train_data), desc=f"Iteration {i+1}")

        for k, batch in enumerate(train_data):
            all_halted = False
            while not all_halted:
                train_state = train_batch(config, train_state, batch)
                all_halted = train_state.carry.halted.all()

            if fabric.is_global_zero:
                progress_bar.update(1)

            if k % 10 == 0:
                metrics = {k: fabric.all_reduce(v) for k, v in train_state.metrics.compute().items()}
                logger.log_metrics(dict(train=metrics), step=train_state.step)

        save_state(train_state, logger)

        eval_metrics = evaluate(train_state.model, test_data)

        logger.log_metrics(dict(eval=eval_metrics), step=train_state.step)
        logger.save()

    if fabric.is_global_zero:
        cp_file = cp_dir(logger) / "final_ema.pt"
        torch.save(train_state.ema_model.module, cp_file)

        artifact = wandb.Artifact(name="model", type="model")
        artifact.add_file(cp_file, name="final_ema.pt")
        aliases = ["latest"]
        logger.experiment.log_model(artifact, aliases=aliases)

if __name__ == "__main__":
    fabric.launch()
    app.run(run)
