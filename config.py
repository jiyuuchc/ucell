import random
from ml_collections import ConfigDict

def base():
    config = ConfigDict()
    config.name = "ucell"
    config.token = ""
    config.ema_decay = 0.999
    config.image_size = 256
    config.train_emb_only = False

    config.seed = random.randint(0, 1000000)

    config.dataset = "scs"
    config.batch_size = 48
    config.epochs_per_iter = 25
    config.n_iters = 16
    config.dataloader_workers = config.get_ref("batch_size") // 8

    config.halt_max_steps = 1
    config.halt_min_steps = 1
    config.halt_exploration_prob = 0.5
    config.halt_threshold = 0.
    config.det_loss_weight = 0.1

    config.model = ConfigDict()
    config.model.patch_size = 8
    config.model.forward_dtype="bfloat16"
    config.model.pos_emb="rope"
    config.model.hidden_size = 1024
    config.model.num_z_tokens = 64
    config.model.num_task_emb_tokens = 64
    config.model.depth = 2
    config.model.num_heads = config.model.get_ref('hidden_size')//64
    config.model.num_tasks = 26
    config.model.seq_len = (config.get_ref("image_size") // config.model.get_ref('patch_size')) ** 2
    config.model.H_cycles = 1
    config.model.L_cycles = 20

    config.opt = ConfigDict()
    config.opt.lr = 1e-4
    config.opt.cosine_annealing = False
    config.opt.weight_decay = 1.0
    config.opt.beta1 = 0.9
    config.opt.beta2 = 0.95
    config.opt.warmup_steps = 2000

    return config


def get_config(cfg="default"):
    config = base()

    if cfg == "train":
        config.halt_max_steps=3
        config.model.H_cycles=1
        config.model.L_cycles=6
    elif cfg == "train_b":
        config.halt_max_steps=7
        config.model.H_cycles=1
        config.model.L_cycles=2
    elif cfg == "train_emb":
        config.halt_max_steps=3
        config.model.H_cycles=1
        config.model.L_cycles=6
        config.train_emb_only = True
        config.opt.lr = 1e-3
        config.opt.cosine_annealing = True
        config.n_iters = 1
    else:
        assert cfg == 'default'

    return config
