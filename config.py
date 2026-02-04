from ml_collections import ConfigDict

def base():
    config = ConfigDict()
    config.name = "ucell"
    config.seed = 42
    config.token = ""
    config.ema_decay = 0.999

    config.dataset = "scs"
    config.batch_size = 48
    config.epochs_per_iter = 50
    config.n_iters = 16
    config.dataloader_workers = config.get("batch_size") // 8

    config.halt_max_steps = 1
    config.halt_min_steps = 1
    config.halt_exploration_prob = 0.5
    config.halt_threshold = 0.
    config.det_loss_weight = 0.1

    config.model = ConfigDict()
    config.model.patch_size = 8
    config.model.forward_dtype="bfloat16"
    config.model.pos_emb="learned"
    config.model.hidden_size = 1024
    config.model.task_emb_len = 256
    config.model.depth = 2
    config.model.num_heads = config.model.get_ref('hidden_size')//64
    config.model.num_tasks = 26
    config.model.seq_len = 1024 * 64 // (config.model.get_ref('patch_size') ** 2)
    config.model.H_cycles = 3
    config.model.L_cycles = 6

    config.opt = ConfigDict()
    config.opt.lr = 1e-4
    config.opt.weight_decay = 1.0
    config.opt.beta1 = 0.9
    config.opt.beta2 = 0.95

    return config


def get_config(cfg="default"):
    config = base()

    if cfg == "ms":
        config.halt_max_steps=3
        config.model.H_cycles=1
    else:
        assert cfg == 'default'

    return config
