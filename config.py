from ml_collections import ConfigDict

def base():
    config = ConfigDict()
    config.name = "ucell"
    config.seed = 42
    config.train_emb_only = False
    config.token = ""
    config.ema_decay = 0.999
    config.image_size = 256
    config.task_id = 0

    config.data_dir = "dataset"
    config.batch_size = 64 # per gpu. Total batch size is batch_size * num_gpus.
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
    config.model.L_cycles = 21

    config.opt = ConfigDict()
    config.opt.lr = 1e-4
    config.opt.cosine_annealing = False
    config.opt.weight_decay = 1.0
    config.opt.beta1 = 0.9
    config.opt.beta2 = 0.95
    config.opt.warmup_steps = 0

    # LoRA fine-tuning parameters
    config.lora = ConfigDict()
    config.lora.rank = 0           # if >0 enables LoRA adapters
    config.lora.alpha = 1.0        # scaling factor for LoRA updates
    config.lora.dropout = 0.0      # dropout applied to adapter input
    # substrings of module names that should receive LoRA (empty = all)
    config.lora.target_modules = ["SwiGLU"]

    return config


def get_config(cfg="default"):
    config = base()

    if cfg == "train_schedule_a" or cfg == "train":
        config.halt_max_steps=3
        config.model.H_cycles=1
        config.model.L_cycles=7
    elif cfg == "train_schedule_b":
        config.halt_max_steps=7
        config.model.H_cycles=1
        config.model.L_cycles=3
    elif cfg == "train_schedule_c":
        config.halt_max_steps=1
        config.model.H_cycles=3
        config.model.L_cycles=7
    elif cfg == "train_emb":
        config.halt_max_steps=1
        config.model.H_cycles=3
        config.model.L_cycles=7
        config.train_emb_only = True
        config.opt.weight_decay=0.0
        config.ema_decay=0.95
    else:
        assert cfg == 'default'

    return config
