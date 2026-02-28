## LoRA fine-tuning support

A lightweight rank‑decomposition adapter can be enabled via configuration to
fine-tune the model without modifying the full parameter set. Add the
following settings to your config file (or supply them in code):

```python
config.lora.rank = 4          # nonzero to activate LoRA
config.lora.alpha = 1.0        # scaling factor (alpha/r applied to adapter)
config.lora.dropout = 0.0      # optional dropout on adapter input
config.lora.target_modules = []  # substrings of module names to adapt
```

When `rank > 0` the training script will automatically inject `LoRALinear`
wrappers around the chosen `nn.Linear`/`CastedLinear` layers, freeze the base
weights, and only update the LoRA parameters.  This works with the existing
training loop and checkpointing.

The default configuration wraps _all_ linear layers; specify
`target_modules` (e.g. `['qkv_proj','o_proj']`) to limit which layers are
augmented.

Existing options such as `train_emb_only` are still honoured; if both flags
are set a warning is printed.

---
