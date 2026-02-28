import warnings
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from .layers import rms_norm, SwiGLU, Attention, CastedEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class FrmCarry:
    z: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict

class FRMBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, rope_max_pos=-1):
        super().__init__()
        self.attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            rope_max_pos=rope_max_pos,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=4,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states = rms_norm(hidden_states, variance_epsilon=1e-5)
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states, variance_epsilon=1e-5)
        return hidden_states


class FRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        dtype = getattr(torch, config.forward_dtype)
        ps = config.patch_size

        self.transformer = nn.Sequential(
            *[FRMBlock(config.hidden_size, config.num_heads, config.seq_len if config.pos_emb=="rope" else -1) 
              for _ in range(config.depth)]
        )

        embed_init_std = 1.0 / math.sqrt(config.hidden_size)

        self.patch_emb = nn.Conv2d(3, config.hidden_size, kernel_size=ps, stride=ps, padding=0)

        self.task_emb = CastedEmbedding(config.num_tasks, config.num_task_emb_tokens * config.hidden_size,
                                init_std=0, cast_to=dtype)
    
        if config.pos_emb == 'learned':
            self.pos_emb = CastedEmbedding(config.seq_len, config.hidden_size, init_std=embed_init_std, cast_to=dtype)
        else:
            assert config.pos_emb == "rope"

        self.lm_head = nn.ConvTranspose2d(config.hidden_size, 3, kernel_size=ps, stride=ps, padding=0)

        self.config = config
        self.dtype = dtype

    def _input_embeddings(self, image: torch.Tensor, task_id: torch.Tensor):
        # Token embedding
        embedding = self.patch_emb(image)
        shape_2d = embedding.shape[-2:]
        embedding = rearrange(embedding, "b c h w -> b (h w) c")

        # Position embeddings
        if self.config.pos_emb == "learned":
            embedding = 0.707106781 * (embedding + self.pos_emb.embedding_weight.to(self.dtype))

        # task embeddings
        task_embedding = self.task_emb(task_id)
        task_embedding = rearrange(task_embedding, "b (l d) -> b l d", d=self.config.hidden_size)
        num_addition_tokens = self.config.num_z_tokens - task_embedding.shape[1]
        task_embedding = F.pad(task_embedding, (0, 0, 0, num_addition_tokens), value=0)

        embedding = torch.cat((task_embedding, embedding), dim=1)

        # Scale
        return math.sqrt(self.config.hidden_size) * embedding, shape_2d
    
    def forward(self, z, batch):
        # Input encoding
        input_embeddings, (h, w) = self._input_embeddings(batch["image"], batch["task_id"])

        D = self.config.num_z_tokens

        def L_level(z):
            for _ in range(self.config.L_cycles):
                z = self.transformer(z + input_embeddings)
            return z

        # Forward iterations
        with torch.no_grad():
            for _ in range(self.config.H_cycles-1):
                z = L_level(z)
        z = L_level(z)

        # LM Outputs
        output = self.lm_head(rearrange(z[:, D:], "b (h w) c -> b c h w", h=h, w=w))

        return z, output

    def predict(self, inputs, task_id=0):
        B, H, W, C = inputs.shape
        
        image = torch.asarray(inputs, copy=True).to(torch.float32).permute(0, 3, 1, 2)
        batch = dict(image=image, task_id=torch.ones([B], dtype=torch.int32) * task_id)

        with torch.no_grad():
            z = torch.zeros([B, self.config.seq_len + self.config.num_z_tokens, self.config.hidden_size])
            _, out = self(z, batch)
            out = out.permute(0, 2, 3, 1)

        return out.detach().cpu().numpy()

    @staticmethod
    def _strip_state_dict_prefixes(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("inner."):
                k = k[len("inner."):]

            new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def _split_legacy_fused_layers(state_dict):
        """Split legacy fused gate_up_proj weights into separate gate_proj /
        up_proj tensors expected by the current SwiGLU architecture.
        This must run *after* LoRA merging so that any adapter contributions
        are already folded into the weight before the split.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if "gate_up_proj" in k:
                gate_weight, up_weight = v.chunk(2, dim=0)
                new_state_dict[k.replace("gate_up_proj", "gate_proj")] = gate_weight
                new_state_dict[k.replace("gate_up_proj", "up_proj")] = up_weight
            else:
                new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def _merge_lora_weights(state_dict, default_lora_scaling=None):
        merged_state_dict = {}
        lora_bases = set()

        for key in state_dict.keys():
            if key.endswith(".orig.weight"):
                base = key[:-len(".orig.weight")]
                if (
                    f"{base}.lora_A" in state_dict
                    and f"{base}.lora_B" in state_dict
                ):
                    lora_bases.add(base)

        for key, value in state_dict.items():
            if key.endswith(".orig.weight"):
                base = key[:-len(".orig.weight")]
                if base in lora_bases:
                    lora_a = state_dict[f"{base}.lora_A"]
                    lora_b = state_dict[f"{base}.lora_B"]

                    scaling_key = f"{base}.scaling"
                    alpha_key = f"{base}.alpha"
                    rank_key = f"{base}.r"
                    if scaling_key in state_dict:
                        scaling = state_dict[scaling_key]
                    elif alpha_key in state_dict and rank_key in state_dict:
                        scaling = state_dict[alpha_key] / state_dict[rank_key]
                    else:
                        if default_lora_scaling is not None:
                            scaling = default_lora_scaling
                        else:
                            raise ValueError(
                                "LoRA checkpoint is missing scaling metadata "
                                f"for '{base}'. Provide default_lora_scaling "
                                "when loading older checkpoints."
                            )

                    lora_a = lora_a.to(device=value.device, dtype=value.dtype)
                    lora_b = lora_b.to(device=value.device, dtype=value.dtype)
                    if torch.is_tensor(scaling):
                        scaling = scaling.to(device=value.device, dtype=value.dtype)

                    delta = torch.matmul(
                        lora_b,
                        lora_a,
                    )
                    merged_state_dict[f"{base}.weight"] = value + (delta * scaling)
                else:
                    merged_state_dict[f"{base}.weight"] = value
            elif key.endswith(".lora_A") or key.endswith(".lora_B"):
                continue
            elif key.endswith(".scaling") or key.endswith(".alpha") or key.endswith(".r"):
                continue
            else:
                merged_state_dict[key] = value

        if lora_bases:
            warnings.warn(
                "Detected LoRA checkpoint keys and merged adapters into base "
                "weights for loading.")

        return merged_state_dict

    
    def load_state_dict(self, state_dict, strict=True, default_lora_scaling=None):
        # remove unnecessary key prefixes (e.g., "module.") added by FrmWrapper, EMA or fabric
        new_state_dict = self._strip_state_dict_prefixes(state_dict)

        # merge LoRA adapter weights when loading checkpoints produced from LoRA training
        new_state_dict = self._merge_lora_weights(
            new_state_dict,
            default_lora_scaling=default_lora_scaling,
        )

        # split legacy fused gate_up_proj into gate_proj + up_proj (must be
        # after LoRA merging so adapters are already folded in)
        new_state_dict = self._split_legacy_fused_layers(new_state_dict)

        # if embedding dim mismatch, give a warning and load only the rest of the state
        model_state_dict = self.state_dict()
        for k in model_state_dict.keys():
            if 'task_emb' in k and k in new_state_dict and new_state_dict[k].shape != model_state_dict[k].shape:
                warnings.warn(f"Shape mismatch for key {k}, model shape {model_state_dict[k].shape}, checkpoint shape {new_state_dict[k].shape}. Skipping this key.")
                new_state_dict[k] = model_state_dict[k]

        super().load_state_dict(new_state_dict, strict=strict)


    def load_checkpoint(self, checkpoint, default_lora_scaling=None):
        # checkpoint can be a path or a path:key where key specifies which part of the checkpoint to load as model state dict
        if ":" in str(checkpoint):
            cp_path, key = str(checkpoint).split(":")
        else:
            cp_path, key = str(checkpoint), 'model'

        cp = torch.load(cp_path, weights_only=False, map_location="cpu")
        is_module_or_state_dict = False

        if isinstance(cp, torch.nn.Module):
            cp = cp.state_dict()
            is_module_or_state_dict = True
        elif isinstance(cp, dict) and all(isinstance(v, torch.Tensor) for v in cp.values()):
            is_module_or_state_dict = True
        
        if key in cp:
            cp = cp[key]
        else:
            if not is_module_or_state_dict:
                warnings.warn(f"Key '{key}' not found in checkpoint {cp_path}. Attempting to load entire checkpoint as model state dict.")

        self.load_state_dict(
            cp,
            strict=key != "ema_model",
            default_lora_scaling=default_lora_scaling,
        )


class FRMWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = FRM(config.model)
        self.config = config

    def initial_carry(self, batch):
        model_cfg = self.config.model
        t_ = batch["task_id"]
        z = torch.tile(
            torch.zeros_like(t_[:, None, None], dtype=self.inner.dtype),
            (1, model_cfg.seq_len + model_cfg.num_z_tokens, model_cfg.hidden_size)
        )

        return FrmCarry(
            z = z,
            steps=torch.zeros_like(t_),
            halted=torch.zeros_like(t_, dtype=torch.bool), 
            current_data=batch,
        )

    def reset_carry(self, carry, batch):
        halted = carry.halted
        new_carry = self.initial_carry(batch)
        replace_halted = lambda x, y: torch.where(halted.view((-1,) + (1,) * (x.ndim - 1)) ,x, y)
        return FrmCarry(
            z = replace_halted(new_carry.z, carry.z),
            steps = replace_halted(new_carry.steps, carry.steps),
            halted = torch.zeros_like(halted),
            current_data = {
                k: replace_halted(v, carry.current_data[k])
                for k, v in batch.items()
            }
        )

    def forward(self, carry, batch):
        config = self.config

        carry = self.reset_carry(carry, batch)
        batch = carry.current_data

        z, output = self.inner(carry.z, batch)

        output = dict(flow = output[:, :2], mask = output[:, 2])

        steps = carry.steps  + 1

        halted = (steps >= config.halt_max_steps)
        
        # losses
        if "label" in batch:
            label = batch['label']
            gt_flow, gt_mask = label[:, :2], label[:, 2]

            det_loss = F.binary_cross_entropy_with_logits(
                output['mask'], gt_mask, reduction="none",
            ).mean(dim=(1,2))

            l2_loss = .5 * torch.square(gt_flow - output['flow']) * (gt_flow != 0).any(dim=1, keepdim=True)
            l2_loss = l2_loss.mean(dim=(1,2,3))

            output['losses'] = dict(det_loss=det_loss, l2_loss=l2_loss)

            with torch.no_grad():
                if self.training:
                    halt_signal = l2_loss < config.halt_threshold
                    should_explore = torch.rand_like(l2_loss) < config.halt_exploration_prob
                    halted |= halt_signal & ~should_explore & (steps >= min(config.halt_min_steps, config.halt_max_steps))

                # Metrics (halted)
                output['metrics'] = {
                    "count": halted.sum(),
                    "steps": torch.where(halted, steps, 0).sum(),
                    "l2_loss": torch.where(halted, l2_loss, 0).sum(),
                    "det_loss": torch.where(halted, det_loss, 0).sum(),
                }

        output['carry'] = FrmCarry(
            z = z.detach(),
            steps = steps,
            halted = halted,
            current_data=batch,
        )
        
        return output

    def load_checkpoint(self, checkpoint, default_lora_scaling=None):
        if default_lora_scaling is None:
            if hasattr(self.config, "lora") and self.config.lora.rank > 0:
                default_lora_scaling = (
                    self.config.lora.alpha / self.config.lora.rank
                )
        self.inner.load_checkpoint(
            checkpoint,
            default_lora_scaling=default_lora_scaling,
        )
