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
    z_H: torch.Tensor    
    z_L: torch.Tensor    
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

        self.task_emb = CastedEmbedding(config.num_tasks, config.task_emb_len * config.hidden_size,
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

        embedding = torch.cat((task_embedding, embedding), dim=1)

        # Scale
        return math.sqrt(self.config.hidden_size) * embedding, shape_2d
    
    def forward(self, z_H, z_L, batch):
        # Input encoding
        input_embeddings, (h, w) = self._input_embeddings(batch["image"], batch["task_id"])

        D = self.config.task_emb_len

        def L_level(z_H, z_L):
            z = torch.concat([z_L, z_H], dim=1)
            for _ in range(self.config.L_cycles + 1):
                z = self.transformer(z + input_embeddings)
            z_L, z_H = z[:, :D], z[:, D:]
            return z_H, z_L

        # Forward iterations
        with torch.no_grad():
            for _ in range(self.config.H_cycles-1):
                z_H, z_L = L_level(z_H, z_L)
        z_H, z_L = L_level(z_H, z_L)

        # LM Outputs
        output = self.lm_head(rearrange(z_H, "b (h w) c -> b c h w", h=h, w=w))

        return (z_H, z_L), output

    def predict(self, inputs):
        B, H, W, C = inputs.shape
        
        image = torch.asarray(inputs, copy=True).to(torch.float32).permute(0, 3, 1, 2)
        batch = dict(image=image, task_id=torch.zeros([B], dtype=torch.int32))

        with torch.no_grad():
            z = torch.zeros([B, self.config.seq_len + self.config.task_emb_len, self.config.hidden_size])
            z_L, z_H = z[:, :self.config.task_emb_len], z[:, self.config.task_emb_len:]
            _, out = self(z_H, z_L, batch)
            out = out.permute(0, 2, 3, 1)

        return out.detach().cpu().numpy()


class FRMWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner = FRM(config.model)
        self.config = config

    def initial_carry(self, batch):
        model_cfg = self.config.model
        t_ = batch["task_id"]
        z_H = torch.tile(
            torch.zeros_like(t_[:, None, None], dtype=self.inner.dtype),
            (1, model_cfg.seq_len, model_cfg.hidden_size)
        )
        z_L = torch.tile(
            torch.zeros_like(t_[:, None, None], dtype=self.inner.dtype),
            (1, model_cfg.task_emb_len, model_cfg.hidden_size)
        )

        return FrmCarry(
            z_H = z_H,
            z_L = z_L,
            steps=torch.zeros_like(t_),
            halted=torch.zeros_like(t_, dtype=torch.bool), 
            current_data=batch,
        )

    def reset_carry(self, carry, batch):
        halted = carry.halted
        new_carry = self.initial_carry(batch)
        replace_halted = lambda x, y: torch.where(halted.view((-1,) + (1,) * (x.ndim - 1)) ,x, y)
        return FrmCarry(
            z_H = replace_halted(new_carry.z_H, carry.z_H),
            z_L = replace_halted(new_carry.z_L, carry.z_L),
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

        (z_H, z_L), output = self.inner(carry.z_H, carry.z_L, batch)

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

            l2_loss = .5 * torch.square(gt_flow - output['flow']).mean(dim=(1,2,3))

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
                z_H = z_H.detach(),
                z_L = z_L.detach(),
                steps = steps,
                halted = halted,
                current_data=batch,
            )
        
        return output
