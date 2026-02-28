import math
import torch
from torch import nn
from .layers import CastedLinear


class LoRALinear(nn.Module):
    """A drop-in replacement for a linear layer that adds a LoRA adapter.

    The original layer is kept in ``self.orig`` and left frozen during
    fine-tuning.  A pair of low-rank matrices (`lora_A`, `lora_B`) are
    learned instead, and their contribution is added to the output of the
    original layer.

    The behaviour is identical to the original layer when ``r == 0``.
    """

    def __init__(self, orig_layer: nn.Module, r: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()

        assert hasattr(orig_layer, "weight"), "LoRA can only wrap layers with a weight"
        self.orig = orig_layer
        self.r = r
        self.alpha = alpha
        self.register_buffer(
            "scaling",
            torch.tensor(alpha / r if r > 0 else 1.0, dtype=torch.float32),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if r > 0:
            in_features = orig_layer.weight.shape[1]
            out_features = orig_layer.weight.shape[0]

            # initialize LoRA parameters
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            # initialize following common LoRA practice: A is random, B zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            # when rank is zero we disable the adapter completely
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # forward through the original module first
        result = self.orig(x)
        if self.r > 0:
            lora_out = self.dropout(x) @ self.lora_A.t()  # (..., r)
            lora_out = lora_out @ self.lora_B.t()  # (..., out)
            result = result + lora_out * self.scaling.to(lora_out.dtype)
        return result


# helper utilities ---------------------------------------------------------

def inject_lora(module: nn.Module, r: int, alpha: float, dropout: float, target_modules=None):
    """Recursively replace appropriate linear modules with ``LoRALinear``.

    Args:
        module: root module to modify in place.
        r: LoRA rank.
        alpha: scaling factor.
        dropout: dropout probability applied to the adapter input.
        target_modules: iterable of strings; if provided, a submodule is only
            converted when one of the strings appears in its name.  By default
            all ``nn.Linear`` and ``CastedLinear`` layers are replaced.
    """
    if target_modules is None:
        target_modules = []

    for name, child in list(module.named_children()):
        full_name = f"{module.__class__.__name__}.{name}"
        convert = False
        if r > 0:
            # if target list specified, check membership
            if target_modules:
                for t in target_modules:
                    if t in name or t in full_name:
                        convert = True
                        break
            else:
                convert = True
        if convert and isinstance(child, (nn.Linear, CastedLinear)):
            new_child = LoRALinear(child, r, alpha, dropout)
            setattr(module, name, new_child)
        else:
            inject_lora(child, r, alpha, dropout, target_modules)


def mark_only_lora_trainable(module: nn.Module):
    """Freeze every parameter except those belonging to LoRA adapters."""
    for name, param in module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


# end of file
