"""Tests for FRM checkpoint loading paths:
    1. Plain base-model Fabric dict checkpoint
    2. Serialised module checkpoint (final_ema.pt style)
    3. LoRA-trained checkpoint (adapters merged at load time)
    4. Legacy fused gate_up_proj checkpoint (split at load time)
    5. Legacy fused gate_up_proj + LoRA (merge then split)
"""

import copy
import warnings

import pytest
import torch

from ucell.frm import FRM, FRMWrapper
from ucell.lora import inject_lora
from ml_collections import ConfigDict


# ---------------------------------------------------------------------------
# Minimal config that keeps the model tiny for fast testing
# ---------------------------------------------------------------------------

def _small_config():
    cfg = ConfigDict()
    cfg.patch_size = 8
    cfg.forward_dtype = "float32"
    cfg.pos_emb = "rope"
    cfg.hidden_size = 64
    cfg.num_z_tokens = 4
    cfg.num_task_emb_tokens = 4
    cfg.depth = 1
    cfg.num_heads = 1
    cfg.num_tasks = 4
    cfg.seq_len = 16   # (32/8)^2
    cfg.H_cycles = 1
    cfg.L_cycles = 1
    return cfg


def _make_model():
    return FRM(_small_config())


def _small_wrapper_config():
    cfg = ConfigDict()
    cfg.model = _small_config()
    cfg.halt_max_steps = 1
    cfg.halt_min_steps = 1
    cfg.halt_exploration_prob = 0.0
    cfg.halt_threshold = 0.0
    cfg.lora = ConfigDict()
    cfg.lora.rank = 2
    cfg.lora.alpha = 8.0
    cfg.lora.dropout = 0.0
    cfg.lora.target_modules = ["SwiGLU"]
    return cfg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _round_trip(model_a, state_dict, strict=True):
    """Load state_dict into a fresh model and return it."""
    model_b = _make_model()
    model_b.load_state_dict(state_dict, strict=strict)
    return model_b


def _weights_equal(m1, m2):
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        if not torch.allclose(p1.float(), p2.float()):
            return False, n
    return True, None


# ---------------------------------------------------------------------------
# 1. Plain base-model round-trip (dict checkpoint)
# ---------------------------------------------------------------------------

def test_base_dict_checkpoint():
    model = _make_model()
    sd = {"model": model.state_dict()}

    model2 = _make_model()
    model2.load_state_dict(sd["model"])

    ok, bad = _weights_equal(model, model2)
    assert ok, f"Weight mismatch at {bad}"


# ---------------------------------------------------------------------------
# 2. Serialised module object (final_ema.pt style)
# ---------------------------------------------------------------------------

def test_module_object_checkpoint(tmp_path):
    model = _make_model()
    pt = tmp_path / "final_ema.pt"
    torch.save(model, pt)

    model2 = _make_model()
    model2.load_checkpoint(str(pt))

    ok, bad = _weights_equal(model, model2)
    assert ok, f"Weight mismatch at {bad}"


# ---------------------------------------------------------------------------
# 3. LoRA checkpoint — adapters merged into base weights
# ---------------------------------------------------------------------------

def test_lora_checkpoint_merge():
    model = _make_model()

    # inject LoRA and do a fake "forward then save"
    lora_model = copy.deepcopy(model)
    inject_lora(
        lora_model,
        r=2,
        alpha=2.0,
        dropout=0.0,
        target_modules=["SwiGLU"],
    )

    # manually set non-zero lora_B so the merge actually changes the weight
    for name, param in lora_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)

    lora_sd = lora_model.state_dict()

    # confirm checkpoint has LoRA keys
    assert any("lora_A" in k for k in lora_sd), "Test setup: expected lora_A keys"
    assert any("orig.weight" in k for k in lora_sd), "Test setup: expected orig.weight keys"

    # load into a plain (non-LoRA) FRM — should not raise
    model2 = _make_model()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model2.load_state_dict(lora_sd)
        assert any("LoRA" in str(x.message) for x in w), "Expected LoRA merge warning"

    # merged weights must differ from base (because lora_B is non-zero)
    # at least one parameter should differ after merge
    all_same = all(
        torch.allclose(p1.float(), p2.float())
        for p1, p2 in zip(model.parameters(), model2.parameters())
    )
    assert not all_same, "Merged weights should differ from original base weights"

    # no LoRA keys should remain in model2's state dict
    sd2 = model2.state_dict()
    assert not any("lora_A" in k or "lora_B" in k or "orig.weight" in k for k in sd2), \
        "LoRA keys leaked into loaded model state dict"


# ---------------------------------------------------------------------------
# 4. Legacy gate_up_proj checkpoint
# ---------------------------------------------------------------------------

def test_legacy_gate_up_proj():
    model = _make_model()
    sd = model.state_dict()

    # Build a legacy state dict by fusing gate_proj + up_proj
    legacy_sd = {}
    for k, v in sd.items():
        if "gate_proj" in k:
            up_key = k.replace("gate_proj", "up_proj")
            fused_key = k.replace("gate_proj", "gate_up_proj")
            legacy_sd[fused_key] = torch.cat([v, sd[up_key]], dim=0)
        elif "up_proj" in k:
            continue
        else:
            legacy_sd[k] = v

    assert any("gate_up_proj" in k for k in legacy_sd)

    model2 = _make_model()
    model2.load_state_dict(legacy_sd)

    ok, bad = _weights_equal(model, model2)
    assert ok, f"Weight mismatch at {bad} after legacy split"


# ---------------------------------------------------------------------------
# 5. Legacy gate_up_proj + LoRA checkpoint
# ---------------------------------------------------------------------------

def test_legacy_gate_up_proj_with_lora():
    model = _make_model()
    lora_model = copy.deepcopy(model)
    inject_lora(
        lora_model,
        r=2,
        alpha=2.0,
        dropout=0.0,
        target_modules=["SwiGLU"],
    )

    for name, param in lora_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)

    lora_sd = lora_model.state_dict()

    # Convert LoRA checkpoint into legacy fused format by replacing
    # gate_proj/up_proj keys with gate_up_proj counterparts.
    legacy_lora_sd = {}
    mlp_prefixes = set()
    for key in lora_sd:
        if ".mlp.gate_proj." in key:
            mlp_prefixes.add(key.split(".gate_proj.")[0])

    for key, value in lora_sd.items():
        if ".mlp.gate_proj." in key or ".mlp.up_proj." in key:
            continue
        legacy_lora_sd[key] = value

    for prefix in mlp_prefixes:
        gate_orig_key = f"{prefix}.gate_proj.orig.weight"
        up_orig_key = f"{prefix}.up_proj.orig.weight"
        gate_a_key = f"{prefix}.gate_proj.lora_A"
        up_a_key = f"{prefix}.up_proj.lora_A"
        gate_b_key = f"{prefix}.gate_proj.lora_B"
        up_b_key = f"{prefix}.up_proj.lora_B"
        gate_scaling_key = f"{prefix}.gate_proj.scaling"
        up_scaling_key = f"{prefix}.up_proj.scaling"

        if gate_orig_key in lora_sd and up_orig_key in lora_sd:
            legacy_lora_sd[f"{prefix}.gate_up_proj.orig.weight"] = torch.cat(
                [lora_sd[gate_orig_key], lora_sd[up_orig_key]],
                dim=0,
            )

        # Build a shape-valid fused LoRA form for legacy gate_up_proj:
        # - A: keep gate branch A (r, in)
        # - B: stack gate/up B along output dimension (2*out, r)
        if (
            gate_a_key in lora_sd
            and up_a_key in lora_sd
            and gate_b_key in lora_sd
            and up_b_key in lora_sd
        ):
            legacy_lora_sd[f"{prefix}.gate_up_proj.lora_A"] = lora_sd[gate_a_key]
            legacy_lora_sd[f"{prefix}.gate_up_proj.lora_B"] = torch.cat(
                [lora_sd[gate_b_key], lora_sd[up_b_key]],
                dim=0,
            )

        if gate_scaling_key in lora_sd and up_scaling_key in lora_sd:
            assert torch.allclose(
                lora_sd[gate_scaling_key],
                lora_sd[up_scaling_key],
            )
            legacy_lora_sd[f"{prefix}.gate_up_proj.scaling"] = lora_sd[
                gate_scaling_key
            ]

    assert any("gate_up_proj" in k for k in legacy_lora_sd)

    model2 = _make_model()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model2.load_state_dict(legacy_lora_sd)

    sd2 = model2.state_dict()
    assert not any("lora_A" in k or "gate_up_proj" in k for k in sd2)


def test_lora_functional_output_matches_merged_reload(tmp_path):
    torch.manual_seed(0)

    config = _small_wrapper_config()
    lora_model = FRMWrapper(config).eval()
    inject_lora(
        lora_model,
        r=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    for name, param in lora_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)

    batch = {
        "image": torch.randn(2, 3, 32, 32),
        "task_id": torch.tensor([0, 1], dtype=torch.int32),
    }
    z_h = torch.zeros(
        2,
        config.model.seq_len,
        config.model.hidden_size,
    )
    z_l = torch.zeros(
        2,
        config.model.num_z_tokens,
        config.model.hidden_size,
    )

    with torch.no_grad():
        _, expected = lora_model.inner(z_h, z_l, batch)

    # Simulate an older LoRA checkpoint that did not persist scaling.
    cp = {
        "model": {
            k: v
            for k, v in lora_model.state_dict().items()
            if not k.endswith(".scaling")
        }
    }
    cp_path = tmp_path / "old_lora_checkpoint.pt"
    torch.save(cp, cp_path)

    reloaded = FRMWrapper(config).eval()
    reloaded.load_checkpoint(str(cp_path))

    with torch.no_grad():
        _, actual = reloaded.inner(z_h, z_l, batch)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
        "Merged-reload output does not match injected LoRA output"
    )


def test_old_lora_checkpoint_without_scaling_requires_default(tmp_path):
    config = _small_wrapper_config()
    lora_model = FRMWrapper(config).eval()
    inject_lora(
        lora_model,
        r=config.lora.rank,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
    )

    cp = {
        "model": {
            k: v
            for k, v in lora_model.state_dict().items()
            if not k.endswith(".scaling")
        }
    }
    cp_path = tmp_path / "old_lora_checkpoint_missing_scaling.pt"
    torch.save(cp, cp_path)

    bare_config = ConfigDict()
    bare_config.model = _small_config()
    bare_config.halt_max_steps = 1
    bare_config.halt_min_steps = 1
    bare_config.halt_exploration_prob = 0.0
    bare_config.halt_threshold = 0.0

    reloaded = FRMWrapper(bare_config).eval()

    with pytest.raises(ValueError, match="missing scaling metadata"):
        reloaded.load_checkpoint(str(cp_path))
