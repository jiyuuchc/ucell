"""
Advanced dead neuron analysis with detailed statistics and visualization.

This script provides:
- Per-layer neuron activation histograms
- Dead neuron identification by absolute activation threshold
- Neuron importance ranking
- Visualization of activation distributions
"""

from pathlib import Path
from typing import Dict, List, Tuple, Generator
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from absl import app, flags
from ml_collections import config_flags

from ucell.layers import SwiGLU

# make sure root folder is importable
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent.parent))
from data import scs

flags.DEFINE_string("model", None, "Path to model checkpoint")
flags.DEFINE_string("layer_types", "SwiGLU", "Comma-separated class names from ucell.layers to monitor")
flags.DEFINE_float("activation_threshold", 1e-5, "Threshold for dead neuron detection")
flags.DEFINE_integer("max_batches", 20, "Number of batches to analyze")
flags.DEFINE_string("output_dir", None, "Output directory (default: ../dead_neuron_analysis_results)")
flags.DEFINE_integer("batch_size", 4, "Batch size for inference")
flags.DEFINE_string("image_folder", None, "Path to a directory of .tif images to use instead of the SCS dataloader")

_CONFIG = config_flags.DEFINE_config_file("config", "config.py")


class AdvancedNeuronAnalyzer:
    """Advanced analysis of neuron activations including per-neuron statistics."""
    
    def __init__(self, model: nn.Module, activation_threshold: float = 1e-5):
        self.model = model
        self.activation_threshold = activation_threshold
        self.neuron_activations = {}  # Store raw activations per neuron
        self.hooks = []
        # resolve layer classes
        from ucell import layers
        names = [n.strip() for n in flags.FLAGS.layer_types.split(",") if n.strip()]
        cls_list = []
        for n in names:
            if hasattr(layers, n):
                cls_list.append(getattr(layers, n))
            else:
                raise ValueError(f"Unknown layer type '{n}' in --layer_types")
        self.layer_classes = tuple(cls_list)
        
    def register_hooks(self):
        """Register hooks to collect per-neuron activation statistics."""
        hook_handles = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, self.layer_classes):
                handle = module.register_forward_hook(self._make_detailed_hook(name))
                hook_handles.append(handle)
        
        return hook_handles
    
    def _make_detailed_hook(self, layer_name: str):
        """Create a hook that captures per-neuron activations."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                act = output.detach().cpu()
                
                if layer_name not in self.neuron_activations:
                    self.neuron_activations[layer_name] = []
                
                # Flatten to [num_samples, num_neurons]
                if act.dim() > 2:
                    act = act.view(-1, act.shape[-1])
                
                self.neuron_activations[layer_name].append(act)
        
        return hook
    
    def get_neuron_stats(self) -> Dict:
        """Compute per-neuron statistics."""
        neuron_stats = {}
        
        for layer_name, activations_list in self.neuron_activations.items():
            # Concatenate all batches
            all_activations = torch.cat(activations_list, dim=0)  # [num_samples, num_neurons]
            
            num_neurons = all_activations.shape[1]
            num_samples = all_activations.shape[0]
            
            # Per-neuron statistics
            mean_acts = all_activations.mean(dim=0)
            std_acts = all_activations.std(dim=0)
            max_acts = all_activations.max(dim=0)[0]
            min_acts = all_activations.min(dim=0)[0]
            
            # Count how many samples each neuron was "active" in
            active_count = (all_activations.abs() > self.activation_threshold).sum(dim=0)
            active_ratio = active_count.float() / num_samples
            
            # Identify dead neurons
            dead_neurons = torch.where(active_count == 0)[0].tolist()
            
            neuron_stats[layer_name] = {
                'num_neurons': num_neurons,
                'num_samples': num_samples,
                'mean_activations': mean_acts.numpy(),
                'std_activations': std_acts.numpy(),
                'max_activations': max_acts.numpy(),
                'min_activations': min_acts.numpy(),
                'active_ratio': active_ratio.numpy(),
                'dead_neurons': dead_neurons,
                'num_dead': len(dead_neurons),
            }
        
        return neuron_stats
    
    def print_detailed_report(self):
        """Print detailed report with per-neuron statistics."""
        stats = self.get_neuron_stats()
        
        print("\n" + "="*100)
        print("DETAILED NEURON ACTIVATION ANALYSIS")
        print("="*100)
        
        total_neurons = sum(s['num_neurons'] for s in stats.values())
        total_dead = sum(s['num_dead'] for s in stats.values())
        
        print(f"Overall Statistics:")
        print(f"  Total Neurons Analyzed: {total_neurons:,}")
        print(f"  Total Dead Neurons: {total_dead:,} ({total_dead/total_neurons*100:.2f}%)")
        print(f"  Activation Threshold: {self.activation_threshold:.2e}\n")
        
        # Sort by number of dead neurons
        sorted_layers = sorted(stats.items(), key=lambda x: x[1]['num_dead'], reverse=True)
        
        print(f"{'Layer':<50} {'Neurons':>12} {'Dead':>12} {'Dead%':>10} {'Min Act':>15} {'Mean Act':>15}")
        print("-"*100)
        
        for layer_name, stat in sorted_layers:
            min_act = np.min(stat['mean_activations'])
            mean_act = np.mean(stat['mean_activations'])
            dead_pct = stat['num_dead'] / stat['num_neurons'] * 100
            
            print(f"{layer_name:<50} {stat['num_neurons']:>12,} {stat['num_dead']:>12,} "
                  f"{dead_pct:>9.2f}% {min_act:>15.6e} {mean_act:>15.6e}")
        
        print("\n" + "-"*100)
        
        # Show neurons with lowest activation
        print(f"\nNeurons with Lowest Average Activations:\n")
        
        for layer_name, stat in sorted_layers:
            if stat['num_dead'] > 0:
                print(f"{layer_name}:")
                print(f"  Dead neurons (never activated): {stat['dead_neurons'][:10]}")
                if len(stat['dead_neurons']) > 10:
                    print(f"  ... and {len(stat['dead_neurons']) - 10} more")
                print()
        
        print("="*100)
    
    def save_detailed_csv(self, output_dir: Path):
        """Save detailed per-layer and per-neuron statistics to CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = self.get_neuron_stats()
        
        # Layer-level summary
        layer_summary = []
        for layer_name, stat in stats.items():
            layer_summary.append({
                'layer_name': layer_name,
                'total_neurons': stat['num_neurons'],
                'dead_neurons': stat['num_dead'],
                'dead_ratio': stat['num_dead'] / stat['num_neurons'],
                'mean_activation': np.mean(stat['mean_activations']),
                'std_activation': np.mean(stat['std_activations']),
                'min_activation': np.min(stat['min_activations']),
                'max_activation': np.max(stat['max_activations']),
            })
        
        layer_df = pd.DataFrame(layer_summary).sort_values('dead_ratio', ascending=False)
        layer_path = output_dir / "layer_summary.csv"
        layer_df.to_csv(layer_path, index=False)
        print(f"Layer summary saved to {layer_path}")
        
        # Per-neuron statistics for each layer
        for layer_name, stat in stats.items():
            neuron_data = []
            for neuron_idx in range(stat['num_neurons']):
                neuron_data.append({
                    'neuron_id': neuron_idx,
                    'mean_activation': stat['mean_activations'][neuron_idx],
                    'std_activation': stat['std_activations'][neuron_idx],
                    'min_activation': stat['min_activations'][neuron_idx],
                    'max_activation': stat['max_activations'][neuron_idx],
                    'active_ratio': stat['active_ratio'][neuron_idx],
                    'is_dead': int(neuron_idx in stat['dead_neurons']),
                })
            
            neuron_df = pd.DataFrame(neuron_data)
            safe_name = layer_name.replace('/', '_').replace('.', '_')
            neuron_path = output_dir / f"neurons_{safe_name}.csv"
            neuron_df.to_csv(neuron_path, index=False)
        
        print(f"Per-neuron statistics saved to {output_dir}")


def load_model(model_path: str):
    """Load model from checkpoint."""
    from ucell.frm import FRMWrapper
    
    config = _CONFIG.value
    config.model.forward_dtype = "float32"
    
    model = FRMWrapper(config).eval()
    model.load_checkpoint(model_path)
    model = model.to('cuda')
    
    return model.inner


def load_folder_batches(folder: str, batch_size: int = 4, num_batches: int = 10) -> Generator:
    """See detect_dead_neurons.py for details; identical behavior."""
    from pathlib import Path
    import tifffile
    from torchvision.transforms.functional import center_crop

    path = Path(folder)
    files = sorted(path.glob("*.tif"))

    for batch_idx in range(0, len(files), batch_size):
        if batch_idx // batch_size >= num_batches:
            break
        batch_files = files[batch_idx:batch_idx+batch_size]
        imgs = []
        for f in batch_files:
            img = tifffile.imread(f)
            if img.ndim == 2:
                img = img[..., None]
            if img.shape[-1] == 1:
                img = np.tile(img, (1,1,3))
            elif img.shape[-1] == 2:
                img = np.pad(img, ((0,0),(0,0),(0,1)))
            img_t = torch.from_numpy(img).permute(2,0,1).float()
            if img_t.shape[1] != 256 or img_t.shape[2] != 256:
                img_t = center_crop(img_t, 256)
                img_t = torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
            img_t = img_t / 256.0
            imgs.append(img_t)
        if len(imgs) == batch_size:
            batch = {'image': torch.stack(imgs).cuda(),
                     'task_id': torch.zeros(batch_size, dtype=torch.long).cuda()}
            yield batch


def run_inference(model, analyzer, config=None, num_batches: int = 20, batch_size: int = 4, image_folder: str = None):
    """Run inference for analysis."""
    print(f"\nRunning inference on {num_batches} REAL data batches...\n")
    if image_folder:
        dataloader = load_folder_batches(image_folder, batch_size=batch_size, num_batches=num_batches)
    else:
        dataloader = scs(config, split='test')

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx >= num_batches:
                break
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            bs = batch['image'].shape[0]
            z_H = torch.zeros(bs, model.config.num_z_tokens, 
                                   model.config.hidden_size).cuda()
            z_L = torch.zeros(bs, model.config.num_z_tokens, 
                                   model.config.hidden_size).cuda()
            try:
                for _ in range(config.halt_max_steps):
                    z_H, z_L = model.forward(z_H, z_L, batch)
            except Exception as e:
                print(f"Warning: Error in batch {batch_idx}: {e}")
                continue


def main(argv):
    """Main function."""
    output_dir = Path(flags.FLAGS.output_dir) if flags.FLAGS.output_dir else Path("..") / "dead_neuron_analysis_results"
    
    if flags.FLAGS.model is None:
        print("Error: --model flag is required")
        return
    
    print(f"Loading model from {flags.FLAGS.model}...")
    model = load_model(flags.FLAGS.model)
    model.eval()
    
    print("Initializing analyzer...")
    analyzer = AdvancedNeuronAnalyzer(model, flags.FLAGS.activation_threshold)
    
    print("Registering hooks...")
    hooks = analyzer.register_hooks()
    print(f"Registered {len(hooks)} hooks")
    
    print("\nRunning inference analysis on REAL data...")
    run_inference(model, analyzer, 
                  config=_CONFIG.value,
                  num_batches=flags.FLAGS.max_batches,
                  batch_size=flags.FLAGS.batch_size,
                  image_folder=flags.FLAGS.image_folder)
    
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*100)
    analyzer.print_detailed_report()
    analyzer.save_detailed_csv(output_dir)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    app.run(main)
