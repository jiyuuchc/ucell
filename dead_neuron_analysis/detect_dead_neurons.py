"""
Script to detect and analyze dead neurons in the FRM model during inference.

Dead neurons are those that have zero or near-zero activations across inference batches.
This can happen due to poor initialization, aggressive regularization, or architectural issues.
"""

from absl import app, flags
from pathlib import Path
from typing import Dict, List, Tuple, Generator
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from ml_collections import config_flags

from ucell.frm import FRMWrapper, FRM
from ucell.utils import patcherize

# ensure project root is on Python path so `data.py` can be imported
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parent.parent))
from data import scs

flags.DEFINE_string("model", None, "Path to model checkpoint")
flags.DEFINE_string("layer_types", "SwiGLU", "Comma-separated class names from ucell.layers to monitor (e.g. 'SwiGLU,Attention')")
flags.DEFINE_float("activation_threshold", 1e-5, "Threshold for considering a neuron active")
flags.DEFINE_integer("max_batches", None, "Maximum number of batches to process (None for all)")
flags.DEFINE_string("output_dir", None, "Output directory for results (default: ../dead_neuron_analysis_results)")
flags.DEFINE_integer("task_id", 0, "Task ID for inference")
flags.DEFINE_string("image_folder", None, "Path to a directory of .tif images to use instead of the SCS dataloader")

_CONFIG = config_flags.DEFINE_config_file("config", "config.py")


class DeadNeuronDetector:
    """Detects and analyzes dead neurons in a neural network."""
    
    def __init__(self, model: nn.Module, activation_threshold: float = 1e-5):
        self.model = model
        self.activation_threshold = activation_threshold
        self.hooks = []
        self.activation_stats = {}
        self.layer_names = []
        # resolve layer classes from flag
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
        """Register forward hooks on layers matching configured classes."""
        hook_handles = []
        
        for name, module in self.model.named_modules():
            if self._is_target_layer(module):
                handle = module.register_forward_hook(
                    self._make_hook(name)
                )
                hook_handles.append(handle)
                self.layer_names.append(name)
        
        return hook_handles
    
    def _is_target_layer(self, module: nn.Module) -> bool:
        """Return True if module is instance of any configured class."""
        return isinstance(module, self.layer_classes)
    
    def _make_hook(self, layer_name: str):
        """Create a forward hook function for capturing activations."""
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Detach and move to CPU to save memory
                act = output.detach().cpu()
                
                # Compute statistics
                act_flat = act.view(-1)
                
                if layer_name not in self.activation_stats:
                    self.activation_stats[layer_name] = {
                        'mean': [],
                        'std': [],
                        'min': [],
                        'max': [],
                        'zero_ratio': [],
                        'total_samples': 0,
                        'total_neurons': act.shape[-1] if act.dim() > 0 else 1,
                    }
                
                self.activation_stats[layer_name]['mean'].append(act.mean().item())
                self.activation_stats[layer_name]['std'].append(act.std().item())
                self.activation_stats[layer_name]['min'].append(act.min().item())
                self.activation_stats[layer_name]['max'].append(act.max().item())
                
                # Count zero activations per neuron (last dimension)
                if act.dim() >= 2:
                    # Reshape to [batch*sequence, neurons]
                    act_2d = act.view(-1, act.shape[-1])
                    near_zero = (act_2d.abs() < self.activation_threshold).float().mean(dim=0)
                    zero_ratio = near_zero.mean().item()
                else:
                    zero_ratio = (act.abs() < self.activation_threshold).float().mean().item()
                
                self.activation_stats[layer_name]['zero_ratio'].append(zero_ratio)
                self.activation_stats[layer_name]['total_samples'] += act.numel()
        
        return hook
    
    def get_dead_neuron_stats(self) -> Dict:
        """Compute dead neuron statistics from collected activations."""
        stats = {}
        
        for layer_name, act_stats in self.activation_stats.items():
            # Average stats across batches
            mean_activation = np.mean(act_stats['mean'])
            std_activation = np.mean(act_stats['std'])
            min_activation = np.min(act_stats['min'])
            max_activation = np.max(act_stats['max'])
            avg_zero_ratio = np.mean(act_stats['zero_ratio'])
            
            num_neurons = act_stats['total_neurons']
            estimated_dead = int(num_neurons * avg_zero_ratio)
            
            stats[layer_name] = {
                'num_neurons': num_neurons,
                'dead_neurons': estimated_dead,
                'dead_ratio': avg_zero_ratio,
                'mean_activation': mean_activation,
                'std_activation': std_activation,
                'min_activation': min_activation,
                'max_activation': max_activation,
                'total_samples': act_stats['total_samples'],
            }
        
        return stats
    
    def print_report(self, stats: Dict = None):
        """Print a formatted report of dead neurons."""
        if stats is None:
            stats = self.get_dead_neuron_stats()
        
        print("\n" + "="*90)
        print("DEAD NEURON ANALYSIS REPORT")
        print("="*90)
        print(f"Activation Threshold: {self.activation_threshold}")
        print(f"Total Layers Analyzed: {len(stats)}\n")
        
        # Summary statistics
        total_neurons = sum(s['num_neurons'] for s in stats.values())
        total_dead = sum(s['dead_neurons'] for s in stats.values())
        avg_dead_ratio = total_dead / total_neurons if total_neurons > 0 else 0
        
        print(f"SUMMARY:")
        print(f"  Total Neurons: {total_neurons:,}")
        print(f"  Total Dead Neurons: {total_dead:,}")
        print(f"  Overall Dead Ratio: {avg_dead_ratio:.4f} ({avg_dead_ratio*100:.2f}%)")
        print("\n" + "-"*90)
        print(f"{'Layer Name':<50} {'Neurons':>12} {'Dead':>12} {'Dead %':>10}")
        print("-"*90)
        
        # Sort by dead ratio
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['dead_ratio'], reverse=True)
        
        for layer_name, stat in sorted_stats:
            dead_pct = stat['dead_ratio'] * 100
            print(f"{layer_name:<50} {stat['num_neurons']:>12,} {stat['dead_neurons']:>12,} {dead_pct:>9.2f}%")
        
        print("-"*90)
        print(f"\nDETAILED STATISTICS:\n")
        
        for layer_name, stat in sorted_stats:
            print(f"\n{layer_name}:")
            print(f"  Neurons: {stat['num_neurons']}")
            print(f"  Dead Neurons: {stat['dead_neurons']} ({stat['dead_ratio']*100:.2f}%)")
            print(f"  Mean Activation: {stat['mean_activation']:.6e}")
            print(f"  Std Activation: {stat['std_activation']:.6e}")
            print(f"  Min Activation: {stat['min_activation']:.6e}")
            print(f"  Max Activation: {stat['max_activation']:.6e}")
            print(f"  Samples Processed: {stat['total_samples']:,}")
        
        print("\n" + "="*90)
    
    def save_report(self, output_path: Path):
        """Save detailed report to CSV."""
        stats = self.get_dead_neuron_stats()
        
        # Create DataFrame
        data = []
        for layer_name, stat in stats.items():
            data.append({
                'layer_name': layer_name,
                'total_neurons': stat['num_neurons'],
                'dead_neurons': stat['dead_neurons'],
                'dead_ratio': stat['dead_ratio'],
                'mean_activation': stat['mean_activation'],
                'std_activation': stat['std_activation'],
                'min_activation': stat['min_activation'],
                'max_activation': stat['max_activation'],
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('dead_ratio', ascending=False)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")
        
        return df


def load_model(model_path: str):
    """Load the FRM model from checkpoint."""
    config = _CONFIG.value
    config.model.forward_dtype = "float32"
    
    model = FRMWrapper(config).eval()
    model.load_checkpoint(model_path)
    model = model.to('cuda')
    
    # Return the inner FRM model (not the wrapper)
    return model.inner




def load_scs_val_batches(config, split: str = 'test', num_batches: int = 10) -> Generator:
    """
    Load real data batches from the SCS dataset.

    Uses the existing data pipeline from data.py to load real cell images
    with proper preprocessing and augmentation (disabled for inference).
    """
    # Create dataloader using the existing data pipeline
    dataloader = scs(config, split=split)

    # Yield batches up to num_batches limit
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        yield batch


def load_folder_batches(folder: str, batch_size: int = 4, num_batches: int = 10) -> Generator:
    """Load TIFF images from a folder as inference batches.

    Images are center‑cropped/resized to 256×256, padded to 3 channels, and
    normalized by 256 to match the format used by the SCS loader.

    Args:
        folder: path containing only .tif files
        batch_size: how many images per batch
        num_batches: maximum number of batches to yield
    """
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
            # convert to tensor C,H,W
            img_t = torch.from_numpy(img).permute(2,0,1).float()
            # center crop then resize if necessary
            if img_t.shape[1] != 256 or img_t.shape[2] != 256:
                img_t = center_crop(img_t, 256)
                img_t = torch.nn.functional.interpolate(img_t.unsqueeze(0), size=(256,256), mode='bilinear', align_corners=False).squeeze(0)
            img_t = img_t / 256.0
            imgs.append(img_t)
        if len(imgs) == batch_size:
            batch = {'image': torch.stack(imgs).cuda(),
                     'task_id': torch.zeros(batch_size, dtype=torch.long).cuda()}
            yield batch




def analyze_with_real_data(model, config, num_batches: int = 10, image_folder: str = None):
    """
    Run analysis with REAL data for accurate dead neuron detection.

    By default this sources batches from the SCS dataset via data.py.
    If `image_folder` is provided it will load TIFFs from that directory
    instead (cropping/normalizing to 256x256). This bypasses the standard
    dataloader.
    """
    source = "folder" if image_folder else "SCS dataset"
    print(f"\nRunning inference on {num_batches} REAL data batches from {source}...\n")

    try:
        if image_folder:
            batch_generator = load_folder_batches(image_folder, batch_size=config.batch_size, num_batches=num_batches)
        else:
            batch_generator = load_scs_val_batches(config, split='test', num_batches=num_batches)
    except Exception as e:
        print(f"Error loading real data: {e}")
        import traceback
        traceback.print_exc()
        return False

    with torch.no_grad():
        for i, batch in enumerate(tqdm(batch_generator, total=num_batches)):
            # Move batch to GPU if not already
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Initialize hidden states for the FRM model
            bs = batch['image'].shape[0]
            z_H = torch.zeros(bs, model.config.num_z_tokens, model.config.hidden_size).cuda()
            z_L = torch.zeros(bs, model.config.num_z_tokens, model.config.hidden_size).cuda()
            try:
                for _ in range(config.halt_max_steps):
                    z_H, z_L = model.forward(z_H, z_L, batch)
            except Exception as e:
                print(f"Warning: Error during inference in batch {i}: {e}")
                continue
    
    return True


def main(argv):
    """Main function."""
    output_dir = Path(flags.FLAGS.output_dir) if flags.FLAGS.output_dir else Path("..") / "dead_neuron_analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    if flags.FLAGS.model is None:
        print("Error: --model flag is required")
        return
    
    print(f"Loading model from {flags.FLAGS.model}...")
    model = load_model(flags.FLAGS.model)
    model.eval()
    
    # Create detector
    detector = DeadNeuronDetector(
        model,
        activation_threshold=flags.FLAGS.activation_threshold
    )
    
    # Register hooks
    print("Registering hooks on MLP layers...")
    hook_handles = detector.register_hooks()
    print(f"Registered {len(hook_handles)} hooks")
    
    # Run analysis
    print("\nStarting inference analysis on REAL data...")
    
    # Always use real data
    success = analyze_with_real_data(
        model,
        config=_CONFIG.value,
        num_batches=flags.FLAGS.max_batches or 20,
        image_folder=flags.FLAGS.image_folder
    )
    if not success:
        print("Real data loading failed, aborting analysis.")
        return
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    # Print report
    detector.print_report()
    
    # Save report
    report_path = output_dir / "dead_neurons_report.csv"
    detector.save_report(report_path)
    
    # Also save a summary JSON
    stats = detector.get_dead_neuron_stats()
    summary = {
        'total_neurons': sum(s['num_neurons'] for s in stats.values()),
        'total_dead_neurons': sum(s['dead_neurons'] for s in stats.values()),
        'activation_threshold': flags.FLAGS.activation_threshold,
        'num_layers_analyzed': len(stats),
        'data_type': 'real',
    }
    
    import json
    summary_path = output_dir / "dead_neurons_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    app.run(main)
