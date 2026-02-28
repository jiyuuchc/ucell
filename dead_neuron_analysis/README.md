# Dead Neuron Analysis Toolkit

Complete suite for detecting and analyzing dead neurons in your FRM model during inference.

## Quick Start

### Basic Analysis
```bash
cd dead_neuron_analysis
python detect_dead_neurons.py --model=../checkpoints/your-run/cp_00000357.pt --max_batches=20
```

### Detailed Per-Neuron Analysis
```bash
cd dead_neuron_analysis
python analyze_neurons_advanced.py \
  --model=../checkpoints/your-run/cp_00000357.pt \
  --max_batches=50
```


## Scripts

### 1. `detect_dead_neurons.py`
Fast layer-level analysis of dead neurons.

**Outputs:**
- `dead_neurons_report.csv` - Summary by layer
- `dead_neurons_summary.json` - Quick statistics

**Key Flags:**
- `--model`: Checkpoint path (required unless `--image_folder` is used)
- `--layer_types`: Comma-separated list of layer class names from `ucell.layers` to monitor (default: `SwiGLU`)
- `--image_folder`: Directory containing `.tif` images for analysis (bypasses SCS)
- `--max_batches`: Number of batches (default: 20)
- `--activation_threshold`: Dead neuron threshold (default: 1e-5)
- `--output_dir`: Results directory (default: ../dead_neuron_analysis_results)

### 2. `analyze_neurons_advanced.py`
Detailed per-neuron activation statistics.

**Outputs:**
- `layer_summary.csv` - Layer-level statistics
- `neurons_<layer_name>.csv` - Per-neuron details for each layer

**Key Flags:**
- `--model`: Checkpoint path (required)
- `--layer_types`: Comma-separated list of layer class names from `ucell.layers` to monitor (default: `SwiGLU`)
- `--max_batches`: Number of batches (default: 20)
- `--activation_threshold`: Dead neuron threshold (default: 1e-5)


### 4. `examples.sh`
Copy-paste ready command examples for all analysis types.

## Documentation

- **ANALYSIS_GUIDE.md** - Detailed explanation of metrics and interpretation
- **DATA_INVARIANCE.md** - Why input data matters for dead neuron detection
- **REAL_DATA_INTEGRATION.md** - How to integrate with your data.py
- **real_data_loader_template.py** - Templates for custom data loading

## Key Concepts

### Dead Neurons
Neurons with zero or near-zero activations across inference batches, indicating:
- Poor initialization
- Gradient flow issues
- Architectural mismatches
- Excessive regularization

### Real vs. Random Data
**Important**: Dead neuron detection is data-dependent!

All scripts use real data from your SCS dataset for accurate production analysis. This ensures results reflect actual dead neurons in your model.

### Metrics

- **Dead Ratio**: Percentage of neurons with near-zero activations
  - Healthy: < 1%
  - Warning: 1-5%
  - Critical: > 5%

- **Mean Activation**: Average activation magnitude (should be non-zero)
- **Active Ratio**: Fraction of samples where a neuron activates
- **Dead Neurons**: Count of neurons below activation threshold

## Workflow

### 1. Quick Assessment (5 minutes)
```bash
python detect_dead_neurons.py --model=../checkpoints/your-run/cp_*.pt --max_batches=20
```

### 2. Comprehensive Analysis (10-30 minutes)
```bash
# using checkpoint(s)
python detect_dead_neurons.py \
  --model=../checkpoints/your-run/cp_*.pt \
  --max_batches=100

python analyze_neurons_advanced.py \
  --model=../checkpoints/your-run/cp_*.pt \
  --max_batches=100

# or using a folder of TIFF images instead of a checkpoint
python detect_dead_neurons.py \
  --image_folder=/path/to/images \
  --max_batches=100
```

## Integration with data.py

Scripts automatically use your `data.py` functions and real data from SCS dataset:
- Uses your preprocessing and task mapping
- Respects configuration from `config.py`
- No additional setup required

## Performance

- **Time**: ~0.5-2 seconds per batch
- **Memory**: ~2-3 GB for batch_size=4
- **Typical runs**:
  - 20 batches: 2-5 minutes
  - 50 batches: 5-10 minutes
  - 100 batches: 10-20 minutes

## Troubleshooting

### Out of Memory
- Reduce `--max_batches`
- Check available GPU memory

### No Hooks Registered
- Verify model loads correctly
- Check that model has SwiGLU layers

### Unexpected Results
- Use `--max_batches=100+` for stable statistics
- Check `activation_threshold` appropriateness
- Verify results across multiple runs for consistency

See documentation files for more detailed guidance.
