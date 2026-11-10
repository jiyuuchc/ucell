#!/bin/bash
#SBATCH --qos=general
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=60g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH -t 8:00:00

# Whole-image context experiment: freeze a pretrained backbone and train only
# the task embedding (collapsed to one shared row) plus the context encoder.
# Compare against scripts/train_emb.sbatch on the same checkpoint and data;
# any difference is down to the whole-image context.
#
# $1 = checkpoint, $2 = data dir (default ~/datasets/cellseg)
set -euo pipefail
CP="$1"
DATADIR="${2:-$HOME/datasets/cellseg}"
cd "$HOME/work/ucell"
source .venv/bin/activate

srun python -u train.py \
    --config config.py:train_ctx \
    --config.model.hidden_size=1024 \
    --config.data_dir="$DATADIR" \
    --config.collapse_task_id=0 \
    --config.sampling_alpha=1.0 \
    --config.epochs_per_iter=10 \
    --config.n_iters=3 \
    --config.model.norm_task_emb=true \
    --init "$CP"
