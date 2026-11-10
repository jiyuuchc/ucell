#!/bin/bash
#SBATCH --qos=general
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=48g
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 4:00:00
#SBATCH -J judge_ft
#
# One arm of the judge-signal experiment: fine-tune the 375-pious-sound
# checkpoint on ~/datasets/judge_ft/<arm>/train (60 musc images whose labels
# come from one of: human, cpsam = Cellpose-SAM masks as-is, judge = those
# masks edited by Opus verdicts) and score on the shared 40-image validation
# split with human labels.
#
# Same recipe for every arm, and the same budget as the few-shot runs in
# analyses/run_few_shot_adapt.sh: LoRA rank 16, ema 0.95, one iteration.
# epochs_per_iter is the per-image repeat count, so 850 x 60 images is the
# ~51k samples that 10240 x 5 shots was.
#
# Usage: sbatch scripts/judge_ft.sbatch <arm>
set -euo pipefail
cd "$HOME/work/ucell"
source .venv/bin/activate

ARM=${1:?usage: judge_ft.sbatch <human|cpsam|judge|judge_hint|judge_inst>}
DATA=$HOME/datasets/judge_ft/$ARM
test -d "$DATA/train" || { echo "missing $DATA/train"; exit 2; }
echo "arm: $ARM  ($(ls "$DATA/train" | grep -c _label) labels)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u few_shot_adapt.py \
    --data_dir "$DATA" \
    --skip_rebuild \
    --sz "${SZ:-106}" \
    --results_dir analyses/judge_ft \
    --predict_outputdir "predictions/judge_ft_${ARM}" \
    --checkpoints_dir "checkpoints/judge_ft_${ARM}" \
    --nocleanup \
    -- \
    --config config.py:train \
    --config.model.hidden_size=1024 \
    --config.model.norm_task_emb=true \
    --init checkpoints/375-pious-sound/final_ema.pt \
    --config.epochs_per_iter="${EPI:-850}" \
    --config.n_iters=1 \
    --config.task_id=0 \
    --config.lora.rank=16 \
    --config.dataloader_workers=2 \
    --config.ema_decay=0.95 \
    ${EXTRA_CFG:-}

echo "[done] arm $ARM"
