#!/bin/bash
#SBATCH --qos=general
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -c 24
#SBATCH --mem=0
#SBATCH --gres=gpu:L40S:3
#SBATCH -t 1-00:00:00
#SBATCH -J scratch-ctx
#
# From-scratch training with the whole-image context encoder, otherwise an
# exact copy of the scripts/scratch_l40s.sbatch recipe so it pairs with that
# baseline (383-rose-pine: `sbatch scripts/scratch_l40s.sbatch 768`, seed
# 20260823):
#
#   config.py:train, ~/datasets/cellseg, norm_task_emb, collapse_task_id=0,
#   batch 64/gpu on 3 L40S, 16 iters @ lr 1e-4 then 5 iters @ lr 1e-5
#
# plus --config.model.ctx_size=$CTX (default 256).  Because the task id is
# collapsed, the table row is a shared bias and the context encoder is the
# only per-image conditioning.
#
# Runs the code in the submit directory (submit from the image-context
# worktree), with the main checkout's .venv, and writes checkpoints and the
# wandb dir to the main checkout's checkpoints/ alongside every other run.
#
# Usage (from the worktree):
#   sbatch scripts/scratch_ctx_l40s.sbatch 768
#   CTX=512 SEED=1 sbatch scripts/scratch_ctx_l40s.sbatch 768
set -euo pipefail
HIDDEN="${1:-768}"
SEED="${SEED:-20260823}"
CTX="${CTX:-256}"
SRC="${SLURM_SUBMIT_DIR:-$PWD}"
UCELL="$HOME/work/ucell"
cd "$SRC"
test -f config.py && grep -q ctx_size config.py || {
    echo "$SRC has no context-encoder code; submit from the worktree" >&2
    exit 2
}
source "$UCELL/.venv/bin/activate"
echo "code: $SRC @ $(git rev-parse --short HEAD) ($(git branch --show-current))"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}

PHASE1_LOG=$(mktemp)
trap 'rm -f "$PHASE1_LOG"' EXIT

common_args=(
    --config config.py:train
    --config.model.hidden_size="$HIDDEN"
    --config.batch_size="${BATCH:-64}"
    --config.data_dir="$HOME/datasets/cellseg"
    --config.model.norm_task_emb=true
    --config.collapse_task_id=0
    --config.seed="$SEED"
    --config.model.ctx_size="$CTX"
    --dir "$UCELL/checkpoints"
)

echo "=============== phase 1: 16 iters @ lr 1e-4 (seed $SEED, ctx $CTX) ==============="
srun python -u train.py \
    "${common_args[@]}" \
    --config.opt.lr=1e-4 \
    --config.n_iters=16 \
    ${EXTRA_CFG:-} 2>&1 | tee "$PHASE1_LOG"

# "Checkpoints saved in <dir>/<n>-<name>" is printed by setup_logging.  No
# fallback to the shared checkpoints/wandb/latest-run symlink: with other
# jobs running it can point at their run (see scripts/scs_l40s.sbatch).
CP_DIR=$(grep -m1 -oE "Checkpoints saved in \S+" "$PHASE1_LOG" | awk '{print $NF}')
RUN_ID=$(grep -oE "/runs/[A-Za-z0-9]+" "$PHASE1_LOG" | tail -1 | cut -d/ -f3)
if [ -z "${RUN_ID:-}" ]; then
    RUN_ID=$(grep -oE "setting up run [A-Za-z0-9]+" "$PHASE1_LOG" | tail -1 | awk '{print $NF}')
fi
echo "phase 1 done: cp_dir=$CP_DIR run_id=$RUN_ID"
test -n "$CP_DIR" && test -n "$RUN_ID" || { echo "cannot locate phase 1 run"; exit 3; }

cp "$CP_DIR/final_ema.pt" "$CP_DIR/final_ema_lr1e-4.pt"

echo "=============== phase 2: 5 iters @ lr 1e-5 (resume $RUN_ID) ==============="
srun python -u train.py \
    "${common_args[@]}" \
    --config.opt.lr=1e-5 \
    --config.n_iters=5 \
    ${EXTRA_CFG:-} \
    --resume "$RUN_ID"

echo "=============== done: $CP_DIR ==============="
ls -la "$CP_DIR"
