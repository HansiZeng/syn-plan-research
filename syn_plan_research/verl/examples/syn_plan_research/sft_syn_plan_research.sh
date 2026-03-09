#!/bin/bash
#SBATCH --job-name=syn-plan-research-eval
#SBATCH --output=sbatch_jobs/sbatch_out/syn-plan-research-eval_%j.out
#SBATCH --partition=superpod-a100
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00

conda activate verl-vllm083


nproc_per_node=4
set -e

# -------- Load config from .env --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
else
    echo "ERROR: $REPO_ROOT/.env not found. Run: cp .env.example .env and fill in your paths + API key."
    exit 1
fi
DATA_ROOT="${DATA_ROOT:?Error: DATA_ROOT is not set in .env}"
export HF_HOME=${DATA_ROOT}/.cache/huggingface


# 传递的参数
TRAIN_PATH=${DATA_ROOT}/data/sft/train.parquet
CKPT_DIR=Qwen/Qwen3-4B

echo "TRAIN_PATH: $TRAIN_PATH"
echo "CKPT_DIR: $CKPT_DIR"

# 写入超参数
ckpt_name="${CKPT_DIR##*/}"
LR=5e-6
train_batch_size=16
total_epochs=5
project_name=multiturn-sft
timestamp=$(date "+%Y%m%d_%H%M%S")
experiment_name="sft_${data_dir_name}_init_${ckpt_name}_lr_${LR}_ep_${total_epochs}_${timestamp}"
save_path=${DATA_ROOT}/checkpoints/sft/${experiment_name}


save_freq=625

echo ✅ save_path: $save_path
echo ✅ save_freq: $save_freq
echo ✅ experiment_name: $experiment_name
# ray stop --force

torchrun --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TRAIN_PATH \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.micro_batch_size=4 \
    data.train_batch_size=$train_batch_size \
    data.max_length=32768 \
    model.partial_pretrain=$CKPT_DIR \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger=['wandb'] \
    trainer.total_epochs=$total_epochs \
    use_remove_padding=true \
    ulysses_sequence_parallel_size=2 \
    optim.lr=$LR \
    trainer.save_freq=$save_freq   #130 steps per epoch, 1300 steps for 5 epochs
