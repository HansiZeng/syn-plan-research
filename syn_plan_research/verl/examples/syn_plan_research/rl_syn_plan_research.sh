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
# Robust Greenland training with S3 resume + periodic sync
set -euo pipefail
set -x

ulimit -n 65535

NGPU=4
export VLLM_USE_V1=1

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
SEARCH_API_KEY="${SERPER_API_KEY:?Error: SERPER_API_KEY is not set in .env}"
export HF_HOME=${DATA_ROOT}/.cache/huggingface

# -------- User configs (same as your original) --------
PROJECT_DIR="$(pwd)"

train_files="['${DATA_ROOT}/data/rl/train.parquet']"
model_path="hzeng/syn-plan-research-4B-sft"  # Can also use local path like "/path/to/model"
val_files="['${DATA_ROOT}/data/eval/validation.parquet']"
train_batch_size=256
ppo_mini_batch_size=$((train_batch_size / 8))

project_name="syn_plan_research"
# hyperparamters:
max_assistant_turns=16
format_score=0.2
total_epochs=3
actor_lr=1e-6
filter_unfinished=True
kl_loss_coef=0.000
rollout_temp=1.0
rollout_top_p=1.0
rollout_top_k=-1

echo "Model path: $model_path"

# -------- Fixed timestamp for deterministic experiment_name --------
timestamp=

# -------- Paths --------
checkpoint_base="${DATA_ROOT}/checkpoints/rl"
CACHE_FILE="${DATA_ROOT}/cache/serper_search_cache.jsonl"


experiment_name="rl_4B_sft_init_${model_path##*/}_bsz${train_batch_size}_ppo_mbsz${ppo_mini_batch_size}_maxturns${max_assistant_turns}_fs${format_score}_klcoef${kl_loss_coef}_actorlr${actor_lr}_epochs${total_epochs}_${timestamp}"
CHECKPOINT_DIR="${checkpoint_base}/${project_name}/${experiment_name}"


echo "📦 Start Training"
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_batch_size=$train_batch_size \
  data.val_batch_size=4096 \
  data.max_prompt_length=16384 \
  data.max_response_length=4096 \
  data.truncation='error' \
  data.custom_cls.path=pkg://verl/utils/dataset/web_research_dataset \
  data.custom_cls.name=WebResearchRLDataset \
  actor_rollout_ref.model.path=$model_path \
  actor_rollout_ref.actor.optim.lr=$actor_lr \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.max_model_len=15000 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.n=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
  actor_rollout_ref.rollout.prompt_length=32768 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.multi_turn.completion_callback=tests.workers.rollout.my_test_web_research_completion_callback.Qwen3CustomToolCompletionCallback \
  actor_rollout_ref.rollout.filter_unfinished=$filter_unfinished \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$project_name" \
  trainer.experiment_name="$experiment_name" \
  trainer.n_gpus_per_node=$NGPU \
  trainer.save_web_search_cache_freq=10 \
  trainer.save_web_search_cache_on_validate=False \
  trainer.nnodes=1 \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  trainer.save_freq=10 \
  trainer.test_freq=100000 \
  tool_server.web_search_server.api_key=$SEARCH_API_KEY \
  data.train_files="$train_files" \
  data.val_files="$val_files"  \
  trainer.total_epochs=$total_epochs \
  reward_model.reward_manager=format_naive \
  +reward_model.reward_kwargs='{format_score: '"$format_score"'}' \
  trainer.debug_rollout_data_dir="$CHECKPOINT_DIR/debug_rollout" \
  trainer.debug_rollout_freq=1 \
  actor_rollout_ref.rollout.temperature=$rollout_temp \
  actor_rollout_ref.rollout.top_p=$rollout_top_p \
  actor_rollout_ref.rollout.top_k=$rollout_top_k 