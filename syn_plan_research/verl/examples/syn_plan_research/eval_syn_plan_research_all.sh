#!/bin/bash
#SBATCH --job-name=syn-plan-research-eval
#SBATCH --output=sbatch_jobs/sbatch_out/syn-plan-research-eval_%j.out
#SBATCH --partition=superpod-a100
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00

source /work/hzeng_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate verl-vllm083

export CUDA_VISIBLE_DEVICES=0,1,2,3
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
export HF_HOME=${DATA_ROOT}/.cache/huggingface
SEARCH_API_KEY="${SERPER_API_KEY:?Error: SERPER_API_KEY is not set in .env}"
CACHE_FILE="${DATA_ROOT}/cache/serper_search_cache.jsonl"

# ============================================================================
# Configuration Options
# ============================================================================

# Option 1: Use HuggingFace dataset (recommended for reproducibility)
val_files_or_dsname="hzeng/syn-plan-research-data-eval"

# Option 2: Use local parquet file(s) - uncomment to use
# val_files_or_dsname="['/path/to/your/eval.parquet']"
# val_files_or_dsname="['${DATA_ROOT}/data/eval/validation.parquet']"

# Model configuration
max_assistant_turns=16
CHECKPOINT_DIR="hzeng/syn-plan-research-4B"  # Can also use local path like "/path/to/model"
BASE_OUTPUT_DIR="${DATA_ROOT}/eval_outputs"

# ============================================================================
# Helper function to run evaluation with different configs
# ============================================================================
run_eval() {
    local pass_at_k=$1
    local data_source=$2
    local eval_output_path=$3
    
    echo ""
    echo "🚀 Starting evaluation..."
    echo "📊 Data source filter: ${data_source:-ALL}"
    echo "📈 Pass@k: $pass_at_k"
    echo "🤖 Model: $CHECKPOINT_DIR"
    echo "📁 Output: $eval_output_path"
    echo ""
    
    python3 -m verl.trainer.async_main_generation_and_eval \
        data.max_prompt_length=16384 \
        data.max_response_length=4096 \
        data.truncation='error' \
        data.custom_cls.path=pkg://verl/utils/dataset/web_research_dataset \
        data.custom_cls.name=WebResearchRLDataset \
        actor_rollout_ref.model.path=$CHECKPOINT_DIR \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.max_model_len=20000 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.eval_output_path=$eval_output_path \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.max_assistant_turns=$max_assistant_turns \
        actor_rollout_ref.rollout.prompt_length=32768 \
        actor_rollout_ref.rollout.response_length=8192 \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.completion_callback=tests.workers.rollout.my_test_web_research_completion_callback.Qwen3CustomToolCompletionCallback \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        trainer.n_gpus_per_node=$NGPU \
        trainer.nnodes=1 \
        tool_server.web_search_server.api_key=$SEARCH_API_KEY \
        tool_server.web_search_server.cache_file=$CACHE_FILE \
        data.val_files="$val_files_or_dsname" \
        validation.pass_at_k="$pass_at_k" \
        ${data_source:+validation.data_source=$data_source} \
        trainer.total_epochs=1
    
    # Clean up Ray processes after each experiment to free GPU memory
    echo "🧹 Cleaning up Ray processes..."
    ray stop --force 2>/dev/null
    sleep 5
    echo "✅ Ray cleanup done."
}

# validation.pass_at_k="$pass_at_k" \
# ${data_source:+validation.data_source=$data_source} \

# ============================================================================
# Run Evaluations
# ============================================================================

# Evaluation 1: Pass@1 on all data sources
run_eval 1 "" "$BASE_OUTPUT_DIR/syn_plan_research_4b_pass_at_1_all/eval.parquet"

# Evaluation 2: Pass@4 on GAIA only
run_eval 4 "web_research_GAIA" "$BASE_OUTPUT_DIR/syn_plan_research_4b_pass_at_4_gaia/eval.parquet"

echo ""
echo "✅ All evaluations completed!"

