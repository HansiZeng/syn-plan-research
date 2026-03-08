#!/bin/bash
# ============================================================================
# Download SynPlanResearch datasets from HuggingFace to local parquet files.
#
# Datasets:
#   - hzeng/syn-plan-research-data-sft   -> data/sft/
#   - hzeng/syn-plan-research-data-eval  -> data/eval/
#   - hzeng/syn-plan-research-data-rl    -> data/rl/
#
# Usage:
#   bash download_parquets_to_local.sh
# ============================================================================

# Activate conda environment
source /work/hzeng_umass_edu/miniconda3/etc/profile.d/conda.sh
conda activate verl-vllm083

# -------- Load config from .env --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
else
    echo "ERROR: $REPO_ROOT/.env not found. Run: cp .env.example .env and fill in your paths."
    exit 1
fi
DATA_ROOT="${DATA_ROOT:?Error: DATA_ROOT is not set in .env}"

# Output directory
OUTPUT_DIR="${DATA_ROOT}/data"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Downloading SynPlanResearch datasets from HuggingFace..."
echo "📁 Output directory: $OUTPUT_DIR"
echo ""

python3 "$SCRIPT_DIR/downoad_parquets_to_local.py" --output_dir "$OUTPUT_DIR" --datasets all

echo ""
echo "✅ Download complete!"
echo ""
echo "You can now use local parquet files in eval_syn_plan_research_all.sh:"
echo "  val_files_or_dsname=\"['$OUTPUT_DIR/eval/validation.parquet']\""
