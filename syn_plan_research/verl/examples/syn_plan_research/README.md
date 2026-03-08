# Syn-Plan-Research Evaluation Scripts

This directory contains evaluation scripts for the Syn-Plan-Research models.

## 📦 Available Resources

### Models on HuggingFace
- `hzeng/syn-plan-research-4B` - 4B model
- `hzeng/syn-plan-research-4B-sft` - 4B SFT model
- `hzeng/syn-plan-research-8B` - 8B model
- `hzeng/syn-plan-research-8B-sft` - 8B SFT model

### Datasets on HuggingFace
- `hzeng/syn-plan-research-data-eval` - Evaluation dataset
- `hzeng/syn-plan-research-data-sft` - SFT training dataset
- `hzeng/syn-plan-research-data-rl` - RL training dataset

## 🚀 Quick Start

### Method 1: Using HuggingFace Dataset (Recommended)

```bash
# Edit eval_syn_plan_research_all.sh
val_files_or_dsname="hzeng/syn-plan-research-data-eval"
CHECKPOINT_DIR="hzeng/syn-plan-research-4B"

# Run evaluation
bash eval_syn_plan_research_all.sh
```

### Method 2: Using Local Parquet File

```bash
# Edit eval_syn_plan_research_all.sh
val_files_or_dsname="['/path/to/your/eval.parquet']"
CHECKPOINT_DIR="/path/to/local/model"  # or HF model name

# Run evaluation
bash eval_syn_plan_research_all.sh
```

## 📝 Data Format

The evaluation dataset should contain the following fields:

```python
{
    'id': '0',
    'question': 'What was Iqbal F. Qadir on when he participated...',
    'data_source': 'web_research_hotpotqa',
    'split': 'test',
    'reward_model': {
        'ground_truth': {
            'target': ['flotilla']
        }
    }
}
```

## 🔧 Configuration Options

### Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data.val_files` | Data source (HF dataset or local path) | `"hzeng/syn-plan-research-data-eval"` or `"['/path/file.parquet']"` |
| `actor_rollout_ref.model.path` | Model path (HF or local) | `"hzeng/syn-plan-research-4B"` or `"/path/to/model"` |
| `actor_rollout_ref.eval_output_path` | Output path for results | `"./eval_outputs/results.parquet"` |
| `actor_rollout_ref.rollout.max_assistant_turns` | Max conversation turns | `16` |
| `trainer.n_gpus_per_node` | Number of GPUs | `2` or `4` |
| `validation.pass_at_k` | Pass@k evaluation | `1` (default) or `4` |
| `validation.data_source` | Filter by data source | `"web_research_GAIA"` |

### Sampling Parameters

```bash
actor_rollout_ref.rollout.val_kwargs.top_p=0.95
actor_rollout_ref.rollout.val_kwargs.top_k=20
actor_rollout_ref.rollout.val_kwargs.temperature=0.6
actor_rollout_ref.rollout.val_kwargs.do_sample=True
```

## 📊 Output Files

After evaluation, you'll find:

```
eval_outputs/
└── your_experiment/
    ├── eval.parquet          # Full evaluation results
    ├── metric.json           # Aggregated metrics by data source
    └── per_source_scores.json  # (Pass@k only) Stats per source
```

## 🔍 Examples

See `eval_examples.sh` for various usage scenarios:

1. **Example 1**: 4B model + HuggingFace eval dataset
2. **Example 2**: 8B-SFT model + Local parquet file
3. **Example 3**: Pass@4 evaluation on GAIA subset
4. **Example 4**: Local checkpoint + HF eval dataset

## 🛠️ How It Works

The evaluation script automatically detects the data source type:

1. **If `data.val_files` starts with `[` and ends with `]`**:
   - Parses as a list of local file paths
   - Uses files directly

2. **Otherwise**:
   - Treats as HuggingFace dataset name
   - Downloads and converts to temporary parquet file
   - Uses the temporary file for evaluation

This is handled by the `prepare_data_files()` function in `verl/trainer/async_main_generation_and_eval.py`.

## 📋 Prerequisites

```bash
# Ensure you're in the project root
cd /work/hzeng_umass_edu/ir-research/SynPlanResearch

# Activate environment
conda activate verl-cu128

# Login to HuggingFace (if using HF models/datasets)
huggingface-cli login

# Set up API keys
export WANDB_API_KEY="your_wandb_key"
export SEARCH_API_KEY="your_serper_api_key"
```

## 🐛 Troubleshooting

### Issue: "Dataset not found"
- Check if HuggingFace dataset name is correct
- Ensure you're logged in: `huggingface-cli login`
- Verify you have access to private datasets

### Issue: "File not found"
- Check if local parquet file path is absolute
- Ensure the file exists and is readable

### Issue: "CUDA out of memory"
- Reduce `trainer.n_gpus_per_node`
- Reduce `actor_rollout_ref.rollout.max_model_len`
- Reduce `actor_rollout_ref.rollout.gpu_memory_utilization`

## 📖 Related Scripts

- `multiturn_web_research_sft.sh` - SFT training script
- `eval_web_research_sft_8b.sh` - Legacy 8B evaluation script
