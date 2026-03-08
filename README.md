# Syn-Plan-Research

Multi-turn web research with reinforcement learning. This repository contains code for SFT, RL training, and evaluation of language models that learn to search the web and synthesize answers.

## Models & Datasets on HuggingFace

| Type | Name |
|------|------|
| Model | `hzeng/syn-plan-research-4B` |
| Model | `hzeng/syn-plan-research-4B-sft` |
| Model | `hzeng/syn-plan-research-8B` |
| Model | `hzeng/syn-plan-research-8B-sft` |
| Dataset | `hzeng/syn-plan-research-data-eval` |
| Dataset | `hzeng/syn-plan-research-data-sft` |
| Dataset | `hzeng/syn-plan-research-data-rl` |

---

## 1. Environment Setup

**Prerequisites**: Linux, NVIDIA A100 (or compatible GPU), Conda.

### Step 1: Create conda environment

```bash
conda create -n verl-vllm083 python=3.10 -y
conda activate verl-vllm083
```

### Step 2: Install PyTorch (2.6.0 + CUDA 12.4)

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### Step 3: Install CUDA Toolkit (required for compiling flash-attn)

```bash
conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit
nvcc --version  # verify: release 12.4
```

### Step 4: Install vLLM

```bash
pip install vllm==0.8.3
```

### Step 5: Install flash-attn (compiles CUDA kernels, may take 10-30 min)

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

### Step 6: Install transformers (do NOT use 5.x)

```bash
pip install transformers==4.52.4
```

### Step 7: Install other dependencies

```bash
pip install accelerate==1.12.0 \
            datasets==4.5.0 \
            ray==2.53.0 \
            peft==0.18.1 \
            wandb==0.24.1 \
            hydra-core==1.3.2 \
            xformers==0.0.29.post2 \
            numpy==1.26.4 \
            pandas==2.3.3 \
            tensordict==0.10.0 \
            uvicorn==0.40.0 \
            fastapi==0.128.0 \
            einops==0.8.2
```

### Step 8: Install verl (editable mode)

```bash
cd syn_plan_research/verl
pip install --no-deps -e .
```

> ⚠️ Use `--no-deps` to avoid overwriting the pinned dependencies above.

### Step 9: Install Playwright browser (required for web page crawling)

```bash
python -m playwright install chromium
```

---

## 2. Configuration

All scripts read paths and API keys from a single `.env` file at the repository root.

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
# Where to store data, checkpoints, caches, and eval outputs.
# This can be a separate high-capacity filesystem from your code directory.
DATA_ROOT="/path/to/your/data/storage"

# Serper API key for web search (get one at https://serper.dev/)
SERPER_API_KEY="your_serper_api_key_here"
```

After setting `DATA_ROOT`, the scripts will use the following directory layout automatically:

```
${DATA_ROOT}/
├── .cache/huggingface/          # HuggingFace cache (HF_HOME)
├── data/
│   ├── sft/train.parquet        # SFT training data
│   ├── rl/train.parquet         # RL training data
│   └── eval/validation.parquet  # Evaluation data
├── cache/
│   └── serper_search_cache.jsonl # Web search cache
├── checkpoints/
│   ├── sft/...                  # SFT checkpoints
│   └── rl/...                   # RL checkpoints
└── eval_outputs/                # Evaluation results
```

> **Note**: Each shell script also contains a `conda activate` line. If your conda setup differs, update the `source` and `conda activate` lines in the scripts to match your environment.

---

## 3. Download Data
Download to local parquet files.

```bash
cd syn_plan_research/verl
bash examples/syn_plan_research/download_parquets_to_local.sh
```

This downloads all datasets to `${DATA_ROOT}/data/`.

### (Optional) Download Web Search Cache

We provide a pre-built Serper search cache that contains retrieval results from previously used queries. Using this cache can significantly reduce your Serper API usage and cost.

Download `serper_search_cache.jsonl` from Google Drive: [link](https://drive.google.com/YOUR_FILE_ID)

Then place it at:

```bash
mkdir -p ${DATA_ROOT}/cache
mv serper_search_cache.jsonl ${DATA_ROOT}/cache/
```

---

## 4. Training & Evaluation

All commands should be run from the `syn_plan_research/verl` directory:

```bash
cd syn_plan_research/verl
```

### 4.1 Evaluation

```bash
bash examples/syn_plan_research/eval_syn_plan_research_all.sh
```

This runs:
1. **Pass@1** evaluation on all data sources
2. **Pass@4** evaluation on GAIA

Results are saved to `${DATA_ROOT}/eval_outputs/`.

See [syn_plan_research/verl/examples/syn_plan_research/README.md](syn_plan_research/verl/examples/syn_plan_research/README.md) for detailed configuration options (model paths, sampling parameters, pass@k settings, etc.).

### 4.2 SFT (Supervised Fine-Tuning)

```bash
bash examples/syn_plan_research/sft_syn_plan_research.sh
```

### 4.3 RL (Reinforcement Learning with GRPO)

```bash
bash examples/syn_plan_research/rl_syn_plan_research.sh
```
