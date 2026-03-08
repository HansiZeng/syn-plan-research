import argparse
import json
import hashlib
from pathlib import Path
import os


def make_ckpt_name(timestamp, key_fields):
    """
    根据核心字段生成 checkpoint 路径。
    """
    flat_str = "_".join(f"{k}-{v}" for k, v in key_fields.items() if v is not None)
    flat_str = f"{flat_str}_{timestamp}"
    return flat_str

def init_path_to_name(model_path):
    if "/claude_rewrite_diverse_prompt_tool_soft_think_minstep_3_maxstep_6-train.parquet/" in model_path:
        return "cld-dp-tst-m3-m6"
    elif "/diverse_prompt_tool_soft_think_minstep_3_maxstep_6-train.parquet/" in model_path:
        return "dp-tst-m3-m6"
    elif "/claude_rewrite_tna_diverse_prompt_tool_soft_think_minstep_3_maxstep_6-train.parquet/" in model_path:
        return "cld-tna-dp-tst-m3-m6"
    elif "/train.parquet/" in model_path:
        return "general"
    elif "/init_Qwen3-8B_diverse_prompt_minstep_3_maxstep_6-checkpoints_lr5e-6_ep_10_20250801_005558/" in model_path:
        return "default_div_prompt"
    elif model_path == "Qwen/Qwen3-8B":
        return "scratch_qwen3_8B"
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_files", required=True)
    parser.add_argument("--val_files", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=None)
    parser.add_argument("--max_assistant_turns", type=int, default=None)
    parser.add_argument("--format_score", type=float, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--save_dir", default="checkpoints")  # 可以给默认目录
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--sft_train_batch_size", type=int, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--filter_unfinished", default="False", type=str)
    parser.add_argument("--rollout_n", type=int, default=16)
    parser.add_argument("--rollout_temp", type=float, default=0.6)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--rollout_top_k", type=int, default=20)
    parser.add_argument("--kl_loss_coef", type=float, default=0.001)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--is_baseline", action="store_true", default=False,
                    help="Set baseline mode (default: False).")
    parser.add_argument("--snp_only", type=str, default="True")
    args = parser.parse_args()

    # 构造 metadata 字典（可扩展）
    metadata = vars(args)
    key_fields = {
        "bz": args.train_batch_size,
        "ep": args.total_epochs,
        "format": args.format_score,
        "turns": args.max_assistant_turns,
        "actor_lr": args.actor_lr,
        "rollout_n": args.rollout_n,
        "rollout_temp": args.rollout_temp,
        "rollout_p": args.rollout_top_p,
        "rollout_k": args.rollout_top_k,
        "filter_unfinished": args.filter_unfinished,
        "kl_coef": args.kl_loss_coef
    }
    if "webshaper" in args.train_files:
        key_fields["add_ds"] = "webshaper"
    if "arpo_deep_research_1K" in args.train_files:
        key_fields["ds"] = "adr_1K"
    if "arpo_deep_research_webshaper_1.6K/" in args.train_files:
        key_fields["ds"] = "adr_1K"
    if "odqa_gpqa_webwalker_16K_arpo1.6K" in args.train_files:
        if "repeat_1" in args.train_files:
            key_fields["ds"] = "general_16K_adr_1.6K_repeat1"
        elif "repeat_3" in args.train_files:
            key_fields["ds"] = "general_16K_adr_1.6K_repeat3"
        elif "repeat_5" in args.train_files:
            key_fields["ds"] = "general_16K_adr_1.6K_repeat5"
    if "odqa_gpqa_webwalker_8K_arpo1.6K" in args.train_files:
        if "repeat_1" in args.train_files:
            key_fields["ds"] = "general_8K_adr_1.6K_repeat1"
        elif "repeat_3" in args.train_files:
            key_fields["ds"] = "general_8K_adr_1.6K_repeat3"
        elif "repeat_5" in args.train_files:
            key_fields["ds"] = "general_8K_adr_1.6K_repeat5"
    if args.resume_path:
        key_fields["resume"] = "true"

    init_name = init_path_to_name(args.model_path)
    if init_name:
        key_fields["init"] = init_name

    # for baseline model 
    if args.is_baseline:
        key_fields["baseline"] = "true"
        key_fields["snp_only"] = args.snp_only

    print("🔑 Key fields for checkpoint naming: ", key_fields)
    
    # 构造 checkpoint 路径
    project_dir = os.path.join(args.save_dir, args.project_name)
    os.makedirs(project_dir, exist_ok=True)  # 确保这个目录存在

    # 构建最终 checkpoint 路径
    ckpt_name = make_ckpt_name(args.timestamp, key_fields)
    ckpt_path = Path(os.path.join(project_dir, ckpt_name))
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 写 metadata
    with open(ckpt_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {ckpt_path / 'metadata.json'}")
    print(f"✅ Checkpoint path: {ckpt_path}")
    print(ckpt_name)

if __name__ == "__main__":
    main()
