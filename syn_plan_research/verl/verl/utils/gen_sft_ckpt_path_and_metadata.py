import argparse
import json
import hashlib
from pathlib import Path
import os

# def make_short_ckpt_name(project_name, timestamp, key_fields):
#     """
#     根据核心字段生成短 checkpoint 路径。
#     """
#     flat_str = "_".join(f"{k}={v}" for k, v in key_fields.items())
#     short_hash = hashlib.md5(flat_str.encode()).hexdigest()[:8]
#     return f"{project_name}_{timestamp}_{short_hash}"

def make_ckpt_name(timestamp, key_fields):
    """
    根据核心字段生成 checkpoint 路径。
    """
    flat_str = "_".join(f"{k}-{v}" for k, v in key_fields.items() if v is not None)
    flat_str = f"{flat_str}_{timestamp}"
    return flat_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_files", required=True)
    parser.add_argument("--val_files", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--total_epochs", type=int, default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--save_dir", default="checkpoints")  # 可以给默认目录
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    # 构造 metadata 字典（可扩展）
    metadata = vars(args)
    key_fields = {
        "bz": args.train_batch_size,
        "ep": args.total_epochs,
        "lr": args.learning_rate,
    }
    
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
