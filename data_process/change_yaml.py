import argparse
import yaml
from pathlib import Path
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True)
    parser.add_argument("--config", default="conf/voxcpm_v2/voxcpm_finetune_all_hindi.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_name = args.data_name

    config_path = Path(args.config)
    if args.output is None:
        output_path = config_path
    else:
        output_path = Path(args.output)

    # 1️⃣ 读取 YAML（保留结构）
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2️⃣ 修改关键字段
    cfg["train_manifest"] = f"datas/{data_name}/train.jsonl"
    if os.path.exists(f"datas/{data_name}/dev.jsonl"):
        cfg["val_manifest"] = f"datas/{data_name}/dev.jsonl"
    else:
        cfg["val_manifest"] = None
    cfg["save_path"] = f"/mnt/speech-work/asr/wantongtang/lym/VoxCPM/exps/{data_name}/checkpoints/finetune_all"
    cfg["tensorboard"] = f"/mnt/speech-work/asr/wantongtang/lym/VoxCPM/exps/{data_name}/logs/finetune_all"

    # 3️⃣ 写回 YAML（保持结构清晰）
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"✔ YAML updated: {output_path}")

if __name__ == "__main__":
    main()