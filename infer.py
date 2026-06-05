import torch
import torch._dynamo
# 完全禁用 dynamo 优化以避免兼容性问题
torch._dynamo.config.disable = True

import json
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from voxcpm import VoxCPM
import soundfile as sf



# 情绪到英文描述的映射
EMOTION_MAP = {
    "angry": "angry tone",
    "happy": "happy tone",
    "sad": "sad tone",
    "fearful": "fearful tone",
    "disgusted": "disgusted tone",
    "surprised": "surprised tone",
    "neutral": "neutral tone"
}


# ====================================================================
# 构建说话人信息
# ====================================================================
def load_speaker_info(enroll_dir="enroll_four_spk"):
    """
    根据 enroll_four_spk 目录下的音频文件和 spk2text.jsonl 构建说话人信息

    Returns:
        dict: {spk_id: {"wav_path": "path/to/wav", "text": "transcript"}}
    """
    enroll_path = Path(enroll_dir)

    if not enroll_path.exists():
        print(f"⚠️  警告: 说话人音频目录不存在: {enroll_path}")
        return {}

    # 读取 spk2text.jsonl
    spk2text_file = enroll_path / "spk2text.jsonl"
    if not spk2text_file.exists():
        print(f"⚠️  警告: spk2text.jsonl 文件不存在: {spk2text_file}")
        return {}

    spk_info = {}

    # 读取说话人文本映射
    spk2text_map = {}
    with open(spk2text_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                spk_id = data['spk']
                text = data['text']
                spk2text_map[spk_id] = text

    # 构建说话人信息字典
    for spk_id, text in spk2text_map.items():
        wav_path = enroll_path / f"{spk_id}.wav"
        if wav_path.exists():
            spk_info[spk_id] = {
                "wav_path": str(wav_path),
                "text": text
            }
            print(f"  ✓ 加载说话人 {spk_id}: {wav_path.name}")
        else:
            print(f"  ✗ 警告: 说话人 {spk_id} 的音频文件不存在: {wav_path}")

    return spk_info


# ====================================================================
# 主程序
# ====================================================================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="VoxCPM2 批量推理脚本")

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="test100_emo_v2.jsonl",
        help="输入的 JSONL 文件路径 (默认: test100_emo_v2.jsonl)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output_voxcpm2",
        help="输出目录路径 (默认: output_voxcpm2)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="./pretrained_models/VoxCPM2",
        help="模型路径 (默认: ./pretrained_models/VoxCPM2)"
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        default="voxcpm2",
        help="模型名称 (默认: voxcpm2)"
    )
    parser.add_argument(
        "--enroll-dir",
        type=str,
        default="enroll_four_spk",
        help="说话人音频目录 (默认: enroll_four_spk)"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.0,
        help=f"CFG 值 (默认: 2.0)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help=f"推理步数 (默认: 10)"
    )
    parser.add_argument(
        "--no-emotion",
        action="store_true",
        help="禁用情绪提示"
    )

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    output_path = Path(args.output)
    print(f"✅ 输出目录: {output_path.absolute()}")

    # 加载说话人信息
    print(f"\n📁 加载说话人信息...")
    ENROLL_SPK = load_speaker_info(args.enroll_dir)
    if ENROLL_SPK:
        print(f"✅ 成功加载 {len(ENROLL_SPK)} 个说话人信息")
        print(ENROLL_SPK)
    else:
        print(f"⚠️  未加载说话人信息，将使用默认音色")

    # 打印 GPU 信息
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        print(f"\n🖥️  GPU: [{gpu_index}] {gpu_name}")
        torch.cuda.reset_peak_memory_stats(gpu_index)
    else:
        gpu_index = None
        print("\n🖥️  GPU: 未检测到 CUDA，将使用 CPU 推理")

    # 加载模型
    print(f"\n🔄 加载模型: {args.model}")
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False)
    print("✅ 模型加载完成")

    # 读取 JSONL 文件
    print(f"\n📖 读取数据: {args.input}")
    data_list = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))
    print(f"✅ 共读取 {len(data_list)} 条数据")

    # 确定是否使用情绪提示
    use_emotion = not args.no_emotion

    # 批量推理
    print(f"\n🚀 开始批量推理...")
    print(f"   情绪提示: {'启用' if use_emotion else '禁用'}")
    print(f"   说话人克隆: {'启用 ({} 个说话人)'.format(len(ENROLL_SPK)) if ENROLL_SPK else '禁用'}")
    print(f"   CFG值: {args.cfg}")
    print(f"   推理步数: {args.steps}")
    print()

    success_count = 0
    error_count = 0
    error_list = []
    total_infer_time = 0.0
    total_audio_duration = 0.0

    for item in tqdm(data_list, desc="推理进度"):
        try:
            utt = item['utt']
            text = item['text']
            spk = item['spk']
            emotion = item['emotion']

            # 构建输出文件名: {utt}_{spk}_{emotion}_{args.name}.wav
            output_filename = f"{utt}_{spk}_{emotion}_{args.name}.wav"
            output_filepath = output_path / output_filename

            # 如果文件已存在，跳过
            if output_filepath.exists():
                tqdm.write(f"⏭️  跳过已存在: {output_filename}")
                success_count += 1
                continue

            # 准备生成文本
            if use_emotion and emotion in EMOTION_MAP:
                # 添加情绪提示
                # emotion_desc = EMOTION_MAP[emotion]
                # generation_text = f"({emotion_desc}){text}"
                generation_text = f"({emotion} tone){text}"
            else:
                # 不使用情绪提示
                generation_text = text

            print("generation_text: ", generation_text)

            # 生成音频
            t0 = time.perf_counter()
            wav = model.generate(
                # prompt_wav_path=ENROLL_SPK.get(spk, {}).get('wav_path', None),  # 可选,用于克隆说话人音色
                # prompt_text=ENROLL_SPK.get(spk, {}).get('text', None),  # 可选,用于提供说话人文本提示
                reference_wav_path=ENROLL_SPK.get(spk, {}).get('wav_path', None),  # 可选,用于提高相似度
                text=generation_text,
                cfg_value=args.cfg,
                inference_timesteps=args.steps,
            )
            total_infer_time += time.perf_counter() - t0

            # 保存音频
            sf.write(str(output_filepath), wav, model.tts_model.sample_rate)
            total_audio_duration += len(wav) / model.tts_model.sample_rate
            success_count += 1

        except Exception as e:
            error_count += 1
            error_info = {
                'utt': item.get('utt', 'unknown'),
                'error': str(e)
            }
            error_list.append(error_info)
            tqdm.write(f"❌ 错误 [{item.get('utt', 'unknown')}]: {str(e)}")

    # 统计结果
    print(f"\n{'='*60}")
    print(f"📊 推理完成统计:")
    print(f"{'='*60}")
    print(f"  ✅ 成功: {success_count}/{len(data_list)}")
    print(f"  ❌ 失败: {error_count}/{len(data_list)}")
    print(f"  📁 输出目录: {output_path.absolute()}")
    if total_audio_duration > 0:
        rtf = total_infer_time / total_audio_duration
        print(f"  ⏱️  推理总时长: {total_infer_time:.2f}s")
        print(f"  🔊 音频总时长: {total_audio_duration:.2f}s")
        print(f"  📈 整体 RTF:   {rtf:.4f}")
    if gpu_index is not None:
        peak_mem = torch.cuda.max_memory_allocated(gpu_index) / 1024 ** 3
        print(f"  💾 显存峰值:   {peak_mem:.2f} GB")

    # 如果有错误，保存错误日志
    if error_list:
        error_log_path = output_path / "error_log.json"
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump(error_list, f, ensure_ascii=False, indent=2)
        print(f"  📝 错误日志: {error_log_path}")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
