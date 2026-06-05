"""
Plot training metrics from merged CSV files.
Usage: python data_process/plot_training_metrics.py outputs/xxx_merged.csv [output.png]
       python data_process/plot_training_metrics.py outputs/  [output.png]  # all CSVs in dir
"""
import sys
import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def extract_step(model_name: str) -> int:
    """Extract step number after 'goodstep', e.g. 'goodstep0004000' -> 4000."""
    m = re.search(r'goodstep0*(\d+)', model_name)
    if not m:
        raise ValueError(f"Cannot parse step from model_name: {model_name!r}")
    return int(m.group(1))


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['step'] = df['model_name'].apply(extract_step)
    df['asr_acc'] = 1.0 - df['wer']
    df = df.sort_values('step').reset_index(drop=True)
    return df


def plot_metrics(df: pd.DataFrame, title: str, out_path: str):
    metrics = [
        ('asr_acc',      'ASR Acc (1-WER)'),
        ('avg_P808_MOS', 'P.808 MOS'),
        ('avg_sim',      'Speaker Sim'),
        ('emo_acc',      'Emotion Acc'),
        ('emo_score',    'Emotion Score'),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for col, label in metrics:
        if col in df.columns:
            ax.plot(df['step'], df[col], marker='o', markersize=4, label=label)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python data_process/plot_training_metrics.py <csv_or_dir> [output.png]")
        sys.exit(1)

    src = args[0]
    if os.path.isdir(src):
        csv_files = sorted(glob.glob(os.path.join(src, '*_merged.csv')))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(src, '*.csv')))
    else:
        csv_files = [src]

    if not csv_files:
        print(f"No CSV files found in {src}")
        sys.exit(1)

    for csv_path in csv_files:
        df = load_csv(csv_path)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        if len(args) >= 2 and len(csv_files) == 1:
            out_path = args[1]
        else:
            out_path = os.path.join(os.path.dirname(csv_path), base + '_metrics.png')
        plot_metrics(df, title=base, out_path=out_path)


if __name__ == '__main__':
    main()
