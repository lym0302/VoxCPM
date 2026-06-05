#!/usr/bin/env python3
"""
Process JSONL files for VoxCPM training.

This script combines format conversion and type fixing:
1. Format conversion:
   - wav_path -> audio
   - emo -> add emotion tone prefix to text (if not "noemo")
   - dur -> duration
2. Type fixing:
   - spk -> string (speaker ID should always be string)
   - duration/dur -> float
   - dataset_id -> int
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def process_jsonl(
    input_path: str,
    output_path: str = None,
    convert_format: bool = True,
    fix_types: bool = True,
    fix_spk: bool = True
):
    """
    Process JSONL file with format conversion and type fixing.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (if None, use input.processed.jsonl)
        convert_format: Whether to convert format (wav_path->audio, emo, etc.)
        fix_types: Whether to fix data types
        fix_spk: Whether to convert spk to string (default: True)
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Use input path as output if not specified
    if output_path is None:
        output_file = input_file.with_suffix('.processed.jsonl')
        print(f"No output path specified, will save to: {output_file}")
    else:
        output_file = Path(output_path)

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Count lines for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # Statistics
    processed_count = 0
    emotion_count = 0
    spk_fixed = 0
    duration_fixed = 0
    dataset_id_fixed = 0
    error_count = 0
    skipped_count = 0

    print(f"Processing {total_lines:,} lines...")
    print(f"  - Format conversion: {'enabled' if convert_format else 'disabled'}")
    print(f"  - Type fixing: {'enabled' if fix_types else 'disabled'}")

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(tqdm(fin, total=total_lines, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                # ============ Step 1: Format Conversion ============
                if convert_format:
                    converted = {}

                    # Convert wav_path -> audio
                    if 'wav_path' in data:
                        converted['audio'] = data['wav_path']
                    elif 'audio' in data:
                        converted['audio'] = data['audio']
                    else:
                        print(f"\nWarning: Line {line_num} has no 'wav_path' or 'audio' field, skipping")
                        skipped_count += 1
                        continue

                    # Handle text and emotion
                    text = data.get('text', '').replace('[laughter]', '[laughing]')
                    emo = data.get('emo', 'noemo')

                    if emo and emo.lower() != 'noemo':
                        converted['text'] = f"({emo} tone){text}"
                        emotion_count += 1
                    else:
                        converted['text'] = text

                    # Convert dur -> duration
                    if 'dur' in data:
                        converted['duration'] = data['dur']
                    elif 'duration' in data:
                        converted['duration'] = data['duration']

                    # Preserve other fields (dataset_id, ref_audio, spk, etc.)
                    for key in data:
                        if key not in ['wav_path', 'audio', 'text', 'emo', 'dur', 'duration']:
                            converted[key] = data[key]

                    data = converted

                # ============ Step 2: Type Fixing ============
                if fix_types:
                    # Fix spk field: convert to string
                    if fix_spk and 'spk' in data:
                        if not isinstance(data['spk'], str):
                            data['spk'] = str(data['spk'])
                            spk_fixed += 1

                    # Fix duration field: convert to float
                    if 'duration' in data:
                        if not isinstance(data['duration'], (int, float)):
                            try:
                                data['duration'] = float(data['duration'])
                                duration_fixed += 1
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(data['duration'], int):
                            # Convert int to float for consistency
                            data['duration'] = float(data['duration'])

                    # Fix dur field: convert to float (in case it still exists)
                    if 'dur' in data:
                        if not isinstance(data['dur'], (int, float)):
                            try:
                                data['dur'] = float(data['dur'])
                                duration_fixed += 1
                            except (ValueError, TypeError):
                                pass
                        elif isinstance(data['dur'], int):
                            data['dur'] = float(data['dur'])

                    # Fix dataset_id field: convert to int
                    if 'dataset_id' in data:
                        if not isinstance(data['dataset_id'], int):
                            try:
                                data['dataset_id'] = int(data['dataset_id'])
                                dataset_id_fixed += 1
                            except (ValueError, TypeError):
                                pass

                # Write processed line
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"\nError: Line {line_num} is not valid JSON: {e}")
                error_count += 1
                # Write original line to preserve data
                fout.write(line + '\n')
            except Exception as e:
                print(f"\nError processing line {line_num}: {e}")
                error_count += 1
                fout.write(line + '\n')

    print(f"\n✓ Processing complete!")
    print(f"  - Total lines processed: {processed_count:,}")
    if convert_format:
        print(f"  - Lines with emotion control: {emotion_count:,}")
    if fix_types:
        print(f"  - spk fields fixed: {spk_fixed:,}")
        print(f"  - duration fields fixed: {duration_fixed:,}")
        print(f"  - dataset_id fields fixed: {dataset_id_fixed:,}")
    if skipped_count > 0:
        print(f"  - Lines skipped: {skipped_count:,}")
    if error_count > 0:
        print(f"  - Errors encountered: {error_count:,}")
    print(f"  - Output saved to: {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL files for VoxCPM training (format conversion + type fixing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full processing (format conversion + type fixing)
  python process_jsonl.py -i input.jsonl -o output.jsonl

  # Only format conversion
  python process_jsonl.py -i input.jsonl -o output.jsonl --no-fix-types

  # Only type fixing
  python process_jsonl.py -i input.jsonl -o output.jsonl --no-convert-format

  # Don't fix spk field
  python process_jsonl.py -i input.jsonl -o output.jsonl --no-fix-spk

  # Process in place (creates .processed.jsonl)
  python process_jsonl.py -i train.jsonl
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output JSONL file (default: input.processed.jsonl)'
    )

    parser.add_argument(
        '--no-convert-format',
        action='store_true',
        help='Skip format conversion (wav_path->audio, emo, etc.)'
    )

    parser.add_argument(
        '--no-fix-types',
        action='store_true',
        help='Skip type fixing (spk, duration, dataset_id)'
    )

    parser.add_argument(
        '--no-fix-spk',
        action='store_true',
        help='Do not convert spk field to string'
    )

    args = parser.parse_args()

    try:
        process_jsonl(
            args.input,
            args.output,
            convert_format=not args.no_convert_format,
            fix_types=not args.no_fix_types,
            fix_spk=not args.no_fix_spk
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
