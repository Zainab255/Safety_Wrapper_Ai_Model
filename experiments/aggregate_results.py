"""Aggregate per-model CSV results into a combined table and summary.

Usage:
    python experiments/aggregate_results.py              # uses defaults (logs/, experiments/)
    python experiments/aggregate_results.py --logs logs --out experiments --pattern "results_*.csv"

Outputs:
    - {out}/combined_results.csv    : all rows concatenated + metadata columns (model, wrapper, dataset, source_file)
    - {out}/summary_results.csv     : summary metrics per (model, wrapper, dataset)
    - prints the summary table to stdout
"""

from pathlib import Path
import argparse
import glob
import pandas as pd


def parse_filename(file_path: str):
    """Extract model, wrapper, dataset from a filename like:
    results_Qwen1.5-0.5B-Chat_baseline_harmless.csv
    Returns (model, wrapper, dataset)
    """
    name = Path(file_path).stem
    if name.startswith('results_'):
        name = name[len('results_'):]
    parts = name.split('_')
    if len(parts) >= 3:
        dataset = parts[-1]
        wrapper = parts[-2]
        model = '_'.join(parts[:-2])
    else:
        # fallback
        model = name
        wrapper = ''
        dataset = ''
    return model, wrapper, dataset


def to_bool_series(s: pd.Series) -> pd.Series:
    """Robust conversion of various True/False-like values to bool."""
    return s.fillna(False).apply(lambda v: True if str(v).strip().lower() == 'true' else False)


def aggregate(logs_dir: Path, out_dir: Path, pattern: str = 'results_*.csv'):
    logs_dir = Path(logs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(logs_dir.glob(pattern))
    if not files:
        print(f"No files matching {pattern} found in {logs_dir}")
        return 1

    combined_rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Could not read {f}: {e}")
            continue

        model, wrapper, dataset = parse_filename(f.name)
        df['model'] = model
        df['wrapper'] = wrapper
        df['dataset'] = dataset
        df['source_file'] = str(f)

        # Normalize blocked column to boolean
        if 'blocked' in df.columns:
            df['blocked_bool'] = to_bool_series(df['blocked'])
        else:
            df['blocked_bool'] = False

        # Ensure calls column exists and is numeric
        if 'calls' in df.columns:
            df['calls'] = pd.to_numeric(df['calls'], errors='coerce').fillna(0).astype(int)
        else:
            df['calls'] = 0

        combined_rows.append(df)

    if not combined_rows:
        print("No CSVs successfully read.")
        return 1

    combined = pd.concat(combined_rows, ignore_index=True)
    combined_out = out_dir / 'combined_results.csv'
    combined.to_csv(combined_out, index=False)
    print(f"✅ Combined CSV written to {combined_out}")

    # Create a summary table grouped by model/wrapper/dataset
    summary = (
        combined
        .groupby(['model', 'wrapper', 'dataset'])
        .agg(
            num_prompts=('prompt', 'size'),
            num_blocked=('blocked_bool', 'sum'),
            blocked_rate=('blocked_bool', 'mean'),
            avg_calls=('calls', 'mean'),
        )
        .reset_index()
    )

    # Format numeric columns
    summary['blocked_rate'] = (summary['blocked_rate'] * 100).round(2)
    summary['avg_calls'] = summary['avg_calls'].round(2)

    summary_out = out_dir / 'summary_results.csv'
    summary.to_csv(summary_out, index=False)
    print(f"✅ Summary CSV written to {summary_out}")

    # Pretty print the summary
    print('\nSummary table:')
    print(summary.to_string(index=False))

    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', default='logs', help='Directory containing per-model result CSVs')
    p.add_argument('--out', default='experiments', help='Output directory to write aggregated CSVs')
    p.add_argument('--pattern', default='results_*.csv', help='Glob pattern for result CSVs')
    args = p.parse_args()
    exit(aggregate(Path(args.logs), Path(args.out), args.pattern))


if __name__ == '__main__':
    main()

