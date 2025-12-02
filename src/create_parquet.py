import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
from loaders.load_dataset import list_sequence_ids, load_radar_for_sequence
from feature_engineering import engineer_features, save_parquet

def build_dataset(dataset_root: Path, out_path: Path, max_sequences: int = 20):
    seq_ids = list_sequence_ids(dataset_root)
    dfs = []

    print(f"Found {len(seq_ids)} sequences.")
    print(f"Processing first {min(max_sequences, len(seq_ids))} sequences...")

    for seq in seq_ids[:max_sequences]:
        print(f" → Loading sequence: {seq}")
        df = load_radar_for_sequence(dataset_root, seq)
        df["sequence"] = seq

        df = engineer_features(df)   # <<< uses your code
        dfs.append(df)

    df_full = pd.concat(dfs, ignore_index=True)

    print(f"\nFinal dataset size: {len(df_full):,} rows")
    save_parquet(df_full, out_path)

def main():
    DATASET_ROOT = Path("/home/rraricha/Datasets/RadarScenes/RadarScenes/data")
    OUT_PATH = Path("data/engineered_features.parquet")

    build_dataset(DATASET_ROOT, OUT_PATH)
"""
if __name__ == "__main__":
    main()"""
