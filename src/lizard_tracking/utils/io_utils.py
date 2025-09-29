from pathlib import Path
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_dataframe(df: pd.DataFrame, out_csv: Path, out_parquet: Path | None = None):
    out_csv = Path(out_csv); ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    pq_path = None
    if out_parquet:
        out_parquet = Path(out_parquet); ensure_dir(out_parquet.parent)
        try:
            df.to_parquet(out_parquet, index=False)
            pq_path = out_parquet
        except Exception as e:
            print(f"[warn] could not save parquet: {e}")
    return out_csv, pq_path