"""Export the human-reviewed working_annotations.csv to the final dataset.

Only rows where the human has filled the `label` column are included.
Text is preserved verbatim — no lowercasing, stripping, or emoji removal.
"""
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
WORK_PATH = DATA_DIR / "working_annotations.csv"
OUT_PATH = DATA_DIR / "takemeter_dataset.csv"


def export_dataset(input_path: str, output_path: str) -> None:
    import pandas as pd

    df = pd.read_csv(input_path, dtype=str)

    labeled = df[df["label"].notna() & (df["label"].str.strip() != "")]
    result = labeled[["text", "label", "notes"]].copy()
    result.to_csv(output_path, index=False)

    skipped = len(df) - len(labeled)
    print(f"Exported {len(labeled)} rows → {output_path}")
    if skipped:
        print(f"Skipped {skipped} unlabeled rows (label column empty).")


def main() -> None:
    if not WORK_PATH.exists():
        sys.exit(f"ERROR: {WORK_PATH} not found — complete human review first")
    export_dataset(str(WORK_PATH), str(OUT_PATH))


if __name__ == "__main__":
    main()
