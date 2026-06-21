import pandas as pd
import pytest
from export_dataset import export_dataset


def _write_working_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_skip_unlabeled_rows(tmp_path):
    input_csv = tmp_path / "working.csv"
    output_csv = tmp_path / "final.csv"
    _write_working_csv(input_csv, [
        {"comment_id": "1", "text": "post one", "label_suggested": "analysis", "label": "analysis", "prelabeled": True, "notes": ""},
        {"comment_id": "2", "text": "post two", "label_suggested": "hot_take", "label": "", "prelabeled": True, "notes": ""},
        {"comment_id": "3", "text": "post three", "label_suggested": "reaction", "label": "reaction", "prelabeled": True, "notes": ""},
    ])
    export_dataset(str(input_csv), str(output_csv))
    result = pd.read_csv(output_csv)
    assert len(result) == 2
    assert set(result["text"]) == {"post one", "post three"}


def test_preserve_text_exactly(tmp_path):
    raw_text = "CRASHHHH did you SEE that 💀"
    input_csv = tmp_path / "working.csv"
    output_csv = tmp_path / "final.csv"
    _write_working_csv(input_csv, [
        {"comment_id": "1", "text": raw_text, "label_suggested": "reaction", "label": "reaction", "prelabeled": True, "notes": ""},
    ])
    export_dataset(str(input_csv), str(output_csv))
    result = pd.read_csv(output_csv)
    assert result["text"].iloc[0] == raw_text


def test_column_schema(tmp_path):
    input_csv = tmp_path / "working.csv"
    output_csv = tmp_path / "final.csv"
    _write_working_csv(input_csv, [
        {"comment_id": "1", "text": "some post", "label_suggested": "hot_take", "label": "hot_take", "prelabeled": True, "notes": "tricky"},
    ])
    export_dataset(str(input_csv), str(output_csv))
    result = pd.read_csv(output_csv)
    assert list(result.columns) == ["text", "label", "notes"]


def test_nan_label_excluded(tmp_path):
    input_csv = tmp_path / "working.csv"
    output_csv = tmp_path / "final.csv"
    df = pd.DataFrame([
        {"comment_id": "1", "text": "nan post", "label_suggested": "analysis", "label": float("nan"), "prelabeled": True, "notes": ""},
        {"comment_id": "2", "text": "valid post", "label_suggested": "analysis", "label": "analysis", "prelabeled": True, "notes": ""},
    ])
    df.to_csv(input_csv, index=False)
    export_dataset(str(input_csv), str(output_csv))
    result = pd.read_csv(output_csv)
    assert len(result) == 1
    assert result["text"].iloc[0] == "valid post"
