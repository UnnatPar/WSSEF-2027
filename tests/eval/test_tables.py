from eval.tables import build_summary_table, save_summary_table


def make_results():
    return {
        "PET+scratch": {
            "pretrain_objective": "none",
            "probe_angular_error": None,
            "finetune_angular_error": 20.0,
        },
        "NeutrinoPET": {
            "pretrain_objective": "MAE",
            "probe_angular_error": 18.0,
            "finetune_angular_error": 11.0,
        },
        "NeutrinoJEPAPET": {
            "pretrain_objective": "JEPA",
            "probe_angular_error": 12.0,
            "finetune_angular_error": 8.0,
        },
    }


def test_build_summary_table_has_expected_columns_and_rows():
    df = build_summary_table(make_results())
    assert len(df) == 3
    assert list(df.columns) == [
        "Model", "Pretrain objective", "Probe angular error (deg)", "Finetune angular error (deg)",
    ]
    jepa_row = df[df["Model"] == "NeutrinoJEPAPET"].iloc[0]
    assert jepa_row["Finetune angular error (deg)"] == 8.0


def test_save_summary_table_writes_csv_and_markdown(tmp_path):
    df = build_summary_table(make_results())
    csv_path = tmp_path / "table1.csv"
    md_path = tmp_path / "table1.md"
    save_summary_table(df, str(csv_path), str(md_path))
    assert csv_path.exists()
    assert md_path.exists()
    assert "NeutrinoJEPAPET" in md_path.read_text()
