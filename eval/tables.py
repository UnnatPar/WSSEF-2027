import pandas as pd


def build_summary_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Table 1. results: {method_name: {"pretrain_objective": str,
    "probe_angular_error": float, "finetune_angular_error": float}}.
    Backs Fig. 1 with exact numbers for quick reference on the poster."""
    rows = []
    for method, values in results.items():
        rows.append({
            "Model": method,
            "Pretrain objective": values.get("pretrain_objective", "none"),
            "Probe angular error (deg)": values.get("probe_angular_error"),
            "Finetune angular error (deg)": values.get("finetune_angular_error"),
        })
    return pd.DataFrame(rows)


def save_summary_table(df: pd.DataFrame, csv_path: str, markdown_path: str | None = None) -> None:
    df.to_csv(csv_path, index=False)
    if markdown_path is not None:
        with open(markdown_path, "w") as f:
            f.write(df.to_markdown(index=False))
