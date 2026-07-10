#!/usr/bin/env bash
# Downloads the IceCube Kaggle competition data and lays it out exactly as
# train/data.py's KaggleParquetDataset expects: batch_dir/batch_N.parquet,
# sensor_geometry.csv, and train_meta.parquet, read directly (no GraphNeT
# conversion step -- verified by direct testing that GraphNeT's own
# ParquetDataset cannot read these raw files anyway).
# Requires `kaggle` CLI configured with API credentials (~/.kaggle/kaggle.json).
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

kaggle competitions download -c icecube-neutrinos-in-deep-ice -p "$DATA_DIR"
unzip -o "$DATA_DIR/icecube-neutrinos-in-deep-ice.zip" -d "$DATA_DIR"

echo "Data ready under $DATA_DIR/:"
echo "  train/                660 parquet batch files"
echo "  train_meta.parquet"
echo "  sensor_geometry.csv"
