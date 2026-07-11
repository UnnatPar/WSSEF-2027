#!/usr/bin/env bash
# Downloads only the IceCube Kaggle batch files that configs/*.yaml actually
# reference, not the full competition dataset (660 files, ~118GB, ~130M
# events -- too big for a Colab disk). The default ranges are the union of
# every train_batches/val_batches/test_batches across configs/*.yaml:
# 1-50 (pretrain) + 595-660 (val+test) = 116 batches, ~23M events.
# Requires `kaggle` CLI configured with API credentials (~/.kaggle/kaggle.json).
set -euo pipefail

DATA_DIR="${1:-data}"
BATCH_RANGES="${2:-1-50 595-660}"

mkdir -p "$DATA_DIR/train"

kaggle competitions download -c icecube-neutrinos-in-deep-ice -f train_meta.parquet -p "$DATA_DIR" --force
kaggle competitions download -c icecube-neutrinos-in-deep-ice -f sensor_geometry.csv -p "$DATA_DIR" --force

for range in $BATCH_RANGES; do
    start="${range%-*}"
    end="${range#*-}"
    for i in $(seq "$start" "$end"); do
        f="$DATA_DIR/train/batch_${i}.parquet"
        [ -f "$f" ] && continue
        kaggle competitions download -c icecube-neutrinos-in-deep-ice \
            -f "train/batch_${i}.parquet" -p "$DATA_DIR/train" --force
    done
done

n_batches=$(ls "$DATA_DIR/train"/*.parquet 2>/dev/null | wc -l)
echo "Data ready under $DATA_DIR/:"
echo "  train/                $n_batches parquet batch files"
echo "  train_meta.parquet"
echo "  sensor_geometry.csv"
