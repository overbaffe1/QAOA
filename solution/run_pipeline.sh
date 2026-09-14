#!/bin/bash
# Пайплайн: метки -> обучение -> самопроверка на h_train.
# Последовательно (память песочницы Arena 3.9 ГБ — параллельные запуски OOM),
# поэтому метки считаются кусками по 64 инстанса (--chunk 64).
# Пути — относительно расположения скрипта, python — из PATH (или $PYTHON).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${PYTHON:-python3}"
mkdir -p "$ROOT/data"

"$PY" "$HERE/generate_labels.py" --steps 250 --restarts 3 --chunk 64 \
    > "$ROOT/data/labels_log.txt" 2>&1
"$PY" "$HERE/train.py"                 > "$ROOT/data/train_log.txt"   2>&1
"$PY" "$HERE/infer.py" --h "$ROOT/h_train.npy" \
    --out "$ROOT/data/submission_selfcheck.csv" --profile fast \
                                     > "$ROOT/data/infer_log.txt"     2>&1
echo "PIPELINE DONE" > "$ROOT/data/pipeline_done.txt"
