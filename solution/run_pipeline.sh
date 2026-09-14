#!/bin/bash
# Пайплайн: метки -> обучение -> самопроверка на h_train.
# Последовательно (память 3.8 ГБ — параллельные запуски OOM).
set -e
cd /home/user/QAOA/solution
/home/user/.venv/bin/python generate_labels.py --steps 250 --restarts 3 > /home/user/QAOA/data/labels_log.txt 2>&1
/home/user/.venv/bin/python train.py > /home/user/QAOA/data/train_log.txt 2>&1
/home/user/.venv/bin/python infer.py --h h_train.npy --out /home/user/QAOA/data/submission_selfcheck.csv > /home/user/QAOA/data/infer_log.txt 2>&1
echo "PIPELINE DONE" > /home/user/QAOA/data/pipeline_done.txt
