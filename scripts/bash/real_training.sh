#!/bin/bash
# Bash script to run the Python training script

python main.py \
  --dataset cifar100 \
  --data_dir data/raw \
  --output_dir results \
  --device cuda \
  --batch_size 512 \
  --num_epochs 120 \
  --lr 0.4 \
  --weight_decay 1e-4 \
  --num_workers 4