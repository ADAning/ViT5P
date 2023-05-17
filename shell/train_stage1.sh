#!/usr/bin/env bash
# stage1 train
python3 ../train.py --min_length 0 --max_length 80 --num_beams 3 --total_batch_size 48 --epochs 80 \
  --visual_path ../pretrained_models/google/vit-base-patch16-224 \
  --language_path ../pretrained_models/imxly/t5-pegasus \
  --image_path ../datasets/image64/{}_imgs.tsv \
  --caption_path ../datasets/caption/{}_caption.jsonl \
  --out_file ../log/val.jsonl
