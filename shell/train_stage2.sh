#!/usr/bin/env bash
# stage2 fine-tune
# Fine-tune on the v5 dataset using the ckpt with the highest cider score in one stage
# Please manually replace the ckpt path, or use our training ckpt fine-tuning
python3 ../train_continue.py --min_length 0 --max_length 80 --num_beams 4 --total_batch_size 48 --epochs 80 \
  --ckpt_path ../pretrained_models/checkpoints/epoch=79-step=50000_stage1.ckpt \
  --visual_path ../pretrained_models/google/vit-base-patch16-224 \
  --language_path ../pretrained_models/imxly/t5-pegasus \
  --image_path ../datasets/image64/{}_imgs.tsv \
  --caption_path ../datasets/caption/{}_caption_v5.jsonl \
  --out_file ../log/fine_tune.jsonl
