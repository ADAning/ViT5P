#!/usr/bin/env bash
# predict
python3 ../test.py --min_length 15 --max_length 88 --num_beams 6 \
  --ckpt_path ../pretrained_models/checkpoints/epoch=72-step=45625_stage2.ckpt \
  --visual_path ../pretrained_models/facebook/vit-base-patch16-224 \
  --language_path ../pretrained_models/imxly/t5-pegasus \
  --image_path ../datasets/image64/{}_imgs.tsv \
  --caption_path ../datasets/caption/{}_caption.jsonl \
  --out_file ../output/temp_predict_files/long.jsonl
python3 ../test.py --min_length 0 --max_length 80 --num_beams 6 \
  --ckpt_path ../pretrained_models/checkpoints/epoch=72-step=45625_stage2.ckpt \
  --visual_path ../pretrained_models/facebook/vit-base-patch16-224 \
  --language_path ../pretrained_models/imxly/t5-pegasus \
  --image_path ../datasets/image64/{}_imgs.tsv \
  --caption_path ../datasets/caption/{}_caption.jsonl \
  --out_file ../output/temp_predict_files/middle.jsonl
python3 ../tes.py --min_length 0 --max_length 60 --num_beams 4 \
  --ckpt_path ../pretrained_models/checkpoints/epoch=72-step=45625_stage2.ckpt \
  --visual_path ../pretrained_models/facebook/vit-base-patch16-224 \
  --language_path ../pretrained_models/imxly/t5-pegasus \
  --image_path ../datasets/image64/{}_imgs.tsv \
  --caption_path ../datasets/caption/{}_caption.jsonl \
  --out_file ../output/temp_predict_files/short.jsonl
# ensemble
python3 ../tools/ensemble.py --file_path ../output/temp_predict_files --en_dict_path ../datasets/case_trans_dict.json --out_file ../output/ensemble_predictions.jsonl
