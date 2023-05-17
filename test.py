import os
import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import ViTModel, T5ForConditionalGeneration

from data.dataset import ImageCaptionDataset
from trainer import TaskModel
from model.vl_model import ViT5P
from tools.utils import T5PegasusTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--min_length', type=int, default=0, help='captions min_length')
    parser.add_argument('--max_length', type=int, default=80, help='captions max_length')
    parser.add_argument('--num_beams', type=int, help='num_beams')
    parser.add_argument('--out_file', type=str, default="predictions.jsonl",
                        help='outfile path and name')

    # pretrained model
    parser.add_argument('--ckpt_path', type=str,
                        help='outfile path and name')
    parser.add_argument('--visual_path', type=str,
                        help='vit model path')
    parser.add_argument('--language_path', type=str,
                        help='t5 pegasus model path')
    # datasets
    parser.add_argument('--image_path', type=str,
                        help='outfile path and name')
    parser.add_argument('--caption_path', type=str,
                        help='caption dataset path')

    args = parser.parse_args()

    out_file = args.out_file
    min_length = args.min_length
    max_length = args.max_length
    num_beams = args.num_beams
    # pretrained model
    ckpt_path = args.ckpt_path
    visual_path = args.visual_path
    language_path = args.language_path
    # datasets
    image_path = args.image_path
    caption_path = args.caption_path

    visual_model = ViTModel.from_pretrained(visual_path)
    language_model = T5ForConditionalGeneration.from_pretrained(language_path)
    tokenizer = T5PegasusTokenizer.from_pretrained(language_path)
    model = ViT5P(
        visual_model=visual_model,
        language_model=language_model,
        pseudo_label_num=0
    )

    visual_lr = 1e-5
    language_lr = 3e-4
    epochs = 80
    gradient_clip = False

    repetition_penalty = 1.2

    test_dataset = ImageCaptionDataset(
        image_path.format("test"),
        caption_path.format("test"),
        visual_path,
        language_path,
        max_length,
    )

    batch_size = 48
    num_wokers = 4

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_wokers,
        collate_fn=test_dataset.test_collate,
        pin_memory=True,
        persistent_workers=True,
    )

    task_model = TaskModel.load_from_checkpoint(
        ckpt_path,
        model=model,
        tokenizer=tokenizer,
        visual_lr=visual_lr,
        language_lr=language_lr,
        min_length=min_length,
        max_length=max_length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        out_file=out_file
    )

    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        precision=32,
    )

    trainer.test(task_model, test_loader)
