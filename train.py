import argparse

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import ViTModel, T5ForConditionalGeneration
from model.vl_model import ViT5P
from data.dataset import ImageCaptionDataset
from tools.utils import T5PegasusTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_length', type=int, default=0, help='captions min_length')
    parser.add_argument('--max_length', type=int, default=80, help='captions max_length')
    parser.add_argument('--total_batch_size', type=int, default=48, help='captions max_length')
    parser.add_argument('--num_beams', default=3, type=int, help='num_beams')
    parser.add_argument('--epochs', type=int, default=80, help='captions max_length')
    parser.add_argument('--out_file', default="predictions.jsonl", type=str,
                        help='outfile path and name')
    # pretrained model
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
    max_length = args.max_length
    total_batch_size = args.total_batch_size
    num_beams = args.num_beams
    epochs = args.epochs

    # pretrained model
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

    train_dataset = ImageCaptionDataset(
        image_path.format("train"),
        caption_path.format("train"),
        visual_path,
        language_path,
        max_length,
    )

    valid_dataset = ImageCaptionDataset(
        image_path.format("valid"),
        caption_path.format("valid"),
        visual_path,
        language_path,
        max_length,
    )

    test_dataset = ImageCaptionDataset(
        image_path.format("test"),
        caption_path.format("test"),
        visual_path,
        language_path,
        max_length,
    )

    num_devices = 1
    batch_size = total_batch_size // num_devices
    num_wokers = 8
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_wokers,
        collate_fn=train_dataset.train_collate,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_wokers,
        collate_fn=valid_dataset.test_collate,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_wokers,
        collate_fn=test_dataset.test_collate,
        pin_memory=True,
        persistent_workers=True,
    )

    checkpoint = ModelCheckpoint(
        save_weights_only=True,
        save_on_train_epoch_end=False,
        monitor="valid CIDEr-D",
        mode="max",
        verbose=True,
        save_top_k=6,
    )

    visual_lr = 1e-5
    language_lr = 3e-4

    gradient_clip = False

    task_model = TaskModel(
        model=model,
        tokenizer=tokenizer,
        visual_lr=visual_lr,
        language_lr=language_lr,
        max_length=max_length,
        training_mode="xe",
        num_beams=num_beams,
        out_file=out_file,
    )

    if num_devices > 1:
        print("only support 1 gpu! please set num_devices=1!")

    else:
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[checkpoint],
            num_sanity_val_steps=0,
            accelerator="gpu",
            devices=1,
            precision=32,
            gradient_clip_val=1.0 if gradient_clip else None,
        )
        trainer.fit(task_model, train_loader, valid_loader)
        trainer.test(task_model, test_loader, ckpt_path="best")
