import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List
import random
import jsonlines
import pandas as pd
import torch
import torch.nn
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor

from tools.utils import T5PegasusTokenizer


@dataclass
class ImageTextPair:
    image: JpegImageFile
    reference_sentences: List[str]


class ImageCaptionDataset(Dataset):
    def __init__(self, image_path, text_path, visual_path, language_path, max_length):
        super().__init__()
        self.image_path = image_path
        self.text_path = text_path
        self.image_text_pairs = self.read_data(image_path, text_path)
        self.idx2image_id = {
            idx: image_id
            for idx, image_id in enumerate(self.image_text_pairs.keys())
        }
        self.max_length = max_length
        self.tokenizer = T5PegasusTokenizer.from_pretrained(language_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(visual_path)
        self.sample_freq = {}

    def read_data(self, image_path, text_path):
        id2captions = {}
        # read captions
        with jsonlines.open(text_path) as reader:
            for id_texts in reader:
                image_id = id_texts["image_id"]
                texts = id_texts["text"]
                # maybe dataset is test dataset have no captions
                # so return an empty string
                id2captions[image_id] = texts if isinstance(texts, list) else [texts]
        # read images
        image_data = pd.read_csv(image_path, sep="\t", header=None)
        columns = image_data.columns
        image_ids = image_data[columns[0]]
        image_base64 = image_data[columns[1]]

        return {
            str(image_id): ImageTextPair(
                Image.open(BytesIO(base64.urlsafe_b64decode(image))),
                id2captions[str(image_id)]
            )
            for image_id, image in zip(image_ids, image_base64) if id2captions.get(str(image_id))
        }

    def __len__(self):
        return len(self.image_text_pairs)

    def get_item_by_image_id(self, image_id):
        return self.image_text_pairs[image_id]

    def __getitem__(self, item):
        image_id = self.idx2image_id[item]
        image_text_pair: ImageTextPair = self.image_text_pairs[image_id]
        return {
            "image_id": image_id,
            "image_text_pair": image_text_pair
        }

    def sample_captions(self, image_id, captions, mode="train", num=3):
        if mode == "finetune":
            # Sampling the lowest frequency sample
            if self.sample_freq.get(image_id):
                cur_freq = sorted(self.sample_freq.get(image_id).items(), key=lambda x: x[1], reverse=False)
                sample_idx = cur_freq[0][0]
                self.sample_freq[image_id][sample_idx] += 1
            else:
                cur_freq = {idx: 1 for idx in range(len(captions))}
                self.sample_freq[image_id] = cur_freq
                sample_idx = 0
                self.sample_freq[image_id][sample_idx] += 1
            return captions[sample_idx]
        elif mode == "train":
            return random.choice(captions)

    def collate(self, batch, mode="train"):
        batch_images = []
        batch_texts = []
        batch_image_ids = []
        for item in batch:
            image_text_pair = item["image_text_pair"]
            batch_images.append(image_text_pair.image)
            batch_texts.append(image_text_pair.reference_sentences)
            batch_image_ids.append(item["image_id"])

        pixel_values = self.feature_extractor(batch_images, return_tensors="pt").pixel_values
        if mode == "train":
            sampled_captions = [self.sample_captions(image_id=image_id, captions=texts, mode="train") for
                                image_id, texts in
                                zip(batch_image_ids, batch_texts)]
        elif mode == "finetune":
            sampled_captions = [self.sample_captions(image_id=image_id, captions=texts, mode="finetune") for
                                image_id, texts in
                                zip(batch_image_ids, batch_texts)]
        text_labels = self.tokenizer(
            sampled_captions,
            padding=True,
            max_length=self.max_length,
            truncation=True,
        )
        labels = text_labels.input_ids
        decoder_attention_mask = text_labels.attention_mask
        """
        no need to replace 0 to 100 because we use mask_select to select loss item
        -> utils.py / func: mask_select
        -> trainer.py / func: Task_model.ce_loss
        # labels = [
        #     [-100 if token == self.tokenizer.pad_token_id else token for token in label]
        #     for label in labels
        # ]
        """
        labels = torch.LongTensor(labels)
        decoder_attention_mask = torch.LongTensor(decoder_attention_mask)

        return {
            "image_ids": batch_image_ids,
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
            "batch_texts": batch_texts,
            "max_length": decoder_attention_mask.size(1),
        }

    def train_collate(self, batch):
        return self.collate(batch, mode="train")

    def test_collate(self, batch):
        return self.collate(batch, mode="train")

    def train_collate_finetune(self, batch):
        return self.collate(batch, mode="finetune")

    def test_collate_finetune(self, batch):
        return self.collate(batch, mode="finetune")
