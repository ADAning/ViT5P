import os
import itertools
import jsonlines

import numpy as np
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from transformers import AdamW
from tools.utils import mask_select, beam_repeat
from criterion.cider import Cider


class TaskModel(pl.LightningModule):
    def __init__(self,
                 model,
                 tokenizer,
                 visual_lr,
                 language_lr,
                 max_length,
                 warmup_ratio=0.,
                 weight_decay=0.,
                 num_beams=3,
                 training_mode="xe",
                 label_smooth=0.1,
                 out_file="predictions.jsonl"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.visual_lr = visual_lr
        self.language_lr = language_lr

        self.num_beams = num_beams

        self.label_smooth = label_smooth
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        self.training_mode = training_mode
        self.cider = Cider()
        self.save_hyperparameters(ignore=['model', 'tokenizer'])
        self.out_file = out_file

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def ce_loss(self, inputs, targets, mask, label_smooth):
        mask = mask[:, 1:]
        inputs = inputs[:, :-1]
        targets = targets[:, 1:]
        inputs = mask_select(inputs, mask)
        targets = mask_select(targets, mask)
        loss = F.cross_entropy(inputs, targets, label_smoothing=label_smooth)
        return loss

    def cider_score_loss(self, sequences, mean_log_probs, batch_gts):
        bsz = sequences.size(0) // self.num_beams
        # rewards
        sequences = sequences[:, 1:]
        caps_gen = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        caps_gen = [s.replace(' ', '') for s in caps_gen]
        caps_gt = list(itertools.chain(*([c, ] * self.num_beams for c in batch_gts)))

        gens = {}
        gts = {}

        assert len(caps_gen) == len(caps_gt)
        for idx, (gt, gen) in enumerate(zip(caps_gt, caps_gen)):
            gens[idx] = [gen, ]
            gts[idx] = gt
        reward = self.cider.compute_score(gts, gens)[1].astype(np.float32)
        reward = torch.from_numpy(reward)
        reward = reward.view(bsz, self.num_beams).to(mean_log_probs.device)
        reward_baseline = torch.mean(reward, -1, keepdim=True)

        loss = -mean_log_probs * (reward - reward_baseline)
        loss = loss.mean()
        reward = reward.mean().item()
        reward_baseline = reward_baseline.mean().item()
        return loss, reward, reward_baseline

    def compute_epoch_loss(self, outputs):
        epoch_steps = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
            epoch_loss = sum(output["loss"].sum() for output in outputs) / epoch_steps
        else:
            epoch_loss = sum([output["loss"] for output in outputs]) / epoch_steps
        return epoch_loss

    def change_training_mode_from_epoch(self):
        # change training strategy after some epochs
        if self.current_epoch < 12:
            self.training_mode = "xe"
        else:
            self.training_mode = "cider"

    def step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        decoder_attention_mask = batch["decoder_attention_mask"]

        if self.training_mode == "xe":
            language_output, image_logits = self(
                pixel_values=pixel_values,
                labels=labels,
                decoder_input_ids=labels,
                decoder_attention_mask=decoder_attention_mask,
            )
            loss = self.ce_loss(language_output.logits, labels, decoder_attention_mask, label_smooth=self.label_smooth)

            return {
                "loss": loss,
            }

        elif self.training_mode == "cider":
            batch_texts = batch["batch_texts"]
            max_length = batch["max_length"]
            batch_size = labels.size(0)

            output = self.model.generate_with_grad(
                pixel_values,
                eos_token_id=self.tokenizer.sep_token_id,
                decoder_start_token_id=self.tokenizer.cls_token_id,
                use_cache=True,
                max_length=max_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
            sequences = output.sequences
            scores = output.scores
            assert scores[0].ndim == 2
            scores = torch.stack(scores, dim=1)
            labels = labels[:, 1:]
            labels = beam_repeat(labels, self.num_beams)
            if (labels.size()[0] != batch_size * self.num_beams) or (labels.size()[1] != scores.size()[1]):
                print(labels.size(), scores.size())
                print(labels)
                print(scores)
                return
            log_probs = torch.gather(scores, dim=-1, index=labels.unsqueeze(-1))
            log_probs = log_probs.squeeze(-1)

            decoder_attention_mask = decoder_attention_mask[:, 1:]
            decoder_attention_mask = beam_repeat(decoder_attention_mask, self.num_beams)
            decoder_attention_mask = decoder_attention_mask.view(batch_size, self.num_beams, -1)
            log_probs = log_probs.view(batch_size, self.num_beams, -1)
            log_probs = log_probs * decoder_attention_mask
            log_probs = log_probs.sum(-1) / decoder_attention_mask.sum(-1)

            loss, reward, reward_baseline = self.cider_score_loss(sequences, log_probs, batch_texts)

            return {
                "loss": loss,
                "reward": reward,
                "reward_baseline": reward_baseline,
            }

        else:
            raise NotImplementedError("Wrong training mode with {}".format(self.training_mode))

    def training_step(self, batch, batch_idx):
        step_result = self.step(batch, batch_idx)
        loss = step_result["loss"]
        # record all training metrics
        self.log_dict({" ".join(["train", k]): v for k, v in step_result.items()})

        return {
            "loss": loss,
        }

    def training_epoch_end(self, outputs):
        train_epoch_loss = self.compute_epoch_loss(outputs)
        self.log("train epoch loss", train_epoch_loss)

    def validation_step(self, batch, batch_idx):
        batch_image_ids = batch["image_ids"]
        pixel_values = batch["pixel_values"]
        batch_texts = batch["batch_texts"]

        gen = {}
        gts = {}
        image_ids = {}

        with torch.no_grad():
            valid_loss = self.step(batch, batch_idx)["loss"]
            model_pred = self.model.generate(
                pixel_values,
                eos_token_id=self.tokenizer.sep_token_id,
                decoder_start_token_id=self.tokenizer.cls_token_id,
                num_beams=self.num_beams,
                use_cache=True,
                max_length=self.max_length,
            )
            model_pred = model_pred[:, 1:].detach().cpu().numpy()
            model_pred = self.tokenizer.batch_decode(model_pred, skip_special_tokens=True)
            model_pred = [s.replace(' ', '') for s in model_pred]

            for i, (gts_i, gen_i, image_id) in enumerate(zip(batch_texts, model_pred, batch_image_ids)):
                gen['%d_%d' % (batch_idx, i)] = [gen_i, ]
                gts['%d_%d' % (batch_idx, i)] = gts_i
                image_ids['%d_%d' % (batch_idx, i)] = image_id

        return {
            "image_ids": image_ids,
            "gen": gen,
            "gts": gts,
            "loss": valid_loss,
        }

    def validation_epoch_end(self, outputs):
        valid_epoch_loss = self.compute_epoch_loss(outputs)
        self.log("valid epoch loss", valid_epoch_loss)
        gen = {}
        gts = {}
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        for output in outputs:
            gen.update(output["gen"])
            gts.update(output["gts"])

        cider_d, _ = self.cider.compute_score(gts, gen)
        self.log("valid CIDEr-D", cider_d, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        gen = {}
        image2batch = {}
        if self.trainer.num_devices > 1:
            outputs = self.all_gather(outputs)
        for output in outputs:
            gen.update(output["gen"])
            image2batch.update({v: k for k, v in output["image_ids"].items()})

        test_result = []
        for image_id in sorted(image2batch.keys()):
            result = {"image_id": image_id, "text": gen[image2batch[image_id]][0]}
            test_result.append(result)

        log_dir = os.path.join(self.trainer.log_dir, "predictions.jsonl")
        with jsonlines.open(log_dir, mode="w") as writer:
            writer.write_all(test_result)
        with jsonlines.open(self.out_file, mode="w") as writer:
            writer.write_all(test_result)
        print("save log  to ", log_dir)
        print("save predictions  to ", self.out_file)

    def configure_optimizers(self):
        """
        Different learning rate for visual encoder and language decoder.
        """
        no_decay = ["bias", "LayerNorm.weight, layer_norm.weight"]
        visual = self.model.visual_model
        language = self.model.language_model
        optimizer_grouped_parameters = [
            {'params': [p for n, p in visual.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and p.requires_grad],
             'lr': self.visual_lr,
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in visual.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'lr': self.visual_lr,
             'weight_decay': 0.},
            {'params': [p for n, p in language.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and p.requires_grad],
             'lr': self.language_lr,
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in language.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             'lr': self.language_lr,
             'weight_decay': 0.},
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        return optimizer
