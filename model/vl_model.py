import torch.nn as nn

from undecorated import undecorated
from types import MethodType


class ViT5P(nn.Module):
    def __init__(self, visual_model, language_model, pseudo_label_num=None):
        super().__init__()
        self.visual_model = visual_model
        self.language_model = language_model
        self.create_generate_with_grad()

        self.visual_size = visual_model.config.hidden_size
        self.language_size = language_model.config.d_model

        # adaptive linear for v & l
        if self.visual_size != self.language_size:
            self.vl_linear = nn.Linear(self.visual_size, self.language_size)

        # enable pseudo topic label mode
        # can not used in inference!!!
        self.pseudo_mode = pseudo_label_num is not None and pseudo_label_num > 0
        if self.pseudo_mode:
            self.pseudo_label_num = pseudo_label_num
            self.pseudo_label_head = nn.Linear(self.visual_size, pseudo_label_num)

    def create_generate_with_grad(self):
        generate_with_grad = undecorated(self.language_model.generate)
        self.language_model.generate_with_grad = MethodType(generate_with_grad, self.language_model)

    def get_image_feature(self, pixel_values):
        return self.visual_model(pixel_values)[0]

    def forward(self, pixel_values, *args, **kwargs):
        """
        :param pixel_values: [batch, patches, hidden]
        """
        image_feature = self.get_image_feature(pixel_values)
        if self.pseudo_mode:
            image_cls = image_feature[:, 0, :]
            image_logits = self.pseudo_label_head(image_cls)
        else:
            image_logits = None

        if self.visual_size != self.language_size:
            image_feature = self.vl_linear(image_feature)

        outputs = self.language_model.forward(input_ids=None, inputs_embeds=image_feature, *args, **kwargs)

        return outputs, image_logits

    def generate(self, pixel_values, *args, **kwargs):
        """
        :param pixel_values: [batch, patches, hidden]
        """
        image_feature = self.get_image_feature(pixel_values)
        outputs = self.language_model.generate(
            inputs_embeds=image_feature,
            *args, **kwargs
        )
        return outputs

    def generate_with_grad(self, pixel_values, *args, **kwargs):
        """
        :param pixel_values: [batch, patches, hidden]
        """
        image_feature = self.get_image_feature(pixel_values)
        outputs = self.language_model.generate_with_grad(
            inputs_embeds=image_feature,
            *args, **kwargs
        )
        return outputs
