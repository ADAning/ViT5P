from transformers import BertTokenizer
from functools import partial
import jieba


class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def mask_select(inputs, mask):
    input_dim = inputs.ndim
    mask_dim = mask.ndim
    mask = mask.reshape(-1).bool()
    if input_dim > mask_dim:
        inputs = inputs.reshape((int(mask.size(-1)), -1))[mask]
    else:
        inputs = inputs.reshape(-1)[mask]
    return inputs


def beam_repeat(inputs, num_beams):
    input_size = inputs.size()
    batch_size = input_size[0]
    other_size = input_size[1:]
    inputs = inputs[:, None, ...]
    repeat_size = [1] * inputs.ndim
    repeat_size[1] = num_beams
    inputs = inputs.repeat(repeat_size)
    inputs = inputs.view((batch_size * num_beams,) + other_size)
    return inputs
