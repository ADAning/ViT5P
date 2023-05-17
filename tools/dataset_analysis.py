"""
原数据集分析
"""
import json

import numpy as np
import pandas as pd

train_data_path = '../datasets/IC_valid.jsonl'


def load_datasets(data_path):
    _data = []
    with open(data_path, 'r') as f:
        raw_data = f.readlines()
        for data_instance in raw_data:
            _data.append(json.loads(data_instance))
    return _data


def analysis(_data):
    text_nums = []  # 文本数量
    signal_text_len = []  # 单个文本长度
    text_len = []  # 描述文本长度
    for data_instance in _data:
        text_nums.append(len(data_instance['text']))
        for i in [len(s) for s in data_instance['text']]:
            signal_text_len.append(i)
        text_len.append(len(' '.join(data_instance['text'])))
    return np.array(text_nums), np.array(signal_text_len), np.array(text_len)


data = load_datasets(train_data_path)
np_text_nums, np_signal_text_len, np_text_len = analysis(data)
pd.DataFrame(np_text_nums).describe()
print()
