"""
数据集预处理
"""
import os.path

import jsonlines
import numpy as np
import pylcs
from tqdm import tqdm

summary_rate = 0.5  # 主题句占比
min_length = 25  # 单个句子最小长度
max_sen_num = 16  # 执行集成算法的 caption 句子数量下限
target_len = 8  # caption 最大句子数量


def load_data(mode):
    root_path = "../dataset/caption/"
    dataset_path = os.path.join(root_path, mode + "_caption.jsonl")
    id2captions = {}
    with jsonlines.open(dataset_path) as reader:
        for id_texts in reader:
            image_id = id_texts["image_id"]
            texts = id_texts["text"]
            texts = [sen + '<SEP>' for sen in texts if len(sen) >= min_length]
            if len(texts) == 0:
                texts = id_texts["text"]
            if len(texts) > max_sen_num:
                texts = texts[:max_sen_num]
            id2captions[image_id] = texts
    return id2captions


def save_data(mode, json_data, version):
    root_path = "../dataset/caption/"
    dataset_path = os.path.join(root_path, mode + "_caption_" + version + ".jsonl")
    with jsonlines.open(dataset_path, 'w') as fw:
        for line in json_data:
            fw.write(line)
    print("saved dataset to ", dataset_path)


def gather_join(texts, idxs):
    """
    取出对应的text，拼接
    """
    return ''.join([texts[i] for i in idxs])


def pseudo_summary(texts):
    """
    集成算法
    """
    source_idxs, target_idxs = list(range(len(texts))), []
    while True:
        sims = []
        for i in source_idxs:
            new_source_idxs = [j for j in source_idxs if j != i]
            new_target_idxs = sorted(target_idxs + [i])
            new_source = gather_join(texts, new_source_idxs)
            new_target = gather_join(texts, new_target_idxs)
            sim = pylcs.lcs(new_source, new_target)
            sims.append(sim)
        new_idx = source_idxs[np.argmax(sims)]
        source_idxs.remove(new_idx)
        target_idxs = sorted(target_idxs + [new_idx])
        source = gather_join(texts, source_idxs)
        target = gather_join(texts, target_idxs)
        if (
                len(source_idxs) == 1 or
                1.0 * len(target) / (len(source) + len(target)) > summary_rate
        ):
            break
    return source.split('<SEP>')[:-1], target.split('<SEP>')[:-1]


def process(mode="train", version="v1"):
    source_data = load_data(mode)
    processed_data = []
    for k, v in tqdm(source_data.items()):
        if len(v) >= max_sen_num:
            source, target = pseudo_summary(v)
        else:
            target = v

        # 按长度排序,截取 top target_len
        if len(target) > target_len:
            target.sort(key=lambda sen_len: len(sen_len), reverse=True)
            target = target[:target_len]
        new_target = []
        for tg in target:
            tg = tg.replace("<SEP>", "")
            if tg[-1] != "。":
                tg += "。"
            new_target.append(tg)
        processed_data.append({"image_id": k, "text": new_target})
    save_data(mode=mode, json_data=processed_data, version=version)


if __name__ == '__main__':
    process("valid", version="v5")
    process("train", version="v5")

    print("done")
