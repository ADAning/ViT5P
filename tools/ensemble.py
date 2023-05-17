"""
预测结果集成模块
"""
import argparse
import json
import os
import re

import numpy as np
import pylcs as pylcs
from jsonlines import jsonlines
from tqdm import tqdm


def load_dict(_path):
    """
    加载词表
    """
    return json.load(open(_path, 'r'))


def process_infer_data(_read_path, _en_dict):
    """
    清洗数据
    """
    target_json = {}
    with jsonlines.open(_read_path) as reader:
        for id_texts in reader:
            image_id = id_texts["image_id"]
            text = id_texts["text"]
            _en_word = re.findall('[a-zA-Z]+', text)
            for w in _en_word:
                if _en_dict.get(w):
                    text.replace(w, _en_dict[w])
            target_json[image_id] = text

    return target_json


def read_file(file_dir, _en_dict):
    """
    扫描目录 读取文件
    """
    all_captions = {}
    for cur_file in os.listdir(file_dir):
        print(os.listdir(file_dir))
        if cur_file.__contains__("jsonl"):
            cur_caption_data = process_infer_data(os.path.join(file_dir, cur_file), _en_dict)
            for k, v in cur_caption_data.items():
                if all_captions.get(k):
                    all_captions[k].append(v)
                else:
                    all_captions[k] = [v]

    # 去重
    for k, v in all_captions.items():
        all_captions[k] = list(set(v))
    return all_captions


def gather_join(texts, idxs):
    """
    取出对应的text，拼接
    """
    return ' '.join([texts[i] for i in idxs])


def pseudo_summary(texts):
    """
    集成策略
    """
    source_idxs, target_idxs = list(range(len(texts))), []
    # while True:
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
    target = gather_join(texts, target_idxs)

    return target


def process(source_data, out_file):
    processed_data = []
    for k, v in tqdm(source_data.items()):
        target = pseudo_summary(v)
        target = target.replace("<SEP>", "")
        if target[-1] != "。":
            target += "。"
        processed_data.append({"image_id": k, "text": target})
    save_data(json_data=processed_data, out_file=out_file)


def save_data(json_data, out_file):
    with jsonlines.open(out_file, 'w') as fw:
        for line in json_data:
            fw.write(line)
    print("saved file to ", out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--en_dict_path', type=str, help='english word  dict path')
    parser.add_argument('--file_path', type=str,
                        help='predicted file path')
    parser.add_argument('--out_file', type=str,
                        help='final predicted file path')
    args = parser.parse_args()
    file_path = args.file_path
    en_dict_path = args.en_dict_path
    out_file = args.out_file
    en_dict = load_dict(en_dict_path)
    captions = read_file(file_dir=file_path, _en_dict=en_dict)
    process(captions, out_file)
    print('done')
