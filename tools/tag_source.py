"""
标注caption来源
"""
from jsonlines import jsonlines


def process(source_data):
    processed_data = []
    for temp_data in source_data:
        temp_data['text'] = temp_data['text'].replace("<SEP>", "")
        if temp_data['text'][-1] != "。":
            temp_data['text'] += "。"
        processed_data.append({"image_id": temp_data['image_id'], "text": temp_data['text']})
    return processed_data


def save_data(json_data):
    dataset_path = "../output/tag_source.jsonl"
    with jsonlines.open(dataset_path, 'w') as fw:
        for line in json_data:
            fw.write(line)
    print("saved dataset to ", dataset_path)


with jsonlines.open("16.854_god.jsonl") as reader:
    target = process(list(reader))

with jsonlines.open("../best_file/long.jsonl") as reader:
    long_source = process(list(reader))

with jsonlines.open("../best_file/middle.jsonl") as reader:
    middle_source = process(list(reader))

with jsonlines.open("../best_file/short.jsonl") as reader:
    short_source = process(list(reader))
tag_data = []
for i, caption_data in enumerate(target):
    source_file = ''
    caption_data["text"] = caption_data["text"].replace("<SEP>", "")
    if caption_data["text"][-1] != "。":
        caption_data["text"] += "。"
    if caption_data["text"] == short_source[i]["text"]:
        source_file = "short.jsonl"
    elif caption_data["text"] == middle_source[i]["text"]:
        source_file = "middle.jsonl"
    elif caption_data["text"] == long_source[i]["text"]:
        source_file = "long.jsonl"
    tag_data.append({"image_id": caption_data["image_id"], "source": source_file, "text": caption_data["text"]})
save_data(tag_data)
