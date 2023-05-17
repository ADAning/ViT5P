"""
base64 to img
"""
import base64
import os
from io import BytesIO

import pandas as pd
import tqdm as tqdm
from PIL import Image

file_types = ["test"]

dataset_base = "../dataset_raw"

for file_type in file_types:
    data = pd.read_csv('../datasets/{}_imgs.tsv'.format(file_type), sep='\t', header=None)
    # print(data)
    columns = data.columns
    image_base64 = data[columns[1]]
    image_ids = data[columns[0]]
    for image_id, image in zip(image_ids, image_base64):
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        output_file = os.path.join(dataset_base, file_type, "{}.png".format(image_id))
        image.save(output_file)
