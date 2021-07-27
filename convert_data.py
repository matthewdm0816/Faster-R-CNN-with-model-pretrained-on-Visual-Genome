"""Convert image features from bottom up attention to numpy array"""

# Example
# python convert_data.py --imgid_list 'img_id.txt' --input_file 'test.csv' --output_file 'test.npy'

import os
import base64
import csv
import sys
import zlib
import json
import argparse

import numpy as np
from icecream import ic
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--imgid_list",
    default="data/coco_precomp/train_ids.txt",
    help="Path to list of image id",
)
parser.add_argument(
    "--input_file",
    default="/media/data/kualee/coco_bottom_up_feature/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv",
    help="tsv of all image data (output of bottom-up-attention/tools/generate_tsv.py), \
                    where each columns are: [image_id, image_w, image_h, num_boxes, boxes, features].",
)
parser.add_argument(
    "--output_file",
    default="test.npy",
    help="Output file path. the file saved in npy format",
)

parser.add_argument("--output_dir", help="Output instance-wise file path.")

parser.add_argument("--concat_bb", action="store_true", help="Whether to concat BB feature")


opt = parser.parse_args()
print(opt)


meta = []
feature = dict()
tags = dict()
for line in open(opt.imgid_list):
    sid = line.strip()
    # ic(sid)
    meta.append(sid)
    feature[sid] = None
    tags[sid] = None

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["image_id", "image_w", "image_h", "tags", "num_boxes", "boxes", "features"]

if __name__ == "__main__":
    with open(opt.input_file, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
        total_n_regions = 0
        for item in tqdm(reader):
            # ic(item)
            # exit(0)
            item["image_id"] = int(item["image_id"])
            item["image_h"] = int(item["image_h"])
            item["image_w"] = int(item["image_w"])
            item["num_boxes"] = int(item["num_boxes"])
            for field in ["boxes", "features"]:
                data = item[field]
                # buf = base64.decodestring(data)
                buf = base64.b64decode(data[1:])
                temp = np.frombuffer(buf, dtype=np.float32)
                item[field] = temp.reshape((item["num_boxes"], -1))
                # ic(field, item[field])
            if isinstance(item["image_id"], int):
                item["image_id"] = str(item["image_id"])
                # ic(item["image_id"], item["image_id"] in feature)
            if item["image_id"] in feature:
                # concat BB to Region features
                # N * D => N * (D + D^)
                # ic(item["features"].shape, item["boxes"].shape)
                n_regions = item["features"].shape[0]
                total_n_regions += n_regions
                if opt.concat_bb:
                    feature[item["image_id"]] = np.concatenate(
                        [
                            item["features"],
                            item["boxes"],
                        ],
                        axis=-1,
                    )
                else:
                    feature[item["image_id"]] = item["features"]
            if item["image_id"] in tags:
                tags[item["image_id"]] = item["tags"]
    ic(total_n_regions / len(feature.keys()))
    # exit(0)
    # data_out = np.stack([feature[sid] for sid in meta], axis=0)
    # print (data_out)
    # print("Final numpy array shape:", data_out.shape)
    # np.save(opt.output_file, data_out)
    # NOTE: save instance-wise feature
    # to .../123.npz
    for sid in tqdm(meta):
        to_save = {"x": feature[sid]} # NOTE: To be compatible with MCAN dataloaders
        out_path = os.path.join(opt.output_dir, "%s.npz" % sid)
        # ic(out_path)
        # with open(out_path, 'wb') as f:
        #     np.save(f, to_save)
        np.savez_compressed(out_path, x=feature[sid], tags=tags[sid])
