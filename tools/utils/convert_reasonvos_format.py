import json

# # Define file paths
source_json_path = 'data/video_datas/reasonvos/meta_expressions.json'
target_json_path = 'data/video_datas/reasonvos/meta_expressions_format.json'

# Load the source JSON file
with open(source_json_path, 'r') as f:
    source_json = json.load(f)

# Initialize the target structure
target_json = {
    "videos": {}
}

anno_id = 0
# Iterate through each video in the source JSON
for id, (vid_id, vid_data) in enumerate(source_json["videos"].items()):
    target_json["videos"][vid_id] = {
        "source": vid_data["source"],
        "vid_id": id,
        "frames": vid_data["frames"],  # Copy frames from source
        "expressions": {}
    }

    # Convert expressions from the source format to the target format
    anno_id_set = set()
    for exp in vid_data["expressions"]:
        exp_id = exp["exp_id"]
        target_json["videos"][vid_id]["expressions"][str(exp_id)] = {
            "exp": exp["exp_text"],
            "obj_id": [int(exp["obj_id"])],  # Convert object ID to integer
            "anno_id": anno_id  # Use exp_id as the annotation ID
        }
        if int(exp["obj_id"]) not in anno_id_set:
            anno_id_set.add(int(exp["obj_id"]))
            if len(anno_id_set) > 1:
                anno_id += 1
    anno_id += 1

# Save the transformed JSON to the target file
with open(target_json_path, 'w') as f:
    json.dump(target_json, f, indent=4)




source_anno_dir = "data/video_datas/reasonvos/Annotations"
target_mask_pkl = "data/video_datas/reasonvos/mask_dict.json"

with open(target_json_path, 'r') as f:
    target_json = json.load(f)["videos"]

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def png_to_coco_rle(png_path):
    # 读取 PNG 掩码图（0/255 或者灰度）
    mask = np.array(Image.open(png_path).convert("L"), dtype=np.uint8)
    
    # 二值化（确保只有 0 和 1）
    mask = (mask > 0).astype(np.uint8)
    
    # 转成 Fortran 顺序（COCO 要求列优先）
    rle = maskUtils.encode(np.asfortranarray(mask))
    # rle = maskUtils.frPyObjects(mask.tolist(), mask.shape[0], mask.shape[1])
    
    # pycocotools 返回的 'counts' 是 bytes，需要转为 utf-8 字符串
    rle["counts"] = rle["counts"].decode("utf-8")
    
    return rle

from collections import defaultdict
import os
mask_dict = defaultdict(list)
for vid_id, vid_data in target_json.items():
    frames = vid_data["frames"]
    source = vid_data["source"]
    for exp_id, expressions in vid_data["expressions"].items():
        anno_id = expressions["anno_id"]
        obj_id = expressions["obj_id"][0]
        if source == "vipseg":
            obj_id = str(obj_id).zfill(9)
        if anno_id not in mask_dict:
            for frame in frames:
                anno_dir_name = f"{source}_{vid_id}_{obj_id}"
                anno_img_file = os.path.join(source_anno_dir, anno_dir_name, f"{frame}.png")
                mask_dict[anno_id].append(png_to_coco_rle(anno_img_file))

# Save the transformed JSON to the target file
with open("data/video_datas/reasonvos/mask_dict.json", 'w') as f:
    json.dump(mask_dict, f, indent=4)
