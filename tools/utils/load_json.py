import json

json_dir = "data/VTG_data/train.caption_coco_format.json"
# json_dir = "data/VTG_data/activitynet_captions_train.json"


with open(json_dir, 'r') as f:
    exp_dict = json.load(f)

print(1)