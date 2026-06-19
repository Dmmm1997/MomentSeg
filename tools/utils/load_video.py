import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

def copy_folder(video, target_dirs, new_target_dirs):
    src = os.path.join(target_dirs, video)
    dst = os.path.join(new_target_dirs, video)
    if os.path.exists(src):
        os.system(f"cp -r {src} {dst}")
        return 1
    return 0

if __name__ == "__main__":
    with open("data/video_datas/mevis/valid/meta_expressions.json", 'r') as f:
        contexts = json.load(f)["videos"]

    target_dirs = "work_dirs/sa2va_qwen3b_mf5-16-16_v50-2-6_charades_videoseg_randomsample/hf_model/evaluation/Annotations"
    new_target_dirs = "work_dirs/sa2va_qwen3b_mf5-16-16_v50-2-6_charades_videoseg_randomsample/hf_model/evaluation/MEVIS/Annotations"
    os.makedirs(new_target_dirs, exist_ok=True)

    dir_list = set(os.listdir(target_dirs))
    valid_videos = [video for video in contexts.keys() if video in dir_list]

    with ProcessPoolExecutor(max_workers=8) as executor:
        func = partial(copy_folder, target_dirs=target_dirs, new_target_dirs=new_target_dirs)
        results = list(tqdm(executor.map(func, valid_videos), total=len(valid_videos)))

    print("Copied:", sum(results))
