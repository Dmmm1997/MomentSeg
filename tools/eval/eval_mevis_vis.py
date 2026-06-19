###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
import time
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as cocomask
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
NUM_WOEKERS = 128

def overlay_mask_on_frame(frame, mask, alpha=0.5, dim=2):
    """
    在原始帧上叠加mask
    :param frame: 原始帧 (H, W, 3)
    :param mask:  二值mask (H, W), 需要是0-255范围
    :param alpha: 透明度
    :return: 带mask的帧
    """
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, dim] = mask  # 仅给红色通道赋值，形成红色mask
    overlayed = cv2.addWeighted(frame, 1 - alpha, colored_mask, alpha, 0)
    return overlayed


def eval_queue(q, rank, out_dict, mevis_pred_path, generate_video):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{mevis_pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f'{mevis_pred_path}/{vid_name}/{exp}/00000.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']
        expression = vid['expressions'][exp]["exp"]

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids: # 查看当前针 所有指代目标的 mask 相加
                mask_rle = mask_dict[str(anno_id)][frame_idx] # mask_dict[str(anno_id)]表示当前anno_id在每一帧的mask
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)
            try:
                pred_masks[frame_idx] = cv2.imread(f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)
            except:
                print("error is : ")
                print(f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png')

        if generate_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = f'{mevis_pred_path}/{vid_name}/'.replace("inference", "video")
            os.makedirs(video_path, exist_ok=True)
            video_path_base = video_path+ f'{exp}_{expression.replace(" ", "_")}'
            video_writer = cv2.VideoWriter(video_path_base + ".mp4", fourcc, 15.0, (w//2, h))
            
            for pred_mask_, gt_mask_, frame_name_ in zip(pred_masks, gt_masks, vid['frames']):
                ori_path = f'data/mevis/valid_u/JPEGImages/{vid_name}/{frame_name_}.jpg'
                frame = cv2.imread(ori_path)
                overlayed_frame_pred = overlay_mask_on_frame(frame, pred_mask_, dim=1, alpha=0.5)
                overlayed_frame_gt = overlay_mask_on_frame(frame, gt_mask_ * 255, dim=2, alpha=0.5)
                overlayed_frame = np.vstack((overlayed_frame_pred, overlayed_frame_gt))
                overlayed_frame = cv2.resize(overlayed_frame, (w//2,h))
                video_writer.write(overlayed_frame)
                
            video_writer.release()
                
        
        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mevis_exp_path", type=str, default="data/mevis/valid_u/meta_expressions.json")
    parser.add_argument("--mevis_mask_path", type=str, default="data/mevis/valid_u/mask_dict.json")
    parser.add_argument("--mevis_pred_path", type=str, default="outputs/dshmp_val/inference")
    parser.add_argument("--save_name", type=str, default="mevis_test.json")
    parser.add_argument("--generate_video", action="store_true", default=True, help="enable video generation with overlaid masks")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))['videos']
    mask_dict = json.load(open(args.mevis_mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mevis_pred_path, args.generate_video))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]
    
    save_dir = os.path.dirname(args.save_name)
    result_save_path = os.path.join(save_dir, "results.txt")
    J = np.mean(j)
    F = np.mean(f)
    J_F = (J + F) / 2
    with open(result_save_path, "w") as file:
        file.write(f"J: {J}\n")
        file.write(f"F: {F}\n")
        file.write(f"J&F: {J_F}\n")

    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))
