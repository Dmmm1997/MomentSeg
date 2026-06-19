###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings
import sys
import os

sys.path.insert(0, '/root/paddlejob/workspace/env_run/daiming/project/MyPub/MomentSeg')

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os.path as osp
import time
import argparse
import json
import numpy as np
from pycocotools import mask as cocomask
from third_parts.revos.utils.metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
import os
import cv2

NUM_WOEKERS = 64

def overlay_mask_on_frame(frame, mask, alpha=0.5, dim=2):
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

        pred = mask_pred_dict[vid_name][exp]

        h, w = pred['prediction_masks'][0]['size']
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']
        expression = vid['expressions'][exp]["exp"]

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_mask = cocomask.decode(pred['prediction_masks'][frame_idx])
            pred_masks[frame_idx] += pred_mask


        if generate_video:
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            video_path = f'{mevis_pred_path}/{vid_name}/'
            os.makedirs(video_path, exist_ok=True)
            video_path_base = video_path+ f'{exp}'
            txt_path = os.path.join(video_path, "expression.txt")
            with open(txt_path, "a") as f:
                f.write("{}\n{}\n".format(exp, expression))
            if w>h:
                video_writer = cv2.VideoWriter(video_path_base + ".webm", fourcc, 10.0, (w//2, h))
            else:
                video_writer = cv2.VideoWriter(video_path_base + ".webm", fourcc, 10.0, (w, h//2))
            for pred_mask_, gt_mask_, frame_name_ in zip(pred_masks, gt_masks, vid['frames']):
                ori_path = f'./data/ref_sav/valid/videos/{vid_name}/{frame_name_}.jpg'
                frame = cv2.imread(ori_path)
                overlayed_frame_pred = overlay_mask_on_frame(frame, pred_mask_ * 255, dim=1, alpha=0.5)
                overlayed_frame_gt = overlay_mask_on_frame(frame, gt_mask_ * 255, dim=2, alpha=0.5)
                if w>h:
                    overlayed_frame = np.vstack((overlayed_frame_pred, overlayed_frame_gt))
                    overlayed_frame = cv2.resize(overlayed_frame, (w//2,h))
                else:
                    overlayed_frame = np.hstack((overlayed_frame_pred, overlayed_frame_gt))
                    overlayed_frame = cv2.resize(overlayed_frame, (w,h//2))
                video_writer.write(overlayed_frame)
                
            video_writer.release()

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument("--mevis_exp_path", type=str,
                        default="data/ref_sav/valid/meta_expressions_valid.json")
    parser.add_argument("--mevis_mask_path", type=str,
                        default="data/ref_sav/valid/mask_dict.json")
    parser.add_argument("--save_name", type=str, default="refsav.json")
    parser.add_argument("--generate_video", action="store_true", help="enable video generation with overlaid masks")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))['videos']
    mask_dict = json.load(open(args.mevis_mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    mask_pred = json.load(open(args.pred_path))
    mask_pred_dict  = mp.Manager().dict(mask_pred)

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])
    
    video_pred_path = os.path.join(osp.dirname(args.pred_path), "refsav_video_vis")
    os.makedirs(video_pred_path, exist_ok=True)

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, video_pred_path, args.generate_video))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict, video_pred_path, args.generate_video)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    output_path = osp.join(osp.dirname(args.pred_path), args.save_name)
    results = {
        'J': round(100 * float(np.mean(j)), 2),
        'F': round(100 * float(np.mean(f)), 2),
        'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))

