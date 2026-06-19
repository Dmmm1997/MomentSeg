import os
import json

import mmengine

import copy
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as _mask
import torch
import cv2

from .base_eval_dataset import BaseEvalDataset
from projects.qwenvl_sam2.evaluation.utils import REFER, Summary, AverageMeter, intersectionAndUnionGPU, master_only


# SEG_PROMPT = "<image>\n {} Please segment it in this image."
SEG_PROMPT = "<image>\n Please segment {} in this image."
SEG_PROMPT_v2 = "<image>\n Please segment {} in this image."
# SEG_PROMPT_v2 = "<image>\n {} Please output segmentation mask."

class ReasonSegDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder
        ):
        super().__init__()
        items = self.json_file_preprocess(image_folder)
        self.items = items

    def __len__(self):
        return len(self.items)

    def real_len(self):
        return len(self.items)

    def polygon_to_mask(points, height, width):
        """
        points: list of [x, y] 坐标
        height, width: mask 尺寸
        """
        # 创建全 0 mask
        mask = Image.new('L', (width, height), 0)
        
        # 转成 tuple 格式
        polygon = [(x, y) for x, y in points]
        
        # 在 mask 上画多边形并填充为 1
        ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)
        
        # 转成 torch.Tensor（0/1）
        return torch.from_numpy(np.array(mask, dtype=np.uint8))

    def json_file_preprocess(self, image_folder):
        files = os.listdir(image_folder)
        json_files = []
        for file in files:
            if file.endswith('.json'):
                json_files.append(file)
        
        items = []
        for idx, json_file in enumerate(json_files):
            with open(os.path.join(image_folder, json_file), 'r') as f:
                data_ = json.load(f)
            image_name = json_file.replace(".json", ".jpg")
            item = {
                "image": os.path.join(image_folder, image_name),
                "img_id": idx,
                "json_file": os.path.join(image_folder, json_file),
            }
            items.append(item)

        return items

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.items[index])
        image = video_obj_info['image']
        json_file = video_obj_info['json_file']
        img_id = video_obj_info['img_id']

        data_dict = {}
        
        frame_image = Image.open(image).convert('RGB')
        gt_masks, exps, is_sentence = get_mask_from_json(json_file, frame_image)       
        gt_masks = torch.from_numpy(gt_masks)  
        exps = [exps[0]]

        gt_masks_ = []
        exps_ = []
        for exp in exps:
            if is_sentence:
                exps_.append(SEG_PROMPT_v2.format(exp))
            else:
                exps_.append(SEG_PROMPT.format(exp))
            gt_masks_.append(gt_masks)

        data_dict["image"] = frame_image
        data_dict['gt_masks'] = torch.stack(gt_masks_,dim=0)
        data_dict['text'] = exps_
        data_dict['img_id'] = img_id

        return data_dict


    @master_only
    def evaluate(self, result, work_dir):
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        for pred_dict in result:
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)
            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            for i_item, _mask in enumerate(_masks):
                if _mask is None:
                    continue

                _target = _targets[i_item: i_item+1]
                for prediction, target in zip(_mask, _target):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )
                    intersection += intersect
                    union += union_
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0

            if isinstance(intersection, torch.Tensor):
                intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
                accuracy_iou = accuracy_iou.cpu().numpy() / _targets.shape[0]
            else:
                intersection, union, accuracy_iou = 0.0, 0.0, 0.0

            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

        cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                       'pixel_union': trackers["union"].sum[1],
                       'gIoU': trackers["gIoU"].avg[1],
                       'mask_counts': trackers["gIoU"].count,
                       }
        class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
        global_iou = cur_results['gIoU']

        print('============================================', 'current')
        print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
        print('============================================', 'current')
        return {'CIoU': class_iou, 'GIoU': global_iou}


def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


def get_mask_from_json(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]

    height, width = img.size[1], img.size[0]

    ### sort polies by area
    area_list = []
    valid_poly_list = []
    for i in inform:
        label_id = i["label"]
        points = i["points"]
        if "flag" == label_id.lower():  ## meaningless deprecated annotations
            continue

        tmp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(tmp_mask, np.array([points], dtype=np.int32), True, 1, 1)
        cv2.fillPoly(tmp_mask, np.array([points], dtype=np.int32), 1)
        tmp_area = tmp_mask.sum()

        area_list.append(tmp_area)
        valid_poly_list.append(i)

    ### ground-truth mask
    sort_index = np.argsort(area_list)[::-1].astype(np.int32)
    sort_index = list(sort_index)
    sort_inform = []
    for s_idx in sort_index:
        sort_inform.append(valid_poly_list[s_idx])

    mask = np.zeros((height, width), dtype=np.uint8)
    for i in sort_inform:
        label_id = i["label"]
        points = i["points"]

        if "ignore" in label_id.lower():
            label_value = 0  # ignored during evaluation
        else:
            label_value = 1  # target

        cv2.polylines(mask, np.array([points], dtype=np.int32), True, label_value, 1)
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), label_value)

    return mask, comments, is_sentence


