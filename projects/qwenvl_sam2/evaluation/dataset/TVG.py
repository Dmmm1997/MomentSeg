import os
import json

import mmengine

from PIL import Image
import copy

from mmengine.dist import master_only
from collections import defaultdict

from .base_eval_dataset import BaseEvalDataset
import torch
import decord

# FIND_PROMPT = "<video>\nHere is a low-resolution video you can refer to. Can you find the key frames range of the text query '{}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range."
FIND_PROMPT = "<video>\nThis is a low-resolution video. Can you find the key frames range of the text query '{}' in this video?"

class TVGDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
    ):
        super().__init__()
        vid2metaid = self.json_file_preprocess(expression_file)
        self.vid2metaid = vid2metaid

        self.image_folder = image_folder

    def __len__(self):
        return len(self.vid2metaid)

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)
        return expression_datas

    def _read_video_decord(self, video_path):
        vr = decord.VideoReader(video_path)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        idx = torch.arange(0, total_frames).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in video]
        return pil_frames, video_fps

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.vid2metaid[index])
        frames, frame_fps = self._read_video_decord(os.path.join(self.image_folder, video_obj_info["video"]))
        data_dict = {}
        exp = video_obj_info['query']
        video_id = video_obj_info['id']
        ori_width, ori_height = frames[0].size

        data_dict['type'] = 'video'
        data_dict['index'] = index
        data_dict['video_id'] = video_id
        data_dict['images'] = frames
        data_dict['start_time'] = video_obj_info['start_time']
        data_dict['end_time'] = video_obj_info['end_time']

        data_dict['duration'] = video_obj_info['duration']

        # data_dict['text_prompt'] = SEG_PROMPT.format(exp) if '?' not in exp else exp # FIXME: ? is not a good sign
        data_dict['find_prompt'] = FIND_PROMPT.format(exp)
        data_dict['image_folder'] = self.image_folder
        data_dict['exp'] = exp

        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width

        return data_dict
