import logging
import os
from typing import Literal

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy

from projects.qwenvl_sam2.datasets.encode_fn import video_lisa_encode_fn
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import decord
from collections import defaultdict

VIDEO_PROMPT = [
    "<video>\nThis is a low-resolution video.",
    "<video>\nYou can view this low-resolution video for reference.",
    "<video>\nThis is a low-resolution video for analysis.",
    "<video>\nHere is a low-resolution video you can refer to.",
    # "<video>\nTake a look at this low-resolution video clip.",
    # "<video>\nPlease review this low-resolution video.",
    # "<video>\nAnalyze this low-resolution video.",
    # "<video>\nCheck out this low-resolution video for details.",
    # "<video>\nThis is a low-resolution reference video.",
    # "<video>\nRefer to this low-resolution visual input.",
    # "<video>\nI have uploaded a low-resolution video for your review.",
    # "<video>\nLet us examine this low-resolution video.",
    # "<video>\nHere is a short low-resolution video for analysis.",
    # "<video>\nThis low-resolution video may provide useful context.",
]

# VIDEO_PROMPT = [
#     "<video>"
# ]

KEY_FRAME_QUESTIONS = [
    "Can you find the key frames range of the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Could you identify the key frames range for the text query '{class_name}' in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Please locate the key frames range where the text query '{class_name}' appears in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Can you determine the key frames range containing the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Find the key frames range that captures the text query '{class_name}' in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Where is the key frames range for the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Please provide the key frames range showing the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Can you extract the key frames range depicting the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Identify the key frames range where the text query '{class_name}' is visible in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Could you output the key frames range for the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Locate the key frames range that highlights the text query '{class_name}' in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Please find the key frames range with the text query '{class_name}' in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Determine the key frames range where the text query '{class_name}' is most prominent in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Extract the key frames range that best represents the text query '{class_name}' in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Can you provide the key frames range of the text query '{class_name}' occurrences in this video? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Please analyze the video and output the key frames range for the text query '{class_name}'? output with format of (start_ratio, end_ratio) as a 0-1 range.",
    "Find the key frames range that showcases the text query '{class_name}' in this low-resolution video? output with format of (start_ratio, end_ratio) as a 0-1 range."
]

FIND_QUESTIONS = [
    "Can you find the key frames range of the text query '{class_name}' in this video?",
    "Could you identify the key frames range for the text query '{class_name}' in this low-resolution video?",
    "Please locate the key frames range where the text query '{class_name}' appears in this video?",
    "Can you determine the key frames range containing the text query '{class_name}' in this video?",

    "Can you find the key frames range of the text query '{class_name}' in this video",
    "Could you identify the key frames range for the text query '{class_name}' in this low-resolution video",
    "Please locate the key frames range where the text query '{class_name}' appears in this video",
    "Can you determine the key frames range containing the text query '{class_name}' in this video",
    # "Find the key frames range that captures the text query '{class_name}' in this low-resolution video?",
    # "Where is the key frames range for the text query '{class_name}' in this video?",
    # "Please provide the key frames range showing the text query '{class_name}' in this video?",
    # "Can you extract the key frames range depicting the text query '{class_name}' in this video?",
    # "Identify the key frames range where the text query '{class_name}' is visible in this low-resolution video?",
    # "Could you output the key frames range for the text query '{class_name}' in this video?",
    # "Locate the key frames range that highlights the text query '{class_name}' in this video?",
    # "Please find the key frames range with the text query '{class_name}' in this low-resolution video?",
    # "Determine the key frames range where the text query '{class_name}' is most prominent in this video?",
    # "Extract the key frames range that best represents the text query '{class_name}' in this low-resolution video?",
    # "Can you provide the key frames range of the text query '{class_name}' occurrences in this video?",
    # "Please analyze the video and output the key frames range for the text query '{class_name}'?",
    # "Find the key frames range that showcases the text query '{class_name}' in this low-resolution video?"
]


ANSWER_LIST = [
    "Sure, ({},{}).",
    "It is ({},{}).",
    "({},{})."
]


KEY_FRAME_ANSWER_LIST = [
    "It is [FIND]. The range is ({},{})",
    "Sure, [FIND]. The range is ({},{})",
    "Sure, it is [FIND]. The range is ({},{})",
    "[FIND]. The range is ({},{})",
]

FIND_ANSWER_LIST = [
    "It is [FIND].",
    "Sure, [FIND].",
    "Sure, it is [FIND].",
    "[FIND].",
]

class CharadesDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    def __init__(self,
                 image_folder,
                 expression_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 select_number=5,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=4096,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 frame_contiguous_sample=False,
                 use_fast=False,
                 arch_type: Literal['qwen', 'qwen_video'] = 'qwen',
                 preprocessor=None,
                 # only work if use_fast = True
                 n_fast_images=50,
                 fast_pool_size=4,
                 fast_token_after_question=False,
                 video_max_frames=100,
                 find_key_frames=False,
                 find_sft=False,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        self.arch_type = arch_type

        self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
        self.VID_CONTEXT_TOKEN = '<|video_pad|>'
        self.IMG_START_TOKEN = '<|vision_start|>'
        self.IMG_END_TOKEN = '<|vision_end|>'
        self._system = 'You are a helpful assistant.'

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid = self.json_file_preprocess(expression_file)
            self.vid2metaid = vid2metaid
            self.videos = list(self.vid2metaid.keys())

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = repeats

        self.downsample_ratio = 0.5

        self.patch_token = 1

        self.preprocessor = BUILDER.build(preprocessor)
        self.video_max_frames = video_max_frames

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        self.frame_contiguous_sample = frame_contiguous_sample

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        self.find_key_frames = find_key_frames
        self.find_sft = find_sft

        # exist_thr
        self.exist_thr = 8
        self.fast_token_after_question = fast_token_after_question
        if self.fast_token_after_question:
            assert self.use_fast

        print("Video temperal grounding dataset, include {} items.".format(len(self.vid2metaid)))

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        return [60000 for _ in range(len(self.vid2metaid)*self.repeats)]
        # length_list = []
        # for data_dict in self.vid2metaid:
        #     cur_len = 300
        #     length_list.append(cur_len)
        # return length_list

    def real_len(self):
        return len(self.vid2metaid)

    def json_file_preprocess(self, expression_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)["annotations"]

        vid2metaid = defaultdict(list)
        for sample_info in expression_datas:
            video_name = sample_info.pop("image_id")
            vid2metaid[video_name].append(sample_info)

        return vid2metaid


    def dataset_map_fn(self, frames, data_dict, select_video_nums=100):
        len_frames = len(frames)
        if len_frames > select_video_nums:
            # selected_video_indexes = np.random.choice(len_frames, select_video_nums, replace=False)
            selected_video_indexes = np.linspace(0,len_frames-1, select_video_nums, dtype=int)
        else:
            selected_video_indexes = np.arange(len_frames)
        selected_video_indexes.sort()
        video_images = [frames[i] for i in selected_video_indexes]
        # prepare text
        expressions = [object_info['caption'] for object_info in data_dict]
        new_range_len_frames = selected_video_indexes[-1]-selected_video_indexes[0] # 选择的图片对应到原始视频中的长度
        time_stamp = [np.clip((np.array(object_info['timestamp'])-selected_video_indexes[0])/new_range_len_frames, 0, 1) for object_info in data_dict] # 新的相对的比例
        text_dict = self.prepare_text(time_stamp, expressions)

        if self.find_key_frames:
            video_key_frames = []
            for object_info in data_dict:
                time_start, time_end = object_info['timestamp']
                key_frames = []
                for ti in selected_video_indexes:
                    if time_start <= ti <= time_end:
                        key_frames.append(1)
                    else:
                        key_frames.append(0)
                video_key_frames.append(np.array(key_frames))
        else:
            video_key_frames = None
    
        ret = {"video_images": video_images, 'conversation': text_dict['conversation'], 'video_key_frames': video_key_frames}
        return ret

    def prepare_text(self, time_stamp, expressions):
        video_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.VID_CONTEXT_TOKEN * 1}' \
                          f'{self.IMG_END_TOKEN}'

        after_question_str = ''
        questions = []
        answers = []
        for i, (exp, ts) in enumerate(zip(expressions,time_stamp)):
            exp = exp.replace('.', '').strip()
            if self.find_sft and self.find_key_frames:
                template = random.choice(KEY_FRAME_QUESTIONS)
                answer_text = random.choice(KEY_FRAME_ANSWER_LIST).format(round(ts[0],3), round(ts[1],3))
            elif self.find_sft:
                template = random.choice(KEY_FRAME_QUESTIONS)
                answer_text = random.choice(ANSWER_LIST).format(round(ts[0],3), round(ts[1],3))
            elif self.find_key_frames:
                template = random.choice(FIND_QUESTIONS)
                answer_text = random.choice(FIND_ANSWER_LIST)
            else:
                raise NotImplementedError
            questions.append(template.format(class_name=exp.lower()))
            answers.append(answer_text)
        qa_list = []

        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                video_token = video_token_str
                video_prompt = random.choice(VIDEO_PROMPT).replace("<video>", video_token)
                question = video_prompt + question
            qa_list.append({'from': 'human', 'value': question + after_question_str})
            qa_list.append({'from': 'gpt', 'value': answer})


        input = ''
        conversation = []
        for msg in qa_list:
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = ''
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}

    def _read_video_decord(self, video_path):
        vr = decord.VideoReader(video_path)
        total_frames, video_fps = len(vr), vr.get_avg_fps()
        idx = torch.arange(0, total_frames).round().long().tolist()
        video = vr.get_batch(idx).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in video]
        return pil_frames, video_fps

    def __getitem__(self, index):
        index = index % self.real_len()
        frames, frame_fps = self._read_video_decord(os.path.join(self.image_folder, self.videos[index]))
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = copy.deepcopy(selected_video_objects)

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=False)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        data_dict = self.dataset_map_fn(frames, video_objects_infos, select_video_nums=self.video_max_frames)

        video_pixel_values = []
        num_video_tokens = None
        if data_dict.get('video_images', None) is not None:
            video_pixel_values = data_dict['video_images']
            assert self.preprocessor is not None
            video_data_dict = self.preprocessor(videos=[video_pixel_values], text="")
            video_data_dict['pixel_values_videos'] = torch.tensor(video_data_dict['pixel_values_videos'], dtype=torch.float)
            video_data_dict['video_grid_thw'] = torch.tensor(video_data_dict['video_grid_thw'], dtype=torch.int)
            num_video_tokens = int(video_data_dict['video_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
            data_dict.update(video_data_dict)
        else:
            print(f'Charades: Skip this sample due to no images existing')
            return self.__getitem__(random.randint(0, self.real_len()))

        for i, conversation in enumerate(data_dict['conversation']):
            input_str = conversation['input']
            if self.VID_CONTEXT_TOKEN in input_str:
                input_str = input_str.replace(self.VID_CONTEXT_TOKEN, self.VID_CONTEXT_TOKEN * num_video_tokens)
                assert input_str.count(self.VID_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][i]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        if len(result["input_ids"]) >= self.max_length:
            print(f'Charades: Skip this sample due to input length exceeding max_length')
            return self.__getitem__(random.randint(0, self.real_len()))
        data_dict.update(result)

        data_dict['type'] = 'video'
        # print("Charades")
        return data_dict

    def visualization_debug(self, data_dict):
        save_folder = os.path.join(self.save_folder, 'sample_{}'.format(self.cur_number))
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self.cur_number += 1

        # images

        show_images = []

        pixel_values = data_dict['pixel_values']
        save_folder_image = os.path.join(save_folder, 'image')
        if not os.path.exists(save_folder_image):
            os.mkdir(save_folder_image)
        for i_image, image_pixel_value in enumerate(pixel_values):
            # print(image_pixel_value.shape)
            image_pixel_value[0] = image_pixel_value[0] * 0.2686
            image_pixel_value[1] = image_pixel_value[1] * 0.2613
            image_pixel_value[2] = image_pixel_value[2] * 0.2757
            image_pixel_value[0] = image_pixel_value[0] + 0.4814
            image_pixel_value[1] = image_pixel_value[1] + 0.4578
            image_pixel_value[2] = image_pixel_value[2] + 0.4082
            image_pixel_value = image_pixel_value * 255
            image_pixel_value = image_pixel_value.permute(1, 2, 0)
            image_pixel_value = image_pixel_value.to(torch.uint8).numpy()
            # print(os.path.join(save_folder_image, '{}.jpg'.format(i_image)))
            # print(image_pixel_value.shape)
            show_images.append(image_pixel_value)
            cv2.imwrite(os.path.join(save_folder_image, '{}.jpg'.format(i_image)), image_pixel_value)

        # text
        input_text = self.tokenizer.decode(data_dict['input_ids'], skip_special_tokens=False)
        with open(os.path.join(save_folder, 'text.json'), 'w') as f:
            json.dump([input_text], f)

        # masks
        save_folder_mask = os.path.join(save_folder, 'mask')
        if not os.path.exists(save_folder_mask):
            os.mkdir(save_folder_mask)
        n_frames = len(pixel_values)
        masks = data_dict['masks']
        _, h, w = masks.shape
        masks = masks.reshape(-1, n_frames, h, w)
        for i_obj, obj_masks in enumerate(masks):
            save_folder_mask_obj_folder = os.path.join(save_folder_mask, 'obj_{}'.format(i_obj))
            if not os.path.exists(save_folder_mask_obj_folder):
                os.mkdir(save_folder_mask_obj_folder)
            for i_frame, f_mask in enumerate(obj_masks):
                f_mask = f_mask.numpy()
                f_mask = f_mask * 255
                f_mask = np.stack([f_mask * 1, f_mask * 0, f_mask * 0], axis=2)
                f_mask = show_images[i_frame] * 0.3 + 0.7 * f_mask
                f_mask = f_mask.astype(np.uint8)
                cv2.imwrite(os.path.join(save_folder_mask_obj_folder, '{}.png'.format(i_frame)), f_mask)
        return



if __name__ == "__main__":
    from xtuner.dataset.map_fns import template_map_fn_factory
    from xtuner.utils import PROMPT_TEMPLATE
    from projects.qwenvl_sam2.models.preprocess.image_resize import DirectResize
    from transformers import AutoTokenizer
    from projects.qwenvl_sam2.models.preprocess.qwen_preprocess import QwenPrepocessor
    import os.path as osp
    dataset = CharadesDataset(
        "data/video_datas/mevis/train/JPEGImages",
        "data/video_datas/mevis/train/meta_expressions.json",
        "data/video_datas/mevis/train/mask_dict.json",
        extra_image_processor=dict(
            type=DirectResize,
            target_length=1024),
        tokenizer=dict(
            type=AutoTokenizer.from_pretrained,
            pretrained_model_name_or_path="./pretrained/Qwen2.5-VL-3B-Instruct",
            trust_remote_code=True,
            padding_side='right'),
        select_number=5,
        sampled_frames=5,
        offline_processed_text_folder=None,
        template_map_fn=dict(
            type=template_map_fn_factory, template=PROMPT_TEMPLATE.qwen_chat),
        max_length=2048,
        lazy=True,
        repeats=1,
        special_tokens=['[SEG]', '<p>', '</p>', '<vp>', '</vp>'],
        frame_contiguous_sample=False,
        use_fast=False,
        arch_type="qwen",
        preprocessor=dict(
            type=QwenPrepocessor,
            model_path="./pretrained/Qwen2.5-VL-3B-Instruct",
        ),
        # only work if use_fast = True
        n_fast_images=50,
        fast_pool_size=4,
        fast_token_after_question=False,
    )


    for i in range(len(dataset)):
        dataset.__getitem__(i)
    
