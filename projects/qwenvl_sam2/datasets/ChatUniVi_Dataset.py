import logging
import os
from typing import Literal

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy
from .encode_fn import video_lisa_encode_fn
import json
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu


def _get_rawvideo_dec(video_path, select_frames=5):

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    elif os.path.exists(video_path.replace('mkv', 'mp4')):
        vreader = VideoReader(video_path.replace('mkv', 'mp4'), ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0
    f_end = len(vreader) - 1
    num_frames = f_end - f_start + 1
    assert num_frames > 0, f'num_frames: {num_frames}, f_start: {f_start}, f_end: {f_end}, fps: {fps}, video_path: {video_path}'
    # T x 3 x H x W
    if num_frames <= select_frames:
        sample_pos = range(f_start, f_end + 1)
    else:
        split_point = np.linspace(0, num_frames, num=select_frames+1, dtype=int)
        sample_pos = [np.random.randint(split_point[i], split_point[i+1]) for i in range(select_frames)]
    patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
    return patch_images


class VideoChatUniViDataset(Dataset):
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
                 json_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 sampled_frames=10,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=4096,
                 lazy=True,
                 repeats=1,
                 special_tokens=None,
                 use_fast=False,
                 n_fast_images=50,
                 fast_pool_size=4,
                 arch_type: Literal['intern_vl', 'qwen', 'qwen_video'] = 'qwen_video',
                 preprocessor=None,
    ):
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        self.sampled_frames = sampled_frames
        assert offline_processed_text_folder or (json_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if offline_processed_text_folder and json_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            json_datas = self.json_file_preprocess(json_file)
            self.json_datas = json_datas
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            if self.lazy:
                self.text_data = build_origin_dataset(json_data, 'train')
            else:
                raise NotImplementedError

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.arch_type = arch_type
        self._system = 'You are a helpful assistant.'
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        if self.arch_type == 'qwen_video':
            self.IMG_CONTEXT_TOKEN = '<|video_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
        self.repeats = repeats

        self.downsample_ratio = 0.5

        self.patch_token = 1

        self.preprocessor = BUILDER.build(preprocessor)

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.use_fast = use_fast
        self.n_fast_images = n_fast_images
        self.fast_pool_size = fast_pool_size

        # for visualization debug
        self.save_folder = './work_dirs/video_debug/'
        self.cur_number = 0

        print("Video Chat dataset, include {} items.".format(len(self.text_data)))

    def __len__(self):
        return self.real_len() * self.repeats

    @property
    def modality_length(self):
        return [5000 for _ in range(len(self.text_data)*self.repeats)]

    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, json_file):
        # prepare expression annotation files
        with open(json_file, 'r') as f:
            json_datas = json.load(f)
        return json_datas

    def dataset_map_fn(self, data_dict, select_k=5):
        assert 'video' in data_dict
        # video
        video_file = data_dict['video']
        video_file = os.path.join(self.image_folder, video_file)
        images = _get_rawvideo_dec(video_file, select_frames=select_k)
        if self.use_fast:
            fast_images = _get_rawvideo_dec(video_file, select_frames=self.n_fast_images)
        else:
            fast_images = None

        conversation = data_dict['conversations']

        # prepare text
        if self.use_fast:
            text_dict = self.prepare_text(
                select_k, conversation, num_image_tokens=self.patch_token,
                n_fast_images=len(fast_images),
            )
        else:
            text_dict = self.prepare_text(
                select_k, conversation, num_image_tokens=self.patch_token,
            )


        ret = {'images': images, 'conversation': text_dict['conversation'], 'fast_images': fast_images}
        return ret

    def prepare_text(self, n_frames, conversation, num_image_tokens=256, n_fast_images=0):

        if self.use_fast:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                          f'{self.FAST_IMG_CONTEXT_TOKEN * n_fast_images * self.fast_pool_size * self.fast_pool_size}' \
                          f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        questions = []
        answers = []

        for conv in conversation:
            if conv['from'] == 'human':
                questions.append(conv['value'].replace('<image>', ''))
            else:
                answers.append(conv['value'])
        assert len(questions) == len(answers)

        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str 
                if "qwen" not in self.arch_type:
                    frame_tokens += '\n'
                # frame_tokens = '=' + ' '
                if self.arch_type=="qwen_video": # for qwen2.5, the token num of a video is 1
                    frame_tokens = frame_tokens * 1
                else:
                    frame_tokens = frame_tokens * n_frames
                frame_tokens = frame_tokens.strip()
                frame_tokens = fast_frame_token_str + frame_tokens
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

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

    def __getitem__(self, index):
        index = index % self.real_len()
        selected_data_dict = copy.deepcopy(self.text_data[index])
        data_dict = self.dataset_map_fn(selected_data_dict, select_k=self.sampled_frames)


        assert 'images' in data_dict.keys()
        if self.use_fast:
            assert 'fast_images' in data_dict.keys()
        pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        if data_dict.get('images', None) is not None:
            frames_files = data_dict['images']
            for frame_image in frames_files:
                frame_image = frame_image.convert('RGB')
                pixel_values.append(frame_image)

            if self.arch_type == 'qwen':
                _data_dict = self.preprocessor(images=pixel_values, text="")
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                num_frames = _data_dict['image_grid_thw'].shape[0]
                num_video_tokens = num_frame_tokens * num_frames
            elif self.arch_type == 'qwen_video':
                _data_dict = self.preprocessor(videos=[pixel_values], text="")
                _data_dict['pixel_values_videos'] = torch.tensor(_data_dict['pixel_values_videos'], dtype=torch.float)
                _data_dict['video_grid_thw'] = torch.tensor(_data_dict['video_grid_thw'], dtype=torch.int)
                num_video_tokens = int(_data_dict['video_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
            else:
                raise NotImplementedError
            data_dict.update(_data_dict)
        else:
            print(f'ChatUniVi: Skip this sample due to no images existing')
            return self.__getitem__(random.randint(0, self.real_len()))

        if self.arch_type == 'qwen':
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str
        elif self.arch_type == 'qwen_video':
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_video_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length, with_image_token=True)
        if len(result["input_ids"]) >= self.max_length:
            print(f'ChatUniVi: Skip this sample due to its long context')
            return self.__getitem__(random.randint(0, self.real_len()))
        data_dict.update(result)
        data_dict['type'] = 'video'
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

        return
