import copy
import random
import glob
import json
import logging
import os
from typing import Literal

import torch

from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from pycocotools import mask as mask_utils

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
import cv2

from projects.glamm.datasets.utils.utils import SEG_QUESTIONS, ANSWER_LIST
from projects.glamm.utils import DEFAULT_IMAGE_TOKEN


SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this image? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this image? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

SENTENCE_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this image? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this image? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

# SENTENCE_QUESTIONS = [
#     "{class_name} Can you segment the target in this image?",
#     "{class_name} Please segment it in this image.",
#     "{class_name} Please respond with segmentation mask.",
#     "{class_name} Please output segmentation mask.",

#     "{class_name} Can you segment the target in this image",
#     "{class_name} Please segment it in this image",
#     "{class_name} Please respond with segmentation mask",
#     "{class_name} Please output segmentation mask",

#     "{class_name} Could you provide a segmentation mask for the target in this image?",
#     "{class_name} Please identify and segment it in this image.",
#     "{class_name} Where is the target in this image? Please respond with a segmentation mask.",
#     "{class_name} Can you highlight the target in this image with a segmentation mask?",

#     "{class_name} Could you provide a segmentation mask for the target in this image",
#     "{class_name} Please identify and segment it in this image",
#     "{class_name} Where is the target in this image? Please respond with a segmentation mask",
#     "{class_name} Can you highlight the target in this image with a segmentation mask",
# ]

class ReasonSegDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self,
                 data_root,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 tokenizer=None,
                 max_length=2048,
                 repeat=4,
                 num_classes_per_sample=3,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 **kwargs):
        self.items = self.json_file_preprocess(data_root)

        self.begin_str = f'{DEFAULT_IMAGE_TOKEN}\n'
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        self.arch_type = arch_type
        self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
        self.IMG_START_TOKEN = '<|vision_start|>'
        self.IMG_END_TOKEN = '<|vision_end|>'

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.image_folder = data_root
        self.template = prompt_template
        self.max_length = max_length
        self._system = 'You are a helpful assistant.'

        self.num_classes_per_sample = num_classes_per_sample
        self.downsample_ratio = 0.5
        self.repeat = repeat

        self.preprocessor = BUILDER.build(preprocessor)
        self.arch_type = arch_type
        self.single_image_mode = single_image_mode
        self._max_refetch = 1000

        print("ReasonSEG dataset, include {} items.".format(self.real_len()))

    def __len__(self) -> int:
        return self.real_len() * self.repeat
    
    def real_len(self) -> int:
        return len(self.items)

    @property
    def modality_length(self):
        return [300 for _ in range(self.real_len()*self.repeat)]

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

    def get_mask_from_json(self, json_path, img):
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

    def _parse_annotations(self, ann_info, is_sentence):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        masks, phrases = [], []
        mask, text = ann_info['mask'], ann_info['text']
        index = np.random.choice(range(len(text)), self.num_classes_per_sample, replace=True)
        for idx in index:
            phrase = text[idx].lower()
            binary_mask = mask.astype(np.uint8)
            masks.append(binary_mask)
            phrases.append(phrase)

        conversation = []
        for i, phrase in enumerate(phrases):
            if is_sentence:
                question = random.choice(SENTENCE_QUESTIONS).format(class_name=phrase)
            else:
                question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

    def prepare_data(self, index):
        item = copy.deepcopy(self.items[index%self.real_len()])
        image = item['image']
        json_file = item['json_file']
        frame_image = Image.open(image).convert('RGB')
        gt_masks, exps, is_sentence = self.get_mask_from_json(json_file, frame_image)
        data_dict = {}
        data_dict["img_path"] = image
        data_dict["mask"] = gt_masks
        data_dict["text"] = exps
        data_dict = self._parse_annotations(data_dict, is_sentence)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image = frame_image
            if hasattr(self, 'extra_image_processor'):
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                out_data_dict['g_pixel_values'] = g_pixel_values

            images = [image]
            assert self.preprocessor is not None
            _data_dict = self.preprocessor(images=images, text="")
            _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
            _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
            num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
            num_image = len(_data_dict['image_grid_thw'])
            out_data_dict.update(_data_dict)
            image_token_str = f'{self.IMG_START_TOKEN}' \
                              f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                              f'{self.IMG_END_TOKEN}'
            image_token_str = image_token_str * num_image
            token_dict = self.get_inputid_labels(data_dict['conversations'], image_token_str)
            out_data_dict.update(token_dict)
        else:
            print(f'ReasonSEG: Skip this sample due to no images existing')
            return self.__getitem__(random.randint(0, len(self)))

        if len(token_dict["input_ids"]) >= self.max_length:
            print(f'ReasonSEG: Skip this sample due to its long context')
            return self.__getitem__(random.randint(0, len(self)))

        return out_data_dict

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        # print('len_ids: ', len(input_ids))
        return {'input_ids': input_ids, 'labels': labels}

    def __getitem__(self, index):
        index = index % self.real_len()
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data


if __name__ == '__main__':
    from transformers import CLIPImageProcessor, AutoTokenizer
    from third_parts.segment_anything.utils.transforms import ResizeLongestSide

    pretrained_model = 'MBZUAI/GLaMM-GranD-Pretrained'
    llm_name_or_path = 'lmsys/vicuna-7b-v1.5'

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path)
    image_processor = dict(
        type=CLIPImageProcessor.from_pretrained,
        pretrained_model_name_or_path='openai/clip-vit-large-patch14-336')
    extra_image_processor = dict(
        type=ResizeLongestSide,
        target_length=1024,
    )
    from xtuner.utils.templates import PROMPT_TEMPLATE

    prompt_template = PROMPT_TEMPLATE.vicuna
    from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, template_map_fn
    from projects.glamm.datasets.collate_fns.glamm_collate_fn import glamm_collate_fn

    dataset = ReferSegmDataset(
        tokenizer=tokenizer,
        special_tokens=['[SEG]'],
        extra_image_processor=extra_image_processor,
        prompt_template=prompt_template,
        data_root='data/coco/',
        data_prefix=dict(img_path='train2014/'),
        ann_file='refcoco+/instances.json',
        split_file='refcoco+/refs(unc).p',
    )
    for i in range(1000):
        dataset[i]



