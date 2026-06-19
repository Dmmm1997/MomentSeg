import json
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import copy
import random
from typing import Literal

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import process_hf_dataset, build_origin_dataset
import torchvision.transforms as T
from xtuner.utils import DEFAULT_IMAGE_TOKEN
from torchvision.transforms.functional import InterpolationMode
from .encode_fn import video_lisa_encode_fn
from .utils import dynamic_preprocess

from .gcg_process import glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn, glamm_refcocog_map_fn

class GCGDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 single_image_mode=False,
                 preprocessor=None,
    ):
        super().__init__()
        assert lazy
        self.lazy = lazy
        self.max_length = max_length

        json_data = self.json_file_preprocess(data_path)
        json_data = DatasetDict({'train': HFDataset.from_list(json_data)})
        self.text_data = build_origin_dataset(json_data, 'train')

        self.image_folder = image_folder

        self._system = ''

        self.arch_type = arch_type
        if self.arch_type == 'qwen':
            self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
            self.IMG_START_TOKEN = '<|vision_start|>'
            self.IMG_END_TOKEN = '<|vision_end|>'
            self._system = 'You are a helpful assistant.'

        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.template_map_fn = template_map_fn
        if isinstance(self.template_map_fn, dict) and self.lazy:
            _type = self.template_map_fn['type']
            del self.template_map_fn['type']
            self.template_map_fn = _type(**self.template_map_fn)

        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)

        assert preprocessor is not None
        self.preprocessor = BUILDER.build(preprocessor)

        self.repeats = repeats
        self.downsample_ratio = 0.5
        self.patch_token = 1

        self.single_image_mode = single_image_mode

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    @property
    def modality_length(self):
        return [4000] * self.real_len() * self.repeats

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_refcocog_map_fn(data_dict)
        return data_dict

    def replace_image_str(self, data_dict, image_str):
        data_dict['conversation'][0]['input'] = \
            data_dict['conversation'][0]['input'].replace(DEFAULT_IMAGE_TOKEN, image_str)
        return data_dict

    def __getitem__(self, index):

        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        # parse datasets
        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        # process image
        image_file = data_dict['image']
        image = Image.open(os.path.join(self.image_folder,
                                        image_file)).convert('RGB')
        ori_width, ori_height = image.size
        if hasattr(self, 'extra_image_processor'):
            g_image = np.array(image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            data_dict['g_pixel_values'] = g_pixel_values

        images = [image]

        assert self.preprocessor is not None
        _data_dict = self.preprocessor(images=images, text="")
        _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
        _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
        num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
        num_image = len(_data_dict['image_grid_thw'])
        data_dict.update(_data_dict)

        image_token_str = f'{self.IMG_START_TOKEN}' \
                            f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                            f'{self.IMG_END_TOKEN}'
        image_token_str = image_token_str * num_image

        data_dict = self.replace_image_str(data_dict, image_token_str)

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(data_dict, tokenizer=self.tokenizer, max_length=self.max_length,
                                      with_image_token=True)
        if len(result["input_ids"]) >= self.max_length:
            print(f'GCG: Skip this sample due to its long context')
            return self.__getitem__(random.randint(0, self.real_len()))
        data_dict.update(result)
        # process mask
        data_dict['masks'] = self.decode_mask(data_dict['masks'], ori_height=ori_height, ori_width=ori_width)

        if data_dict['masks'] is None:
            print(f'GCG: Skip this sample due to mask is none')
            return self.__getitem__(random.randint(0, self.real_len()))

        return data_dict

class RefCOCOgGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def json_file_preprocess(self, data_path):
        json_data = json.load(open(data_path))

        # convert {id: dict} to dict(..., id=xx)
        for idx in range(len(json_data)):
            id = list(json_data[idx].keys())[0]
            json_data[idx] = json_data[idx][id]
            json_data[idx].update({'id': id})
        return json_data

class GranDfGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_granf_map_fn(data_dict)
        return data_dict

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)

            for rle in object_mask:
                m = mask.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

class OpenPsgGCGDataset(GranDfGCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_openpsg_map_fn(data_dict)
        return data_dict


class FlickrGCGDataset(GCGDataset):
    def __init__(self,
                 image_folder,
                 data_path=None,
                 tokenizer=None,
                 max_length=8196,
                 special_tokens=None,
                 template_map_fn=None,
                 extra_image_processor=None,
                 lazy=True,
                 repeats=1,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'qwen',
                 preprocessor=None,
                 ):
        super().__init__(
            image_folder=image_folder,
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            special_tokens=special_tokens,
            template_map_fn=template_map_fn,
            extra_image_processor=extra_image_processor,
            lazy=lazy,
            repeats=repeats,
            single_image_mode=single_image_mode,
            arch_type=arch_type,
            preprocessor=preprocessor,
        )

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_flickr_map_fn(data_dict)
        return data_dict

    def json_file_preprocess(self, data_path):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]

        # convert {id: dict} to dict(..., id=xx)
        from pycocotools.coco import COCO
        self.coco = COCO(data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{data_path}'!"
        print(f'Removed {removed_img_count} images.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]

        # obtain_annotations
        for data_info in data_infos:
            ann_ids = self.coco.getAnnIds(imgIds=data_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            data_info.update({'ann_info': ann_info})
        return data_infos

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask.decode(object_mask).astype(np.uint8)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks