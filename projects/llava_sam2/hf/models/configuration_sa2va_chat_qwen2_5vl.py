# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from .configuration_internlm2 import InternLM2Config
from .configuration_phi3 import Phi3Config
from transformers import AutoConfig, LlamaConfig, Qwen2Config
from transformers import Qwen2_5_VLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class Sa2VAChatQwen2_5VLConfig(PretrainedConfig):
    model_type = 'sa2va_chat_qwen2_5vl'
    is_composition = True

    def __init__(
            self,
            mllm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            min_pixels=16*16*28*28,
            max_pixels=16*16*28*28,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            mllm_path=None,
            **kwargs):
        super().__init__(**kwargs)
        if mllm_config is not None:
            self.mllm_config = Qwen2_5_VLConfig(**mllm_config)
        else:
            mllm_config = {}
            logger.info('mllm_config is None. Initializing the Qwen2_5ConfigQwen2_5_VLConfig with default values.')
            self.mllm_config = mllm_config

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.mllm_path = mllm_path

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.tie_word_embeddings = False

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if self.mllm_config:
            output['mllm_config'] = self.mllm_config.to_dict()
        else:
            output['mllm_config'] = self.mllm_config
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        output['min_pixels'] = self.min_pixels
        output['max_pixels'] = self.max_pixels

        return output
