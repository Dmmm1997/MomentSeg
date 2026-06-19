# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import Any, List, Optional, Tuple, Union

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.utils.checkpoint
import transformers

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, AutoProcessor)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
from transformers.utils import ModelOutput, logging
from transformers import StoppingCriteriaList, StoppingCriteria

from .configuration_momentseg_chat import MomentSegConfig
import re
import os
from .cross_attn import SEG_ATTN


from transformers import (
    Qwen2_5_VLForConditionalGeneration,
)

from .sam2 import SAM2
# from .sam2_1 import SAM2_1
from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import resize, to_pil_image

import torch.nn.functional as F
import torch
from scipy.ndimage import gaussian_filter1d

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))

class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        cur_text = cur_text.replace('\r', '').replace('\n', '')
        return cur_text[-self.length:] == self.stop_word

def get_stop_criteria(
    tokenizer,
    stop_words=[],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria

class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class QwenPrepocessor():
    def __init__(self, model_path, min_pixels=12*12*28*28, max_pixels=24*24*28*28) -> None:
        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, images=None, videos=None, text=None, **kwargs):
        return self.processor(images=images, videos=videos, text=text, min_pixels=self.min_pixels, max_pixels=self.max_pixels, fps=1, **kwargs)



class MomentsegModel(Qwen2_5_VLForConditionalGeneration):
    config_class = MomentSegConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        pass

    def __init__(self, config: MomentSegConfig, use_flash_attn=True, convert_to_hf=False):
        attn_implementation = "flash_attention_2" if has_flash_attn and use_flash_attn else "eager"
        config.mllm_config._attn_implementation = attn_implementation
        super().__init__(config.mllm_config)
        self.model_path = config.mllm_path
        self.perframe_seg = config.perframe_seg

        if convert_to_hf:
            self.config = config

        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        self.video_min_pixels = config.video_min_pixels
        self.video_max_pixels = config.video_max_pixels

        self.template = "qwen_chat"
        self.template = self.template.replace('-', '_')
        self.downsample_ratio = 0.5

        self.conv_template = PROMPT_TEMPLATE[self.template]
        self.template = self.conv_template

        self.grounding_encoder = SAM2()
        # self.grounding_encoder = SAM2_1()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = config.mllm_config.hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.find_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.img_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        if self.perframe_seg:
            self.seg_attn = SEG_ATTN(256)
        self.init_prediction_config = False
    
    def preparing_for_generation(self, tokenizer, max_new_tokens=2048, torch_dtype=torch.bfloat16):
        # set stop criteria and generation configs for model
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = tokenizer
        self.bot_name = 'BOT'
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )

        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True
        self.torch_dtype = torch_dtype
        self.to(torch_dtype)
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.use_thumbnail = True
        self.image_size = 448

        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<|image_pad|>'
        self.VID_CONTEXT_TOKEN = '<|video_pad|>'
        self.IMG_START_TOKEN = '<|vision_start|>'
        self.IMG_END_TOKEN = '<|vision_end|>'
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'
        self._system = 'You are a helpful assistant.'

        self.patch_token = 1
        self.transformer = None
        self.image_preprocessor = QwenPrepocessor(self.model_path, self.min_pixels, self.max_pixels)
        self.video_preprocessor = QwenPrepocessor(self.model_path, self.video_min_pixels, self.video_max_pixels)
        # self.video_preprocessor.min_pixels = 400 * 28 * 28
        # self.video_preprocessor.max_pixels = 400 * 28 * 28

        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        self.img_token_idx = tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.find_token_idx = tokenizer.convert_tokens_to_ids('[FIND]')
        self.video_token_idx = tokenizer.convert_tokens_to_ids(self.VID_CONTEXT_TOKEN)
        return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def find_start_end_sft(self, predict_find, video_len):
        start, end = extract_kf_indices(predict_find)
        if start is None or start>=end or start>=1 or end<=0:
            print("not matching.-----{}-----".format(predict_find))
            start, end = 0.0, 1.0
        start = np.clip(round(start * video_len), 0, video_len - 1)
        end = np.clip(round(end * video_len), 0, video_len - 1)
        return start, end

    def find_start_end_FIND(self, find_tokens, video_tokens, video_len, video_grid_thw, threshold=0.3):
        if len(find_tokens)>0:
            if len(find_tokens)>1:
                find_tokens = find_tokens[0:1]
            video_tokens = video_tokens.reshape(video_grid_thw[0][0], -1, video_tokens.shape[-1])
            video_tokens_pooled = video_tokens.mean(1)
            video_tokens_pooled = video_tokens_pooled.unsqueeze(0).transpose(1,2) # 1, C, t
            video_tokens_pooled = F.interpolate(video_tokens_pooled, size=video_len, mode='linear', align_corners=True).squeeze().transpose(0,1) # T,C
            similarity = F.cosine_similarity(find_tokens.unsqueeze(1), video_tokens_pooled.unsqueeze(0), dim=-1).reshape(-1)
            find_logits = similarity / 0.07
            find_prob = find_logits.sigmoid().float().detach().cpu().numpy()
            find_prob_smooth = gaussian_filter1d(find_prob, sigma=1.0)
            find_start, find_end = find_longest_left_right_index(find_prob_smooth, threshold=threshold)
            max_center_index = find_max_window_center(find_prob_smooth, 5)
        else:
            find_start = 0
            find_end = video_len-1
            max_center_index = video_len//2
            find_prob_smooth = None
        return find_start, find_end, max_center_index, find_prob_smooth 

    def predict_keyframe(
            self,
            image=None,
            video=None,
            text=None,
            find_text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            num_frames=5,
            video_max_frames=100,
            inference_mode="combine", # (multi-frame or video or combine)
            dynamic_image_size=False,
            query_text="",
            threshold=0.3,
        ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        input_dict = {}
        assert video is not None
        pixel_values = []
        video_frame_values = []
        extra_pixel_values = []
        ori_image_size = video[0].size

        len_frames = len(video)

        if len(video) > video_max_frames + 1:
            # video_index = np.random.choice(len_frames, video_max_frames, replace=False)
            video_index = np.linspace(0, len_frames - 1, video_max_frames, dtype=int)
        else:
            video_index = np.arange(0, len_frames)
        video_index.sort()

        for frame_idx, frame_image in enumerate(video):
            assert ori_image_size == frame_image.size
            g_image = np.array(frame_image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            extra_pixel_values.append(g_image)
            if frame_idx in video_index:
                video_frame_values.append(frame_image)

        video_data_dict = self.video_preprocessor(videos=video_frame_values, text="")
        video_data_dict['pixel_values_videos'] = torch.tensor(video_data_dict['pixel_values_videos'], dtype=torch.float).cuda()
        video_data_dict['video_grid_thw'] = torch.tensor(video_data_dict['video_grid_thw'], dtype=torch.int).cuda()
        num_video_tokens = int(video_data_dict['video_grid_thw'][0].prod() * (self.downsample_ratio ** 2))

        input_dict.update(video_data_dict)

        video_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.VID_CONTEXT_TOKEN * num_video_tokens}' \
                        f'{self.IMG_END_TOKEN}'

        ret_masks = []
        find_text = find_text.replace('<video>', video_token_str)
        input_text = ''
        input_text += self.template['SYSTEM'].format(system=self._system)
        input_text += self.template['INSTRUCTION'].format(
            input=find_text, round=1, bot_name=self.bot_name)
        input_text = past_text + input_text
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool).cuda()
        
        pixel_values = input_dict.get("pixel_values", None)
        pixel_values_videos = input_dict.get("pixel_values_videos", None)
        image_grid_thw = input_dict.get("image_grid_thw", None)
        video_grid_thw = input_dict.get("video_grid_thw", None)

        mm_inputs_find = {
            'pixel_values':  pixel_values,
            'pixel_values_videos': pixel_values_videos,
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'image_grid_thw': image_grid_thw,
            'video_grid_thw': video_grid_thw
        }

        generate_output_find= self.generate(
            **mm_inputs_find,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            max_new_tokens=100
        )

        predict_find = self.tokenizer.decode(
            generate_output_find.sequences[0][mm_inputs_find['input_ids'].shape[1]:], skip_special_tokens=False).strip()
        # find the key frames and 
        start, end = self.find_start_end_sft(predict_find, len(video_index))
        
        # if have <FIND>, find the <FIND> hidden states
        hidden_states = generate_output_find.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        find_hidden_states = self.find_hidden_fcs(last_hidden_states)
        video_tokens = get_seg_hidden_states(
            find_hidden_states, generate_output_find.sequences[0][:-1],
            seg_id=self.video_token_idx
        )
        find_tokens = get_seg_hidden_states(
            find_hidden_states, generate_output_find.sequences[0][:-1],
            seg_id=self.find_token_idx
        )
        find_start, find_end, _, _ = self.find_start_end_FIND(find_tokens, video_tokens, len(video_index), video_grid_thw, threshold=threshold)

        temporal_grounding_dict = {}
        temporal_grounding_dict['pred_start'] = int(video_index[start])
        temporal_grounding_dict['pred_end'] = int(video_index[end])
        temporal_grounding_dict['pred_start_find'] = int(video_index[find_start])
        temporal_grounding_dict['pred_end_find'] = int(video_index[find_end])
        return {'prediction': predict_find, 'prediction_masks': ret_masks, 'temporal_grounding_dict': temporal_grounding_dict}
    
    def predict_forward_find_seg(
            self,
            image=None,
            video=None,
            text=None,
            find_text=None,
            past_text='',
            tokenizer=None,
            num_frames=8,
            video_max_frames=100,
            inference_mode="multi-frame", # (multi-frame or video or combine)
        ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        input_dict = {}
        assert video is not None
        pixel_values = []
        video_frame_values = []
        extra_pixel_values = []
        ori_image_size = video[0].size

        len_frames = len(video)

        if len(video) > video_max_frames + 1:
            # video_index = np.random.choice(len_frames, video_max_frames, replace=False)
            video_index = np.linspace(0, len_frames - 1, video_max_frames, dtype=int)
        else:
            video_index = np.arange(0, len_frames)
        video_index.sort()

        for frame_idx, frame_image in enumerate(video):
            assert ori_image_size == frame_image.size
            g_image = np.array(frame_image)  # for grounding
            g_image = self.extra_image_processor.apply_image(g_image)
            g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
            extra_pixel_values.append(g_image)
            if frame_idx in video_index:
                video_frame_values.append(frame_image)

        video_data_dict = self.video_preprocessor(videos=video_frame_values, text="")
        video_data_dict['pixel_values_videos'] = torch.tensor(video_data_dict['pixel_values_videos'], dtype=torch.float).cuda()
        video_data_dict['video_grid_thw'] = torch.tensor(video_data_dict['video_grid_thw'], dtype=torch.int).cuda()
        num_video_tokens = int(video_data_dict['video_grid_thw'][0].prod() * (self.downsample_ratio ** 2))

        input_dict.update(video_data_dict)

        video_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.VID_CONTEXT_TOKEN * num_video_tokens}' \
                        f'{self.IMG_END_TOKEN}'

        ret_masks = []
        find_text = find_text.replace('<video>', video_token_str)
        input_text = ''
        input_text += self.template['SYSTEM'].format(system=self._system)
        input_text += self.template['INSTRUCTION'].format(
            input=find_text, round=1, bot_name=self.bot_name)
        input_text = past_text + input_text
        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool).cuda()
        
        pixel_values_videos = input_dict.get("pixel_values_videos", None)
        video_grid_thw = input_dict.get("video_grid_thw", None)

        mm_inputs_find = {
            'pixel_values':  None,
            'pixel_values_videos': pixel_values_videos,
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'image_grid_thw': None,
            'video_grid_thw': video_grid_thw
        }

        generate_output_find= self.generate(
            **mm_inputs_find,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            max_new_tokens=128
        )

        predict_find = self.tokenizer.decode(
            generate_output_find.sequences[0][mm_inputs_find['input_ids'].shape[1]:], skip_special_tokens=False).strip()
        
        find_prob_logits = None
        center_index = None
        sample_center = None
        if len_frames<num_frames:
            frame_index = list(range(0, len_frames))
        else:
            hidden_states = generate_output_find.hidden_states
            last_hidden_states = [item[-1][0] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            find_hidden_states = self.find_hidden_fcs(last_hidden_states)
            video_tokens = get_seg_hidden_states(
                find_hidden_states, generate_output_find.sequences[0][:-1],
                seg_id=self.video_token_idx
            )
            find_tokens = get_seg_hidden_states(
                find_hidden_states, generate_output_find.sequences[0][:-1],
                seg_id=self.find_token_idx
            )
            _, _, sample_center, find_prob_logits = self.find_start_end_FIND(find_tokens, video_tokens, len_frames, video_grid_thw, 0.4) #  len(video_index)

            if find_prob_logits is None:
                frame_index = np.linspace(0, len_frames - 1, num_frames, dtype=int)
            else:
                frame_index, center_index = proportional_weighted_sampling(find_prob_logits, sample_center, num_frames-1)
                if len(frame_index)!=len(set(frame_index)):
                    frame_index = np.linspace(0, len_frames - 1, num_frames, dtype=int)
                    center_index = None


        pixel_values = [video[idx] for idx in frame_index]

        input_dict = {}
        image_data_dict = self.image_preprocessor(images=pixel_values, text="")
        image_data_dict['pixel_values'] = torch.tensor(image_data_dict['pixel_values'], dtype=torch.float).cuda()
        image_data_dict['image_grid_thw'] = torch.tensor(image_data_dict['image_grid_thw'], dtype=torch.int).cuda()
        num_image_tokens = int(image_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
        num_frames = image_data_dict['image_grid_thw'].shape[0]

        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
        ]).to(self.torch_dtype)

        input_dict['g_pixel_values'] = g_pixel_values

        image_token_str = f'{self.IMG_START_TOKEN}' \
                        f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{self.IMG_END_TOKEN}'
        image_token_str = image_token_str * num_frames

        if inference_mode=="multi-frame":
            input_dict.update(image_data_dict)
            text = text.replace('<image>', image_token_str)
        elif inference_mode=="video":
            input_dict.update(video_data_dict)
            text = text.replace('<video>', video_token_str)
        elif inference_mode=="combine":
            input_dict.update(image_data_dict)
            input_dict.update(video_data_dict)
            text = text.replace('<image>', image_token_str)
            text = text.replace('<video>', video_token_str)
        else:
            print("image_mode")

        input_text = ''
        input_text += self.template['SYSTEM'].format(system=self._system)
        input_text += self.template['INSTRUCTION'].format(
            input=text, round=1, bot_name=self.bot_name)

        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool).cuda()
        
        pixel_values = input_dict.get("pixel_values", None)
        pixel_values_videos = input_dict.get("pixel_values_videos", None)
        image_grid_thw = input_dict.get("image_grid_thw", None)
        video_grid_thw = input_dict.get("video_grid_thw", None)

        mm_inputs = {
            'pixel_values':  pixel_values,
            'pixel_values_videos': pixel_values_videos,
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'image_grid_thw': image_grid_thw,
            'video_grid_thw': video_grid_thw
        }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
            max_new_tokens=128
        )

        predict = self.tokenizer.decode(
            generate_output.sequences[0][mm_inputs['input_ids'].shape[1]:], skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        img_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.img_token_idx
        )
        all_img_hidden_states = self.img_hidden_fcs(img_hidden_states)
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            nf = 1 if video is None else min(num_frames, len_frames)
            if self.perframe_seg:
                h,w = image_grid_thw[0][1:]
                all_img_hidden_states = all_img_hidden_states.reshape(nf, -1, 256).transpose(1,2).reshape(nf,256,h//2,w//2)
                seg_hidden_states_list = [item.unsqueeze(0) for item in self.seg_attn(seg_hidden_states.repeat(nf, 1), all_img_hidden_states)]
            else:
                seg_hidden_states_list = [seg_hidden_states] * nf

            pred_masks = self.grounding_encoder.language_embd_inference_withbap(sam_states, seg_hidden_states_list, frame_index, center_index)

            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 
                'prediction_masks': ret_masks,
                'find_logits':find_prob_logits, 
                'sample_center': sample_center,
                'sample_index': frame_index}

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            num_frames=5,
            video_max_frames=50,
            inference_mode="combine", # (multi-frame or video or combine)
            dynamic_image_size=False,
            **kwargs,
        ):
        if not self.init_prediction_config:
            assert tokenizer
            self.preparing_for_generation(tokenizer=tokenizer)

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'prompt_masks': None,
                'vp_overall_mask': None,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < num_frames:
                        pixel_values.append(frame_image)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                if inference_mode=="multi-frame":
                    _data_dict = self.image_preprocessor(images=pixel_values, text="")
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float).cuda()
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int).cuda()
                    num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_frames = _data_dict['image_grid_thw'].shape[0]
                    num_video_tokens = num_image_tokens * num_frames
                    input_dict.update(_data_dict)
                    input_dict['vp_overall_mask'] = None
                elif inference_mode=="video":
                    _data_dict = self.video_preprocessor(videos=pixel_values, text="")
                    _data_dict['pixel_values_videos'] = torch.tensor(_data_dict['pixel_values_videos'], dtype=torch.float).cuda()
                    _data_dict['video_grid_thw'] = torch.tensor(_data_dict['video_grid_thw'], dtype=torch.int).cuda()
                    num_video_tokens = int(_data_dict['video_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    input_dict.update(_data_dict)
                    input_dict['vp_overall_mask'] = None
                else:
                    raise NotImplementedError
            else: # @TODO: support images
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                if dynamic_image_size:
                    images = dynamic_preprocess(image, self.min_dynamic_patch,
                                                self.max_dynamic_patch,
                                                self.image_size, self.use_thumbnail)
                else:
                    images = [image]

                if mask_prompts is not None:
                    vp_overall_mask = torch.Tensor([False] * (len(images) - 1) + [True])
                    input_dict['vp_overall_mask'] = vp_overall_mask
                else:
                    input_dict['vp_overall_mask'] = None
                
                _data_dict = self.image_preprocessor(images=images, text="")
                _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float).cuda()
                _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int).cuda()
                num_image_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                num_frames = _data_dict['image_grid_thw'].shape[0]
                input_dict.update(_data_dict)

            input_dict['g_pixel_values'] = g_pixel_values

            if mask_prompts is not None:
                # reshape mask prompts to feature size
                mask_prompts = [torch.Tensor(item).to(pixel_values.device) for item in mask_prompts]
                mask_prompts = [F.interpolate(
                    item.unsqueeze(0),
                    size=(int(self.image_size // self.patch_size * self.downsample_ratio),
                          int(self.image_size // self.patch_size * self.downsample_ratio)),
                    mode='nearest').squeeze(0) for item in mask_prompts]
                region_pixels = []
                for mask_prompt in mask_prompts[0]:
                    region_pixels.append(mask_prompt.bool().to(torch.int64).sum())

                vp_token_str = '\nThere are {} part regions in the picture: '.format(len(mask_prompts[0]))
                for i in range(len(mask_prompts[0])):
                    vp_token_str = vp_token_str + \
                                   f"region{i + 1}" + self.VP_START_TOKEN + \
                                   self.IMG_CONTEXT_TOKEN * region_pixels[i] + \
                                   self.VP_END_TOKEN
                    if i == len(mask_prompts[0]) - 1:
                        vp_token_str = vp_token_str + '.\n'
                    else:
                        vp_token_str = vp_token_str + ', '
            else:
                vp_token_str = ''

            if inference_mode != "video" or video is None:
                image_token_str = f'{self.IMG_START_TOKEN}' \
                                f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                                f'{self.IMG_END_TOKEN}'
                image_token_str = image_token_str * num_frames
            else:
                image_token_str = f'{self.IMG_START_TOKEN}' \
                                f'{self.VID_CONTEXT_TOKEN * num_video_tokens}' \
                                f'{self.IMG_END_TOKEN}'

            ret_masks = []

            if '<image>' in text or mask_prompts is not None:
                assert past_text is None or len(past_text) == 0
            text = text.replace('<image>\n', image_token_str + vp_token_str)
            input_text = ''
            input_text += self.template['SYSTEM'].format(system=self._system)
            input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).cuda().unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool).cuda()
            
            pixel_values = input_dict.get("pixel_values", None)
            pixel_values_videos = input_dict.get("pixel_values_videos", None)
            image_grid_thw = input_dict.get("image_grid_thw", None)
            video_grid_thw = input_dict.get("video_grid_thw", None)

            mm_inputs = {
                'pixel_values':  pixel_values,
                'pixel_values_videos': pixel_values_videos,
                'input_ids': ids,
                'attention_mask': attention_mask,
                'position_ids': None,
                'past_key_values': None,
                'labels': None,
                'image_grid_thw': image_grid_thw,
                'video_grid_thw': video_grid_thw
            }

        generate_output = self.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            use_cache=True,
        )

        predict = self.tokenizer.decode(
            generate_output.sequences[0][mm_inputs['input_ids'].shape[1]:], skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            nf = 1 if video is None else min(num_frames, len(g_pixel_values))
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * nf)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}    
    

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def extract_kf_indices(text):
    # pattern = r"<kf>\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?\s*</kf>"
    pattern = r"\s*\(?\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)?\s*"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        try:
            # 直接转成 float
            start = float(match.group(1))
            end = float(match.group(2))
            return start, end
        except ValueError:
            return None, None  # 如果提取到的不是合法数字，返回 None
    else:
        return None, None

def find_longest_left_right_index(arr, threshold=0.3):
    mask = np.array(arr) > threshold
    max_len = start = end = tmp_start = 0
    for i, val in enumerate(mask):
        if val:
            if i == 0 or not mask[i-1]:
                tmp_start = i
            if i - tmp_start + 1 > max_len:
                max_len = i - tmp_start + 1
                start, end = tmp_start, i
    return (start, end) if max_len > 0 else (0, len(mask)-1)

def find_max_window_center(arr, window_size):
    arr = np.array(arr)
    if len(arr) < window_size:
        return len(arr)//2

    # 计算滑动窗口的和
    window_sums = np.convolve(arr, np.ones(window_size), mode='valid')
    max_index = np.argmax(window_sums)

    # 计算中心索引
    center_index = max_index + window_size // 2
    return center_index

import numpy as np

def proportional_weighted_sampling(arr, center, total_samples=8):
    """
    基于中心点的左侧和右侧权重和分配样本数量，并在左、右分别进行CDF采样。

    arr: 一维数组，表示权重（非负）
    center: 中心点位置
    T: 数组的总长度
    total_samples: 总采样数量（包括中心点）
    """

    # 计算左侧和右侧的权重和
    left_sum = arr[:center].sum()
    right_sum = arr[center + 1:].sum()

    # 按照比例分配样本数量
    total_sum = left_sum + right_sum
    left_count = int(round(total_samples * (left_sum / total_sum)))
    right_count = total_samples - left_count

    # 进行逆CDF采样
    def cdf_sampling(arr, count):
        """根据arr的权重进行逆CDF采样"""
        arr = np.array(arr, dtype=float)
        total = arr.sum()
        if total == 0:
            # 全零时均匀采样
            return np.linspace(0, len(arr) - 1, count, dtype=int).tolist()

        pdf = arr / total
        cdf = np.cumsum(pdf)

        # 生成均匀分布的采样点
        sample_points = (np.arange(count) + 0.5) / count

        indices = []
        for sp in sample_points:
            # 找到CDF中第一个大于sp的位置（逆CDF采样）
            idx = np.searchsorted(cdf, sp)
            indices.append(idx)

        return sorted(indices)

    # 左侧采样
    left_indices = cdf_sampling(arr[:center], left_count)

    # 右侧采样
    right_indices = cdf_sampling(arr[center + 1:], right_count)

    # 左侧和右侧的索引需要调整为原数组的索引
    left_indices = [i for i in left_indices]  # 左侧索引已经是原始的
    right_indices = [center + 1 + i for i in right_indices]  # 右侧索引加上center+1偏移量

    # 合并左右采样结果
    indices = left_indices + [center] + right_indices

    return sorted(indices), len(left_indices)

