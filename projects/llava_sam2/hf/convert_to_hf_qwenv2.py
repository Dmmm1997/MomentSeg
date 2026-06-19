import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
from transformers import Qwen2_5_VLConfig

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth-model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default='./work_dirs/hf_model', help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model._merge_lora()
    model.mllm.transfer_to_hf = True

    all_state_dict = model.all_state_dict()

    name_map = {'mllm.': '', '.gamma': '.g_weight'}

    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    # build the hf format model
    from projects.llava_sam2.hf.models.configuration_sa2va_chat_qwen2_5vl import Sa2VAChatQwen2_5VLConfig
    from projects.llava_sam2.hf.models.modeling_sa2va_chat_qwen2_5vl_new import Sa2VAChatQwen2_5VLModel
    config_dict = {}
    config_dict["min_pixels"] = cfg.min_pixels
    config_dict["max_pixels"] = cfg.max_pixels
    config_dict["mllm_path"] = cfg.path
    config_dict["template"] = cfg.template
    config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat_qwen2_5vl.Sa2VAChatQwen2_5VLConfig',
         'AutoModel': 'modeling_sa2va_chat_qwen2_5vl_new.Sa2VAChatQwen2_5VLModel',
         'AutoModelForCausalLM': 'modeling_sa2va_chat_qwen2_5vl_new.Sa2VAChatQwen2_5VLModel'}
    
    sa2va_hf_config = Sa2VAChatQwen2_5VLConfig(
        **config_dict
    )
    hf_sa2va_model = Sa2VAChatQwen2_5VLModel(sa2va_hf_config)
    hf_sa2va_model.load_state_dict(all_state_dict_new, strict=True)
    if "Qwen2.5-VL-3B-Instruct" in cfg.path:
        print("path:{}-----lm_head_copy".format(cfg.path))
        hf_sa2va_model.model.lm_head.weight = torch.nn.Parameter(hf_sa2va_model.model.model.embed_tokens.weight.clone())
    
    hf_sa2va_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)

    print(f"Save the hf model into {args.save_path}")

    # copy the files
    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {args.save_path}")

if __name__ == '__main__':
    main()