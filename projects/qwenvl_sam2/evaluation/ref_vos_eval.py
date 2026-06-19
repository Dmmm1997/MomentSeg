import argparse
import json
import os

import mmengine
import numpy as np
from PIL import Image

import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoTokenizer
from projects.qwenvl_sam2.hf.models.modeling_momentseg import MomentsegModel
from projects.qwenvl_sam2.evaluation.dataset import RefVOSDataset
from projects.qwenvl_sam2.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future

def collate_single(x):
    return x[0]


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def mask_save(item, mask_prediction, work_dir):
    vid_id = item['video_id']
    exp_id = item['exp_id']
    save_path = os.path.join(work_dir, 'Annotations', vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        mask = Image.fromarray(mask.astype(np.float32) * 255).convert('L')
        file_name = item['frames'][id_m]
        save_file = os.path.join(save_path, file_name + ".png")
        mask.save(save_file)


DATASETS_INFO = {
    'DAVIS': {
        'data_root': 'data/video_datas/davis17/',
        'image_folder': 'data/video_datas/davis17/valid/JPEGImages/',
        'expression_file': 'data/video_datas/davis17/meta_expressions/valid/meta_expressions.json',
        'mask_file': 'data/video_datas/davis17/valid/mask_dict.pkl',
    },
    'MEVIS': {
        'data_root': 'data/video_datas/mevis/valid/',
        'image_folder': 'data/video_datas/mevis/valid/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid/meta_expressions.json',
        'mask_file': None,
    },
    'MEVIS_U': {
        'data_root': 'data/video_datas/mevis/valid_u/',
        'image_folder': 'data/video_datas/mevis/valid_u/JPEGImages',
        'expression_file': 'data/video_datas/mevis/valid_u/meta_expressions.json',
        'mask_file': 'data/video_datas/mevis/valid_u/mask_dict.json',
    },
    'REFYTVOS': {
        'data_root': 'data/video_datas/rvos/',
        'image_folder': 'data/video_datas/rvos/valid/JPEGImages/',
        'expression_file': 'data/video_datas/rvos/meta_expressions/valid/meta_expressions.json',
        'mask_file': None,
    },
    'REVOS': {
        'data_root': 'data/video_datas/revos/',
        'image_folder': 'data/video_datas/revos/',
        'expression_file': 'data/video_datas/revos/meta_expressions_valid_.json',
        'mask_file': None,
    },
    'REASONVOS':{
        'data_root': 'data/video_datas/reasonvos/',
        'image_folder': 'data/video_datas/reasonvos/JPEGImages/',
        'expression_file': 'data/video_datas/reasonvos/meta_expressions_format.json',
        'mask_file': None,
    },
    # 'RefSAV':{
    #     'data_root': 'data/ref_sav/valid/',
    #     'image_folder': 'data/ref_sav/valid/videos',
    #     'expression_file': 'data/ref_sav/valid/meta_expressions_valid.json',
    #     'mask_file': 'data/ref_sav/valid/mask_dict.json',
    # }
    'REF_SAV': {
        'data_root': 'data/video_datas/ref_sav_eval',
        'image_folder': 'data/video_datas/ref_sav_eval/videos',
        'expression_file': 'data/video_datas/ref_sav_eval/meta_expressions_valid.json',
        'mask_file': 'data/video_datas/ref_sav_eval/mask_dict.json',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='MEVIS',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--frame_num', type=int, default=5) # dummy
    parser.add_argument('--video_max_frames', type=int, default=100) # dummy
    parser.add_argument('--inference_mode', type=str, choices=['video', 'multi-frame', 'combine'], default="video")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    args = parse_args()

    work_dir = args.work_dir
    if work_dir is None:
        work_dir = 'work_dirs/foobar'

    if args.launcher == 'none':
        rank = 0
        world_size = 1
    elif args.launcher == 'pytorch':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
    elif args.launcher == 'slurm':
        _init_dist_slurm('nccl')
        rank, world_size = get_dist_info()

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]


    dataset = RefVOSDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
        mask_file=dataset_info['mask_file'],
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_single,
    )
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    
    forward_func = model.predict_forward_find_seg
    model.video_min_pixels = 4*4*28*28
    model.video_max_pixels = 6*6*28*28
        
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            result = forward_func(
                video=item['images'],
                text=item['text_prompt'],
                find_text=item['find_prompt'],
                tokenizer=tokenizer,
                num_frames=args.frame_num,
                inference_mode=args.inference_mode,
                video_max_frames=args.video_max_frames,
            )

        text_idx = 0
        text_prediction = result['prediction']
        find_logits = result.get("find_logits", None)
        sample_center = result.get("sample_center", None)
        sample_index = result.get("sample_index", None)
        if len(result['prediction_masks']) > 0:
            mask_prediction = result['prediction_masks'][text_idx]
        else:
            print(text_prediction)
            mask_prediction = np.zeros((item['length'], item['ori_height'], item['ori_width']), dtype=np.uint8)

        if args.submit:
            async_func(executor, mask_save, item=item, mask_prediction=mask_prediction, work_dir=os.path.join(work_dir, args.dataset))
            encoded_mask = None
        else:
            encoded_mask = mask_to_rle(mask_prediction)

        result = {
            'index': item['index'],
            'video_id': item['video_id'],
            'exp_id': item['exp_id'],
            'text_prediction': text_prediction,
            'frames': item['frames'],
            'exp': item['text_prompt'],
            'prediction_masks': encoded_mask,
            'find_logits': find_logits.tolist() if find_logits is not None else None,
            'sample_center': int(sample_center) if sample_center is not None else None,
            'sample_index': [int(ind) for ind in sample_index] if sample_index is not None else None,
        }
        results.append(result)


    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    if not args.submit:
        results = collect_results_cpu(results, len(dataset))
        if get_rank() == 0:
            final_results = {}
            for item in results:
                vid_id = item['video_id']
                exp_id = item['exp_id']
                if vid_id not in final_results:
                    final_results[vid_id] = {}
                assert exp_id not in final_results[vid_id]
                final_results[vid_id][exp_id] = item
            os.makedirs(work_dir, exist_ok=True)
            json.dump(final_results, open(f'{work_dir}/{args.dataset}.json', 'w'))

    if rank == 0:
        print('Done')
