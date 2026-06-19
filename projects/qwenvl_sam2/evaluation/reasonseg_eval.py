import argparse
import copy
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
import random

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)


# from projects.qwenvl_sam2.hf.models.modeling_sa2va_chat_qwen2_5vl import Sa2VAChatQwen2_5VLModel

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import ReasonSegDataset

def collate_single(x):
    return x[0]

DATASETS_INFO = {
    'REASONSEG_VAL': {
        'data_root': 'data/reason_seg/val',
    },
    'REASONSEG_TEST': {
        'data_root': 'data/reason_seg/test',
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='REASONSEG_VAL',
        help='Specify a ref dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--min_pixels', type=int, default=4*4*28*28)
    parser.add_argument('--max_pixels', type=int, default=24*24*28*28)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    # setting the minmax pixels
    model.min_pixels = args.min_pixels
    model.max_pixels = args.max_pixels

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]

    dataset = ReasonSegDataset(
        image_folder=dataset_info["data_root"],
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
    for data_batch in tqdm.tqdm(dataloader):
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        del data_batch['img_id'], data_batch['gt_masks']

        texts = data_batch['text']
        del data_batch['text']
        pred_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred_mask = model.predict_forward(**_data_batch, tokenizer=tokenizer)['prediction_masks']
            if len(pred_mask) == 0:
                # give a zero mask
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0]
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_reasonseg_' + args.dataset + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    results_dir = os.path.join(args.model_path, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "{}.txt".format(args.dataset))
    if get_rank() == 0:
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)
        with open(results_path, "w") as F:
            F.write(str(metric))

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()
