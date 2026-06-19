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
# from projects.qwenvl_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
# from projects.qwenvl_sam2.hf.models.modeling_sa2va_chat_qwen2_5vl import Sa2VAChatQwen2_5VLModel
from projects.qwenvl_sam2.evaluation.dataset import TVGDataset
from projects.qwenvl_sam2.evaluation.utils import _init_dist_pytorch, _init_dist_slurm, get_dist_info, get_rank, collect_results_cpu

import concurrent.futures
from pycocotools import mask as cocomask


def async_func(executor, func, **kwargs):
    future = executor.submit(func, **kwargs)
    return future


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(cocomask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def collate_single(x):
    return x[0]


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
    'CHARADES': {
        'data_root': 'data/VTG/NumPro_FT',
        'image_folder': 'data/VTG/NumPro_FT/videos_1FPS',
        'expression_file': 'data/VTG/NumPro_FT/charades_test.json'
    },
    'ActivityNet': {
        'data_root': 'data/VTG/NumPro_FT',
        'image_folder': 'data/VTG/NumPro_FT/videos_1FPS',
        'expression_file': 'data/VTG/NumPro_FT/activitynet_val_2_test.json'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='RefVOS')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_INFO.keys(),
        default='CHARADES',
        help='Specify a dataset')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--work_dir', type=str, default=None)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--frame_num', type=int, default=5) # dummy
    parser.add_argument('--video_max_frames', type=int, default=50) # dummy
    parser.add_argument('--visualize', type=bool, default=False) # dummy
    parser.add_argument('--inference_mode', type=str, choices=['video', 'multi-frame', 'combine'], default="video")
    parser.add_argument('--threshold', type=float, default=0.3)
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

    model.video_min_pixels = 4*4*28*28
    model.video_max_pixels = 8*8*28*28

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    dataset_info = DATASETS_INFO[args.dataset]


    dataset = TVGDataset(
        image_folder=dataset_info['image_folder'],
        expression_file=dataset_info['expression_file'],
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
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_single,
    )
    results = []
    results_find = []
    executor = concurrent.futures.ThreadPoolExecutor()
    for item in tqdm.tqdm(dataloader):
        with torch.no_grad():
            result = model.predict_keyframe(
                video=item['images'],
                text=None,
                find_text=item['find_prompt'],
                tokenizer=tokenizer,
                num_frames=args.frame_num,
                inference_mode=args.inference_mode,
                video_max_frames=args.video_max_frames,
                query_text=item['exp'],
                threshold=args.threshold,
            )

        text_idx = 0
        video_id = item['video_id']
        text_prediction = result['prediction']
        temporal_ground_dict = result['temporal_grounding_dict']

        result = {
            'id': item['video_id'],
            'gt_start': item['start_time'],
            'gt_end': item['end_time'],
            'pred_start': temporal_ground_dict["pred_start"],
            'pred_end': temporal_ground_dict["pred_end"],
            'duration': item['duration'],
        }

        if args.visualize:
            generate_visualization(images = item['images'], query_text = item['exp'], video_name = item["video_id"], timestamp = [result["pred_start"],result["pred_end"]], save_dir=os.path.join(work_dir,"{args.dataset}/visualize"))

        result_find = {
            'id': item['video_id'],
            'gt_start': item['start_time'],
            'gt_end': item['end_time'],
            'pred_start': temporal_ground_dict["pred_start_find"],
            'pred_end': temporal_ground_dict["pred_end_find"],
            'duration': item['duration'],
        }
        results.append(result)
        results_find.append(result_find)

    executor.shutdown(wait=True)
    print(f'[Rank {rank}] : Finished.')
    
    os.makedirs(work_dir, exist_ok=True)
    json.dump(results, open(f'{work_dir}/{args.dataset}.json', 'w'), indent=4)
    json.dump(results_find, open(f'{work_dir}/{args.dataset}_find.json', 'w'), indent=4)

    if rank == 0:
        print('Done')


from PIL import Image, ImageDraw, ImageFont
def generate_visualization(images, query_text, video_name, timestamp, save_dir):
    os.makedirs(os.path.join(save_dir, video_name), exist_ok=True)
    start, end = timestamp
    processed_frames = []

    font = ImageFont.truetype("arial.ttf", 30)

    for i, image in enumerate(images):
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # 添加红色边框
        if start <= i <= end:
            border_width = 5
            draw.rectangle(
                [0, 0, img_draw.width - 1, img_draw.height - 1],
                outline="red",
                width=border_width
            )

        # 添加 query_text（蓝色文字）
        draw.text((50, 20), query_text, fill="blue", font=font)

        processed_frames.append(img_draw)

    # 保存为视频
    save_path = os.path.join(save_dir, video_name, query_text)
    save_video_from_pil_frames(processed_frames, save_path, fps=1)

import cv2

def save_video_from_pil_frames(pil_frames, save_path, fps=1):
    width, height = pil_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for pil_img in pil_frames:
        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"✅ 视频已保存到: {save_path}")

