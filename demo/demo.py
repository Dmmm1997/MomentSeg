import argparse
import os

from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.generation import GenerationMixin
import transformers.modeling_utils as modeling_utils

import cv2
import decord
import torch
import numpy as np
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def _patch_transformers_compat(model=None):
    if not hasattr(modeling_utils, "GenerationMixin"):
        modeling_utils.GenerationMixin = GenerationMixin

    if model is None:
        return

    base_model = getattr(model, "model", None)
    if base_model is None or hasattr(base_model, "embed_tokens"):
        return

    language_model = getattr(base_model, "language_model", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if embed_tokens is None and hasattr(model, "get_input_embeddings"):
        embed_tokens = model.get_input_embeddings()
    if embed_tokens is not None:
        base_model.embed_tokens = embed_tokens

    if not hasattr(model, "rope_deltas"):
        model.rope_deltas = None


_patch_transformers_compat()


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('source', help='Path to image file or video')
    parser.add_argument('--model_path', default="checkpoints/MomentSeg-3B")
    parser.add_argument('--work-dir', default="output/demo", help='Directory used to save results.')
    parser.add_argument('--text', type=str, default="Please segment the person standing in the center wearing blue clothes.")
    parser.add_argument('--num-frames', type=int, default=8)
    parser.add_argument('--video-max-frames', type=int, default=100)
    parser.add_argument('--inference-mode', type=str, default="multi-frame", choices=["video", "multi-frame", "combine"])
    args = parser.parse_args()
    return args


def visualize_image(pred_mask, image, output_path):
    if Visualizer is None:
        return

    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = np.array(image)
    
    visualizer = Visualizer()
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='r', alphas=0.4)
    visual_result = visualizer.get_image()
    
    cv2.imwrite(output_path, visual_result)


def save_video_from_frames(frames, output_path, fps=30):
    if not frames:
        return

    if isinstance(frames[0], Image.Image):
        frame_array = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2BGR)
    else:
        frame_array = np.array(frames[0])

    height, width = frame_array.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        if isinstance(frame, Image.Image):
            frame_array = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            frame_array = np.array(frame)
        video_writer.write(frame_array)
    
    video_writer.release()


def save_masked_video_from_frames(original_frames, pred_masks, output_path, fps=30):
    if not original_frames or pred_masks is None or len(pred_masks) == 0:
        print("Warning: Empty frames or masks, skipping video saving")
        return
    
    if Visualizer is None:
        print("Warning: mmengine not available, skipping masked video saving")
        return

    if isinstance(original_frames[0], Image.Image):
        frame_array = cv2.cvtColor(np.array(original_frames[0]), cv2.COLOR_RGB2BGR)
    else:
        frame_array = np.array(original_frames[0])

    height, width = frame_array.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = min(len(original_frames), len(pred_masks))
    for i in range(num_frames):
        frame = original_frames[i]
        pred_mask = pred_masks[i]

        if isinstance(frame, Image.Image):
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            img = np.array(frame)

        try:
            visualizer = Visualizer()
            visualizer.set_image(img)
            visualizer.draw_binary_masks(pred_mask, colors='r', alphas=0.4)
            visual_result = visualizer.get_image()
            video_writer.write(visual_result)
        except Exception as e:
            print(f"Warning: Failed to visualize frame {i}: {e}")
            video_writer.write(img)

    video_writer.release()
    print(f"Saved masked video with {num_frames} frames to {output_path}")


def _read_video_decord(video_path):
    vr = decord.VideoReader(video_path)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    idx = torch.arange(0, total_frames).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in video]
    return pil_frames, video_fps


if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    _patch_transformers_compat(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    video_extensions = {".mp4", ".avi", ".mkv"}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    video_flag = 0

    vid_frames = []
    fps = 10

    if os.path.isfile(cfg.source) and os.path.splitext(cfg.source)[1].lower() in video_extensions:
        video_flag = 1
        vid_frames, fps = _read_video_decord(cfg.source)
        print(f"Loaded video with {len(vid_frames)} frames at {fps} fps")
    elif os.path.isfile(cfg.source) and os.path.splitext(cfg.source)[1].lower() in image_extensions:
        vid_frames = [Image.open(cfg.source).convert('RGB')]
        print(f"Loaded single image")
    else:
        if not os.path.isdir(cfg.source):
            raise FileNotFoundError(f"Input source does not exist: {cfg.source}")
        image_files = []
        video_flag = 1
        for filename in sorted(list(os.listdir(cfg.source))):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(filename)

        for filename in sorted(image_files):
            img_path = os.path.join(cfg.source, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                vid_frames.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
        print(f"Loaded {len(vid_frames)} images from folder")

    if not vid_frames:
        raise ValueError(f"No valid frames were loaded from {cfg.source}")

    if video_flag == 0:
        img_frame = vid_frames[0]
        print(f"The input is:\n{cfg.text}")
        print(f'Total frames: {len(vid_frames)}')
        text = "<image>\n"+cfg.text
        with torch.no_grad():
            result = model.predict_forward(
                image=img_frame,
                text=text,
                tokenizer=tokenizer,
            )
    else:
        print(f"The input is:\n{cfg.text}")
        print(f'Total frames: {len(vid_frames)}')
        find_prompt= "<video>\nHere is a low-resolution video you can refer to. Can you find the key frames range of the text query '{}'".format(cfg.text)
        text = "<image>\n"+cfg.text
        model.video_min_pixels = 4*4*28*28
        model.video_max_pixels = 6*6*28*28
        with torch.no_grad():
            result = model.predict_forward_find_seg(
                video=vid_frames,
                text=text,
                find_text=find_prompt,
                tokenizer=tokenizer,
                num_frames=cfg.num_frames,
                inference_mode=cfg.inference_mode,
                video_max_frames=cfg.video_max_frames,
            )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    os.makedirs(cfg.work_dir, exist_ok=True)

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        print(f"Got prediction masks with shape: {pred_masks.shape if hasattr(pred_masks, 'shape') else 'unknown'}")
        
        if video_flag == 0:
            output_path = os.path.join(cfg.work_dir, f"result_{os.path.basename(cfg.source)}")
            visualize_image(pred_masks[0], vid_frames[0], output_path)
            print(f"Result saved to {output_path}")
        else:
            source_name = os.path.splitext(os.path.basename(cfg.source))[0] if os.path.isfile(cfg.source) else "folder_video"
            output_video_path = os.path.join(cfg.work_dir, f"segmented_{source_name}.mp4")

            print(f"Original frames: {len(vid_frames)}")
            print(f"Prediction masks shape: {pred_masks.shape if hasattr(pred_masks, 'shape') else 'no shape'}")

            save_masked_video_from_frames(vid_frames, pred_masks, output_video_path, fps)
            print(f"Segmented video saved to {output_video_path}")

            original_video_path = os.path.join(cfg.work_dir, f"original_{source_name}.mp4")
            save_video_from_frames(vid_frames, original_video_path, fps)
            print(f"Original video saved to {original_video_path}")
    else:
        os.makedirs(cfg.work_dir, exist_ok=True)
        if video_flag == 0:
            output_path = os.path.join(cfg.work_dir, f"original_{os.path.basename(cfg.source)}")
            if isinstance(vid_frames[0], Image.Image):
                img_array = cv2.cvtColor(np.array(vid_frames[0]), cv2.COLOR_RGB2BGR)
            else:
                img_array = np.array(vid_frames[0])
            cv2.imwrite(output_path, img_array)
            print(f"Original image saved to {output_path}")
        else:
            source_name = os.path.splitext(os.path.basename(cfg.source))[0] if os.path.isfile(cfg.source) else "folder_video"
            output_video_path = os.path.join(cfg.work_dir, f"processed_{source_name}.mp4")
            save_video_from_frames(vid_frames, output_video_path, fps)
            print(f"Video saved to {output_video_path}")
