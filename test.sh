#!/bin/bash

MODEL_PATHS=(
  "checkpoints/MomentSeg-3B"
)

NUM_GPUS=${NUM_GPUS:-2}
MASTER_PORT=${MASTER_PORT:-29506}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

video_mode="multi-frame" # choices=['multi-frame', 'video', 'combine']
num_frames=(8)
video_max_frames=(100)
keyframe_threshold=0.4
torchrun_args=(--nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT")

echo ">>> Evaluation settings: NUM_GPUS=$NUM_GPUS, MASTER_PORT=$MASTER_PORT"

for model_path in "${MODEL_PATHS[@]}"; do

  model_name=$(basename "$model_path")
  work_dir=$model_path/evaluation

  for num_frame in "${num_frames[@]}"; do
    for video_max_frame in "${video_max_frames[@]}"; do

      echo ">>> Running model: $model_name"
      echo ">>> model_path: $model_path"
      echo ">>> num_frame: $num_frame, video_max_frame: $video_max_frame"

      # Ref-YTB
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset REFYTVOS \
        --submit \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \

      cd "$work_dir/REFYTVOS/"
      zip -qr RefYTB.zip Annotations/
      cd "$REPO_ROOT"

      # MEVIS
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset MEVIS \
        --submit \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \

      cd "$work_dir/MEVIS/"
      zip -qr MEVIS.zip Annotations/
      cd "$REPO_ROOT"


      # ReVOS
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset REVOS \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \

      python tools/eval/eval_revos.py "$work_dir/REVOS.json" --save_json_name "revos_valid.json"

      # MEVIS_U
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset MEVIS_U \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \

      python tools/eval/eval_mevis.py "$work_dir"/MEVIS_U.json --save_name "mevis_valu.json" # --generate_video

      # DAVIS
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset DAVIS \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame

      python tools/eval/eval_davis.py "$work_dir"/DAVIS.json --save_name "refer-davis17.json" #--generate_video

      # ReasonVOS
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/ref_vos_eval.py \
        "$model_path" \
        --dataset REASONVOS \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \

      python tools/eval/eval_reasonvos.py "$work_dir"/REASONVOS.json --save_name "reasonvos_val.json" #--generate_video


      # RefCOCO
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco \
        --split val \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco \
        --split testA \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco \
        --split testB \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco_plus \
        --split val \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco_plus \
        --split testA \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcoco_plus \
        --split testB \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcocog \
        --split val \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/refcoco_eval.py \
        "$model_path" \
        --dataset refcocog \
        --split test \
        --launcher pytorch


      # ReasonSeg
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/reasonseg_eval.py \
        "$model_path" \
        --dataset REASONSEG_VAL \
        --launcher pytorch

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/reasonseg_eval.py \
        "$model_path" \
        --dataset REASONSEG_TEST \
        --launcher pytorch
      
      # TVG
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/tvg_eval.py \
        "$model_path" \
        --dataset CHARADES \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \
        --threshold $keyframe_threshold \
        
      # echo "-----metrics for charades-STA sft-------"
      # python tools/eval/eval_tvg.py "$work_dir"/CHARADES.json --save_name "charades.json"
      echo "-----metrics for charades-STA <FIND>-------"
      python tools/eval/eval_tvg.py "$work_dir"/CHARADES_find.json --save_name "charades_find.json"

      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/tvg_eval.py \
        "$model_path" \
        --dataset ActivityNet \
        --launcher pytorch \
        --work_dir "$work_dir" \
        --frame_num $num_frame \
        --inference_mode $video_mode \
        --video_max_frames $video_max_frame \
        --threshold $keyframe_threshold \
        
      # echo "-----metrics for activitynet-caption sft-------"
      # python tools/eval/eval_tvg.py "$work_dir"/ActivityNet.json --save_name "activitynet.json"
      echo "-----metrics for activitynet-caption <FIND>-------"
      python tools/eval/eval_tvg.py "$work_dir"/ActivityNet_find.json --save_name "activitynet_find.json"

      # GCG
      torchrun \
        "${torchrun_args[@]}" \
        projects/qwenvl_sam2/evaluation/gcg_eval.py \
        "$model_path" \
        --split val \
        --save_dir "$work_dir/GCG" \
        --launcher pytorch

      python projects/qwenvl_sam2/evaluation/metrics_gcg.py --split val --prediction_dir_path "$work_dir/GCG" --work_dir "$work_dir"
      python projects/qwenvl_sam2/evaluation/metrics_gcg.py --split test --prediction_dir_path "$work_dir/GCG" --work_dir "$work_dir"
    done
  done
  echo ">>> Finished model: $model_name"
  echo "---------------------------------------------"
done
