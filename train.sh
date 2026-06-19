
#!/bin/bash
set -e

NUM_GPUS=${NUM_GPUS:-2}
PORT=${PORT:-29500}

export PORT

CONFIG_LIST=(
  "projects/qwenvl_sam2/configs/momentseg-3B.py"
  # Add more config paths here.
)

for config in "${CONFIG_LIST[@]}"; do
  config_name=$(basename "$config" .py)
  echo ">>> Running config: $config_name"

  echo ">>> GPUs: $NUM_GPUS, PORT: $PORT"

  bash tools/dist.sh train "$config" "$NUM_GPUS"

  pth_path=$(cat work_dirs/"$config_name"/last_checkpoint)

  python projects/qwenvl_sam2/hf/convert_to_hf_qwenv2.py \
    "$config" \
    --pth-model "$pth_path" \
    --save-path work_dirs/"$config_name"/hf_model

  model_path=work_dirs/"$config_name"/hf_model
  work_dir=$model_path/evaluation

  echo ">>> Finished config: $config_name"
  echo "---------------------------------------------"
done

# bash test_tmp.sh
