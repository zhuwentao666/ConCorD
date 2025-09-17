#!/bin/bash

# filepath: /home/cdw/English_instruciton/DeCO/run_evaluate_chair.sh

# 设置默认参数
INPUT_FILE="ConCorD/Qwen-VL-Chat_coco_results_ConCorD_t-1_s15_e24_a0.05_p0.9_k50_ik1_cl3_sb0.0_daf1.0_beam1.jsonl"  # 替换为实际的输入文件路径
COCO_PATH="./annotations"  # 替换为 COCO 数据集的 annotations 目录路径
CACHE_PATH="chair_cache.pkl"          # 缓存文件路径
OUTPUT_FILE="output_results.json"     # 输出文件路径

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file not found at $INPUT_FILE"
  exit 1
fi

# 检查 COCO annotations 目录是否存在
if [ ! -d "$COCO_PATH" ]; then
  echo "Error: COCO annotations directory not found at $COCO_PATH"
  exit 1
fi

# 运行 Python 脚本
python3 evaluate_chair.py \
  --input_file "$INPUT_FILE" \
  --coco_path "$COCO_PATH" \
  --cache_path "$CACHE_PATH" \
  --output_file "$OUTPUT_FILE"

# 检查脚本是否成功运行
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully. Results saved to $OUTPUT_FILE"
else
  echo "Error: Evaluation failed."
fi