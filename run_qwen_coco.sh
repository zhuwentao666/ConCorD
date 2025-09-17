#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

declare -a MODEL_PATHS=(
    # "/visual/instructblip-vicuna-7b" 
    # "/visual/InternVL2-8B" 
    "/visual/llava-1.5-7b-hf" 
    # "/visual/Qwen2.5-VL-7B-Instruct"
)

# 为每个模型配置对应的 ALPHA 值
declare -a ALPHA_VALUES=(
    # 0.2  # instructblip-vicuna-7b 对应的 ALPHA
    # 0.2  # InternVL2-8B 对应的 ALPHA
    0.2 # llava-1.5-7b-hf 对应的 ALPHA
    # 0.2  # Qwen2.5-VL-7B-Instruct 对应的 ALPHA
)

SAVE_DIR="./"
COCO_DATA_FILE="./annotations/coco_data.json"
COCO_IMAGE_PATH="./val2014/"
LOG_PATH="./qwen_coco_logs"

# 对每个模型循环运行
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    ALPHA="${ALPHA_VALUES[$i]}"

    # 设置基础参数
    MODEL_NAME=${MODEL_PATH##*/}

    # ConCorD/DOLA 参数
    USE_ConCorD=false # 设置为 false 或 true
    USE_DOLA=false # 设置为 false 或 true
    START_LAYER=15
    END_LAYER=24
    # ConCorD 独有参数
    THRESHOLD_TOP_P=0.9
    THRESHOLD_TOP_K=50
    INTERMEDIATE_TOP_K=0
    NUM_CONSENSUS_LAYERS=1
    STRUCTURE_TOKEN_BONUS=0.0
    DYNAMIC_ALPHA_FACTOR=0.0

    # DOLA 独有参数 (如果需要，在此添加)
    # DOLA_PARAM_X=value
    # 生成参数
    MAX_NEW_TOKENS=512
    NUM_BEAMS=5
    TOP_P=0.9
    TEMPERATURE=-1

    # 检查 DeCO 和 DOLA 是否互斥
    if [ "$USE_ConCorD" = true ] && [ "$USE_DOLA" = true ]; then
        echo "USE_ConCorD 和 USE_DOLA 不能同时为 true。"
        exit 1
    fi


    # 构建结果文件名，包含关键参数
    if [ "$USE_ConCorD" = true ]; then
        METHOD_STR="ConCorD_t${TEMPERATURE}_s${START_LAYER}_e${END_LAYER}_a${ALPHA}_p${THRESHOLD_TOP_P}_k${THRESHOLD_TOP_K}_ik${INTERMEDIATE_TOP_K}_cl${NUM_CONSENSUS_LAYERS}_sb${STRUCTURE_TOKEN_BONUS}_daf${DYNAMIC_ALPHA_FACTOR}_beam${NUM_BEAMS}"
        ANSWERS_FILE="./${SAVE_DIR}/${MODEL_NAME}_coco_results_${METHOD_STR}.jsonl"
    elif [ "$USE_DOLA" = true ]; then
        # 注意：这里假设 DOLA 也使用 START_LAYER 和 END_LAYER，根据实际情况调整 DOLA_STR
        METHOD_STR="dola_t${TEMPERATURE}_s${START_LAYER}_e${END_LAYER}"
        ANSWERS_FILE="./${SAVE_DIR}/${MODEL_NAME}_coco_results_${METHOD_STR}.jsonl"
    else
        ANSWERS_FILE="./${SAVE_DIR}/${MODEL_NAME}_t${TEMPERATURE}_beam${NUM_BEAMS}_coco_results_baseline.jsonl"
    fi

    # 构建命令参数
    METHOD_ARGS=""
    if [ "$USE_ConCorD" = true ]; then
        METHOD_ARGS="--use_ConCorD --start_layer ${START_LAYER} --end_layer ${END_LAYER} --alpha ${ALPHA} --threshold_top_p ${THRESHOLD_TOP_P} --threshold_top_k ${THRESHOLD_TOP_K} --intermediate_top_k ${INTERMEDIATE_TOP_K} --num_consensus_layers ${NUM_CONSENSUS_LAYERS} --structure_token_bonus ${STRUCTURE_TOKEN_BONUS} --dynamic_alpha_factor ${DYNAMIC_ALPHA_FACTOR}"
    elif [ "$USE_DOLA" = true ]; then
        # 假设 DOLA 需要 --use_dola, --start_layer, --end_layer
        METHOD_ARGS="--use_dola --start_layer ${START_LAYER} --end_layer ${END_LAYER}"
        # 如果有其他 DOLA 参数，像这样添加: --dola_param_x ${DOLA_PARAM_X}
    fi

    # 打印执行信息
    echo "运行Qwen-VL COCO评测"
    echo "结果文件: ${ANSWERS_FILE}"
    if [ "$USE_ConCorD" = true ]; then
        echo "方法: ConCorD"
    elif [ "$USE_DOLA" = true ]; then
        echo "方法: DOLA"
    else
        echo "方法: Baseline"
    fi


    # 执行Python脚本
    python qwen_coco.py \
        --model-path ${MODEL_PATH} \
        --coco-data-file ${COCO_DATA_FILE} \
        --coco-image-path ${COCO_IMAGE_PATH} \
        --answers-file ${ANSWERS_FILE} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        --num_beams ${NUM_BEAMS} \
        --top_p ${TOP_P} \
        --temperature ${TEMPERATURE} \
        --log_path ${LOG_PATH} \
        --use_flash_attn \
        --use_bf16 \
        --prompt "Describe this image in detail." \
        ${METHOD_ARGS} # 使用 METHOD_ARGS 替代 DECO_ARGS

    echo "评测完成，结果保存在 ${ANSWERS_FILE}"
    echo "----------------------------------------"
done
