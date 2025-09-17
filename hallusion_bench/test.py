import json
import re
import os
import glob
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from collections import defaultdict

# 常见否定表达的正则（包含缩写）

def label_from_prediction(pred):
    """如果文本中包含否定词，则预测为 0（否），否则为 1（是）"""
    return 0 if pred.lower().startswith('no') else 1

def process_single_file(json_path):
    # 使用字典存储每个subcategory的数据
    subcategory_data = defaultdict(lambda: {'y_true': [], 'y_pred': []})
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if "VS" in item['category']:
                continue
            
            subcategory = item.get('subcategory', 'unknown')
            true_label = int(item["gt_answer"])
            pred_label = label_from_prediction(item["model_prediction"])
            
            subcategory_data[subcategory]['y_true'].append(true_label)
            subcategory_data[subcategory]['y_pred'].append(pred_label)

    # 计算整体指标
    y_true_all = []
    y_pred_all = []
    for data in subcategory_data.values():
        y_true_all.extend(data['y_true'])
        y_pred_all.extend(data['y_pred'])

    precision_all, recall_all, f1_all, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="binary"
    )

    # 提取模型名称（从文件路径中）
    model_name = os.path.basename(json_path).replace('_results_baseline.jsonl', '').replace('_results_coco.jsonl', '')
    if 'baseline' in json_path:
        model_name += '_baseline'
    elif 'coco' in json_path:
        model_name += '_coco'

    # 计算每个subcategory的F1
    subcategory_f1 = {}
    for subcategory, data in sorted(subcategory_data.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        if len(y_true) == 0:
            subcategory_f1[subcategory] = 0.0
            continue
            
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        subcategory_f1[subcategory] = f1

    # 按指定顺序输出一行结果
    subcategory_order = ['figure', 'illusion', 'math', 'ocr', 'video']
    result_line = f"{model_name}"
    result_line += f"\t{f1_all:.4f}"
    
    for sub in subcategory_order:
        if sub in subcategory_f1:
            result_line += f"\t{subcategory_f1[sub]:.4f}"
        else:
            result_line += f"\t0.0000"
    
    return result_line

def main(path):
    # 判断是文件还是文件夹
    if os.path.isfile(path):
        # 单个文件
        result = process_single_file(path)
        print(result)
    elif os.path.isdir(path):
        # 文件夹，查找所有jsonl文件
        jsonl_files = glob.glob(os.path.join(path, "*.jsonl"))
        
        if not jsonl_files:
            print(f"在文件夹 {path} 中没有找到 .jsonl 文件")
            return
        
        # 打印表头
        print("Model\tOverall_F1\tFigure_F1\tIllusion_F1\tMath_F1\tOCR_F1\tVideo_F1")
        
        # 处理每个文件
        for jsonl_file in sorted(jsonl_files):
            try:
                result = process_single_file(jsonl_file)
                print(result)
            except Exception as e:
                print(f"处理文件 {jsonl_file} 时出错: {e}")
    else:
        print(f"错误: {path} 不是有效的文件或文件夹")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python test.py path/to/your.jsonl 或 python test.py path/to/folder/")
    else:
        main(sys.argv[1])