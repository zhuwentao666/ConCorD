import argparse
import json
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import glob
import re

def contains_no_word(text):
    return bool(re.search(r'\bno\b', text, re.IGNORECASE))
def evaluate_single_file(input_file):
    """评估单个文件并返回指标"""
    labels = []
    responses = []
    error_ls = []
    origin_ls = []
    # 打开问题josnl
    with open('pope_coco.jsonl', "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            origin_ls.append(item)
            
    i = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "label" not in item or "response" not in item:
                continue
            label = item["label"].lower()
            response = item["response"].lower()
            labels.append(0 if contains_no_word(label) else 1)
            responses.append(0 if contains_no_word(response) else 1)
            # if labels[-1] == responses[-1]:
            #     with open("llava_base_no_error.jsonl", "a", encoding="utf-8") as error_file:
            #         error_file.write(json.dumps(origin_ls[i], ensure_ascii=False) + "\n")
                    
            i += 1

    precision = precision_score(labels, responses)
    recall = recall_score(labels, responses)
    f1 = f1_score(labels, responses)
    accuracy = accuracy_score(labels, responses)
    
    return precision, recall, f1, accuracy

def process_folder(input_folder, output_dir):
    """处理整个文件夹"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有.jsonl文件
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    
    if not jsonl_files:
        print(f"在文件夹 {input_folder} 中没有找到.jsonl文件")
        return
    
    results = {}
    
    for file_path in jsonl_files:
        file_name = os.path.basename(file_path)
        precision, recall, f1, accuracy = evaluate_single_file(file_path)
        
        # 从文件名提取模型和方法信息
        base_name = file_name.replace('.jsonl', '')
        results[base_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
        
        # 保存单个文件结果
        output_file = os.path.join(output_dir, f"result_{file_name}")
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(f"Precision: {precision:.4f}\n")
            out.write(f"Recall:    {recall:.4f}\n")
            out.write(f"F1 Score:  {f1:.4f}\n")
            out.write(f"Accuracy:  {accuracy:.4f}\n")
    
    # 按名字排序
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    # 计算最长名字长度，用于对齐
    max_name_length = max(len(name) for name in results.keys())
    
    # 打印汇总结果
    print("\n" + "="*80)
    print(f"{'模型_方法':^{max_name_length}} | {'F1值':^8} | {'Precision':^10} | {'Recall':^8} | {'Accuracy':^8}")
    print("="*80)
    
    for model_method, metrics in sorted_results:
        print(f"{model_method:<{max_name_length}} | {metrics['f1']:^8.4f} | {metrics['precision']:^10.4f} | {metrics['recall']:^8.4f} | {metrics['accuracy']:^8.4f}")
    
    print("="*80)
    print(f"共评估 {len(results)} 个模型/方法")
    
    # 保存汇总结果
    summary_file = os.path.join(output_dir, "summary_results.txt")
    with open(summary_file, "w", encoding="utf-8") as out:
        out.write("POPE 评估结果汇总\n")
        out.write("="*80 + "\n")
        out.write(f"{'模型_方法':^{max_name_length}} | {'F1值':^8} | {'Precision':^10} | {'Recall':^8} | {'Accuracy':^8}\n")
        out.write("="*80 + "\n")
        
        for model_method, metrics in sorted_results:
            out.write(f"{model_method:<{max_name_length}} | {metrics['f1']:^8.4f} | {metrics['precision']:^10.4f} | {metrics['recall']:^8.4f} | {metrics['accuracy']:^8.4f}\n")
        
        out.write("="*80 + "\n")
        out.write(f"共评估 {len(results)} 个模型/方法\n")
        
        out.write("\n详细结果:\n")
        out.write("="*80 + "\n")
        for model_method, metrics in sorted_results:
            out.write(f"\n{model_method}:\n")
            out.write(f"  Precision: {metrics['precision']:.4f}\n")
            out.write(f"  Recall:    {metrics['recall']:.4f}\n")
            out.write(f"  F1 Score:  {metrics['f1']:.4f}\n")
            out.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
    
    print(f"\n汇总结果已保存到: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate F1 score from JSONL responses.")
    parser.add_argument("input_path", help="Path to the input .jsonl file or folder containing .jsonl files")
    parser.add_argument("output_dir", help="Directory to save the result")
    args = parser.parse_args()

    input_path = args.input_path
    output_dir = args.output_dir
    
    if os.path.isfile(input_path):
        # 处理单个文件（原有功能）
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(input_path)
        output_file = os.path.join(output_dir, f"result_{base_name}")
        
        precision, recall, f1, accuracy = evaluate_single_file(input_path)
        
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(f"Precision: {precision:.4f}\n")
            out.write(f"Recall:    {recall:.4f}\n")
            out.write(f"F1 Score:  {f1:.4f}\n")
            out.write(f"Accuracy:  {accuracy:.4f}\n")
        
        print(f"accuracy: {accuracy:.4f}")
        print(f"Results written to {output_file}")
        
    elif os.path.isdir(input_path):
        # 处理整个文件夹
        process_folder(input_path, output_dir)
    else:
        print(f"错误：{input_path} 不是有效的文件或文件夹路径")

if __name__ == "__main__":
    main()