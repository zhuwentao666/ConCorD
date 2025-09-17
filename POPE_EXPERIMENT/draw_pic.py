import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def open_jsonl_files(file1_path, file2_path):
    """
    打开并读取两个JSONL文件
    
    参数:
    file1_path (str): 第一个JSONL文件的路径
    file2_path (str): 第二个JSONL文件的路径
    
    返回:
    tuple: 包含两个文件内容的元组，每个文件内容是一个列表，列表中的每个元素是一个JSON对象
    """
    file1_data = []
    file2_data = []
    
    # 打开并读取第一个文件
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            for line in f1:
                if line.strip():  # 跳过空行
                    file1_data.append(json.loads(line))
        print(f"成功读取文件: {file1_path}")
        print(f"第一个文件中包含 {len(file1_data)} 条记录")
    except Exception as e:
        print(f"读取文件 {file1_path} 时出错: {e}")
    
    # 打开并读取第二个文件
    try:
        with open(file2_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                if line.strip():  # 跳过空行
                    file2_data.append(json.loads(line))
        print(f"成功读取文件: {file2_path}")
        print(f"第二个文件中包含 {len(file2_data)} 条记录")
    except Exception as e:
        print(f"读取文件 {file2_path} 时出错: {e}")
    
    return file1_data, file2_data

def collect_probabilities(data1, data2):
    """
    收集每一层的幻觉token和非幻觉token的完整概率列表
    
    参数:
    data1: 包含标签信息的数据
    data2: 包含各层token概率的数据
    
    返回:
    tuple: 包含每层幻觉token和非幻觉token概率列表的字典
    """
    # 初始化存储每层概率列表的字典
    hallucination_probs = defaultdict(list)
    non_hallucination_probs = defaultdict(list)
    
    # 遍历每个样本
    for input_item, output_item in zip(data1, data2):
        label = input_item["label"]
        
        # 确定幻觉token和非幻觉token
        if label == 'yes':
            hallucination_token = "no"
            non_hallucination_token = "yes"
        else:
            hallucination_token = "yes"
            non_hallucination_token = "no"
        
        # 遍历每一层，收集概率
        for layer, probs in output_item.items():
            # if int(layer) < 14:
            #     continue
            hallucination_prob = probs.get(hallucination_token, [0])[0]
            non_hallucination_prob = probs.get(non_hallucination_token, [0])[0]
            
            hallucination_probs[int(layer)].append(hallucination_prob)
            non_hallucination_probs[int(layer)].append(non_hallucination_prob)
            
    return hallucination_probs, non_hallucination_probs


def calculate_hallucination_degree(data1, data2):
    """
    计算每一层所有样本的平均幻觉度。
    幻觉度定义为: P(幻觉token) - P(非幻觉token)
    
    参数:
    data1: 包含标签信息的数据
    data2: 包含各层token概率的数据
    
    返回:
    tuple: (layers, mean_degrees, std_degrees)
           layers: 层数列表
           mean_degrees: 每层平均幻觉度列表
           std_degrees: 每层幻觉度的标准差列表
    """
    hallucination_degrees = defaultdict(list)
    
    # 遍历每个样本
    for input_item, output_item in zip(data1, data2):
        label = input_item["label"]
        
        # 确定幻觉token和非幻觉token
        if label == 'yes':
            hallucination_token = "no"
            non_hallucination_token = "yes"
        else:
            hallucination_token = "yes"
            non_hallucination_token = "no"
            
        # 遍历每一层，计算幻觉度
        for layer, probs in output_item.items():
            hallucination_prob = probs.get(hallucination_token, [0])[0]
            non_hallucination_prob = probs.get(non_hallucination_token, [0])[0]
            
            degree = hallucination_prob - non_hallucination_prob
            hallucination_degrees[int(layer)].append(degree)
            
    # 计算每层的平均值和标准差
    sorted_layers = sorted(hallucination_degrees.keys())
    mean_degrees = [np.mean(hallucination_degrees[layer]) for layer in sorted_layers]
    std_degrees = [np.std(hallucination_degrees[layer]) for layer in sorted_layers]
    
    return sorted_layers, mean_degrees, std_degrees

def plot_violin_plot(hallucination_degrees):
    """
    使用小提琴图展示每一层幻觉度的分布情况。

    参数:
    hallucination_degrees: 包含每层幻觉度列表的字典
    """
    # 将数据转换为 pandas DataFrame，这是 seaborn 喜欢的格式
    import seaborn as sns # 导入 seaborn
    import pandas as pd  # 导入 pandas
    layers = []
    degrees = []
    for layer, degree_list in hallucination_degrees.items():
        layers.extend([layer] * len(degree_list))
        degrees.extend(degree_list)
    
    df = pd.DataFrame({'Layer': layers, 'Hallucination Degree': degrees})
    print(df['Hallucination Degree'].describe())  # 打印统计信息
    
    
    plt.figure(figsize=(20, 10))
    
    # 使用 seaborn 绘制小提琴图
    sns.violinplot(x='Layer', y='Hallucination Degree', data=df, palette='coolwarm', inner='quartile')
    
    # 添加 y=0 的参考线
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Baseline (Degree = 0)')
    
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Hallucination Degree (P_hal - P_non_hal)', fontsize=14)
    plt.title('Distribution of Hallucination Degree Across Layers (Violin Plot)', fontsize=16)
    plt.xticks(rotation=45, fontsize=12) # 如果x轴标签重叠，可以旋转
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('hallucination_degree_violin.png', dpi=300)
    print("小提琴图已保存为 hallucination_degree_violin.png")
    
    
def plot_hallucination_trend(layers, mean_degrees, std_degrees):
    """
    绘制平均幻觉度随层数变化的趋势图。
    
    参数:
    layers: 层数列表
    mean_degrees: 每层平均幻觉度列表
    std_degrees: 每层幻觉度的标准差列表
    """
    plt.figure(figsize=(18, 8))
    
    mean_degrees = np.array(mean_degrees)
    std_degrees = np.array(std_degrees)
    
    # 绘制平均幻觉度曲线
    plt.plot(layers, mean_degrees, marker='o', linestyle='-', color='royalblue', label='Average Hallucination Degree')
    
    # 绘制标准差范围 (置信区间)，代表数据波动范围
    plt.fill_between(layers, mean_degrees - std_degrees, mean_degrees + std_degrees,
                     color='lightblue', alpha=0.4, label='Standard Deviation')
                     
    # 添加 y=0 的参考线，作为幻觉与非幻觉倾向的分界线
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (Degree = 0)')
    
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Hallucination Degree (P_hal - P_non_hal)', fontsize=14)
    plt.title('Trend of Average Hallucination Degree Across Layers', fontsize=16)
    plt.xticks(layers, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('hallucination_degree_trend.png', dpi=300)
    print("趋势图已保存为 hallucination_degree_trend.png")
    # plt.show()
    
def drow_vip(file1_path, file2_path):
    
    
    # 读取数据
    data1, data2 = open_jsonl_files(file1_path, file2_path)
    
    if not data1 or not data2:
        print("数据加载失败，程序退出。")
        return

    # 收集每层的概率
    hallucination_probs, non_hallucination_probs = collect_probabilities(data1, data2)
    
    # 绘制箱形图
    # plot_box_plot(hallucination_probs, non_hallucination_probs)

    # 计算幻觉度
    hallucination_degrees = defaultdict(list)
    for input_item, output_item in zip(data1, data2):
        label = input_item["label"]
        if label == 'yes':
            hallucination_token = "no"
            non_hallucination_token = "yes"
        else:
            hallucination_token = "yes"
            non_hallucination_token = "no"
        for layer, probs in output_item.items():
            hallucination_prob = probs.get(hallucination_token, [0])[0]
            non_hallucination_prob = probs.get(non_hallucination_token, [0])[0]
            degree = hallucination_prob - non_hallucination_prob
            hallucination_degrees[int(layer)].append(degree)

    # 绘制小提琴图
    plot_violin_plot(hallucination_degrees)
    
def plot_box_plot(hallucination_probs, non_hallucination_probs):
    """
    绘制幻觉token和非幻觉token在各层概率分布的箱形图
    
    参数:
    hallucination_probs: 每层幻觉token的概率列表
    non_hallucination_probs: 每层非幻觉token的概率列表
    """
    layers = sorted(hallucination_probs.keys())
    
    # 准备绘图数据
    h_data = [hallucination_probs[layer] for layer in layers]
    non_h_data = [non_hallucination_probs[layer] for layer in layers]
    
    plt.figure(figsize=(18, 6))
    
    # 设置箱形图的位置
    positions_h = np.array(range(len(layers))) * 2.0 - 0.4
    positions_non_h = np.array(range(len(layers))) * 2.0 + 0.4
    
    # 绘制幻觉token的箱形图，不显示离群点
    bp_h = plt.boxplot(h_data, positions=positions_h, widths=0.6, patch_artist=True,
                       showfliers=False,  # 添加此行
                       boxprops=dict(facecolor='lightcoral', alpha=0.8),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))

    # 绘制非幻觉token的箱形图，不显示离群点
    bp_non_h = plt.boxplot(non_h_data, positions=positions_non_h, widths=0.6, patch_artist=True,
                           showfliers=False,  # 添加此行
                           boxprops=dict(facecolor='lightblue', alpha=0.8),
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(color='black'),
                           capprops=dict(color='black'))

    plt.xlabel('Layer', fontsize=20)
    plt.ylabel('Probabilities', fontsize=20)
    # plt.title('Box', fontsize=16)
    
    # 设置X轴刻度和标签
    plt.xticks(np.arange(0, len(layers) * 2, 2), layers, fontsize=20)
    plt.yticks(fontsize=20)
    
    # 添加图例
    plt.legend([bp_h["boxes"][0], bp_non_h["boxes"][0]], ['Token With Hallucination', 'Token Without Hallucination'], loc='upper left', fontsize=20)
    
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig('token_probabilities_boxplot.png', dpi=300)
    print("图像已保存为 token_probabilities_boxplot.png")
    # plt.show()

if __name__ == "__main__":
    # 在这里指定两个JSONL文件的路径
    file1_path = "instructblip-vicuna-7b_t-1_beam1POPE_coco_results_baseline.jsonl"
    file2_path = "blip_error_intermediate_layer_probabilities.jsonl"
    
    # 打开两个文件
    data1, data2 = open_jsonl_files(file1_path, file2_path)
    
    filter_data1, filter_data2 = [], []
    for i_1, i_2 in zip(data1, data2):
        res, label = i_1["response"], i_1["label"]
        if res.lower() != label.lower():
            filter_data1.append(i_1)
            filter_data2.append(i_2)
    
    # 收集每层的概率
    hallucination_probs, non_hallucination_probs = collect_probabilities(filter_data1, filter_data2)
    
    # 绘制箱形图
    plot_box_plot(hallucination_probs, non_hallucination_probs)
    
    # layers, mean_degrees, std_degrees = calculate_hallucination_degree(data1, data2)
    # plot_hallucination_trend(layers, mean_degrees, std_degrees)
    # drow_vip(file1_path, file2_path)
    