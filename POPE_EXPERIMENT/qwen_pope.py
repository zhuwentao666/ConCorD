import os
import argparse
import torch
import json
from tqdm import tqdm
import sys
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, GenerationConfig, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer,AutoModelForImageTextToText, LlavaNextForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, AutoModel, Blip2Processor, Blip2ForConditionalGeneration
from qwen_vl_utils import process_vision_info
import logging
import datetime
from glob import glob
from transformers import set_seed
import json
from internVL2 import get_response
import re

# 创建简化版的 dist_util 和 logger 模块，避免依赖外部文件
class SimpleDist:
    @staticmethod
    def setup_dist(args):
        # 简化版本，不进行分布式设置
        pass

    @staticmethod
    def is_primary():
        # 单进程模式下总是主进程
        return True

    @staticmethod
    def device():
        # 返回当前可用设备
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_entity(text: str) -> str | None:
    """
    从 text 中匹配第一个 'a ' 或 'an ' 之后的所有内容
    返回实体字符串，找不到则返回 None
    """
    m = re.search(r'\b(?:an|a)\s+(.+)', text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None

def is_a_or_an(text: str) -> bool:
    if " a " in text:
        return "a"
    elif " an " in text:
        return "an"
    

# 创建简化版的 logger
def create_simple_logger(log_dir=None):
    logger = logging.getLogger("qwen_coco")
    logger.setLevel(logging.INFO)
    
    # 如果已经有处理程序，不再添加
    if logger.handlers:
        return logger
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_file = os.path.join(log_dir, f'log-{time_str}.txt')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# 创建 dist_util 的实例
dist_util = SimpleDist()

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        import requests
        from io import BytesIO
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def generate_response(model, processor, image_file, prompt, gen_config):
    # 设置生成配置
    with torch.no_grad():
        if 'Qwen-VL-Chat' in args.model_path:
            for k, v in gen_config.to_dict().items():
                model.generation_config.__setattr__(k, v)
            query = model.tokenizer.from_list_format([
                {'image': image_file},
                {'text': prompt},
            ])
            output_text, history = model.chat(model.tokenizer, query=query, history=None)
            
        elif 'Qwen2.5-VL' in args.model_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            
            generated_ids = model.generate(**inputs, generation_config=gen_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]
            
        elif 'llava' in args.model_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_file,
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, generation_config=gen_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]
        elif 'instructblip' in args.model_path:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                    **inputs,
                    generation_config=gen_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]
            
        elif 'InternVL2' in args.model_path:
            output_text = get_response(model, model.tokenizer, image_file, prompt, gen_config)

        elif 'SmolVLM2' in args.model_path:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "path": image_file}
                    ]
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device).to(dtype=torch.bfloat16)
            # Generate outputs
            generated_ids = model.generate(**inputs, generation_config=gen_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            generated_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
            )
            output_text = generated_texts[0]
        elif 'blip2' in args.model_path:
            image = Image.open(image_file).convert("RGB")
            inputs = processor(images=image, text=f"Question: {prompt} Answer:", return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **inputs,
                generation_config=gen_config
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]
            
        else:
            raise ValueError("Unsupported model type.")

    return output_text


def eval_model(args):
    # 设置GPU和日志 - 使用简化版本
    dist_util.setup_dist(args)
    device = dist_util.device()

    # 设置实验文件夹
    if dist_util.is_primary():
        os.makedirs(args.log_path, exist_ok=True)
        experiment_dir = f"{args.log_path}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_simple_logger(experiment_dir)
        logger.info(f"实验目录创建在 {experiment_dir}")
    else:
        logger = create_simple_logger(None)
    

    tokenizer = None
    processor = None
    
    # 如果启用flash_attention_2
    if args.use_flash_attn:
        if 'Qwen2.5-VL' in args.model_path:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        elif 'llava' in args.model_path:
            model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
            model.config.num_hidden_layers = 32
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            
        elif 'Qwen-VL-Chat' in args.model_path:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()
            model.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        elif 'instructblip' in args.model_path:
            model = InstructBlipForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            processor = InstructBlipProcessor.from_pretrained(args.model_path, trust_remote_code=True)
            model.config.num_hidden_layers = 32
        
        elif 'InternVL2' in args.model_path:
            model = AutoModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map='auto'
            ).eval()
            model.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
            model.config.num_hidden_layers = 32
            
        elif 'SmolVLM2' in args.model_path:
            processor = AutoProcessor.from_pretrained(args.model_path)
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2"
            ).to("cuda")
            model.config.num_hidden_layers = 24
        elif 'blip2' in args.model_path:
            processor = Blip2Processor.from_pretrained(args.model_path)
            model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_path, device_map={"": 0}, torch_dtype=torch.float16
        ).to("cuda")
            model.config.num_hidden_layers = 32
            
        else:
            raise ValueError("Unsupported model type for flash attention.")
    else:
        raise ValueError("Flash attention is not enabled. Please set --use_flash_attn.")
    
    # 设置生成配置
    gen_config = GenerationConfig(
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        do_sample=True if args.temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k = 50,
        seed = 42
        # repetition_penalty=1.1 if args.temperature > 0 else 1.0,
    )

    # 设置DECO相关参数
    gen_config.use_ConCorD = args.use_ConCorD
    gen_config.early_exit_layers = list(range(args.start_layer, args.end_layer))
    gen_config.alpha = args.alpha if args.use_ConCorD else 0.0
    gen_config.threshold_top_p = args.threshold_top_p if args.use_ConCorD else 0.0
    gen_config.threshold_top_k = args.threshold_top_k if args.use_ConCorD else 0
    gen_config.intermediate_top_k = args.intermediate_top_k 
    gen_config.num_consensus_layers = args.num_consensus_layers
    gen_config.structure_token_bonus = args.structure_token_bonus
    gen_config.dynamic_alpha_factor = args.dynamic_alpha_factor
    
    assert not args.use_dola or not args.use_ConCorD
    if args.use_dola:
        gen_config.dola_layers = gen_config.early_exit_layers
        print("启用Dola")
    
    print(f"参数设置:{gen_config}")
    print(f"DECO层范围: {gen_config.early_exit_layers}")
    # print(f"Model: {model}")
    
    # 范围必须小于模型的层数
    if args.use_ConCorD:
        assert args.start_layer < model.config.num_hidden_layers
        assert args.end_layer <= model.config.num_hidden_layers
        assert args.start_layer < args.end_layer
    
    # 创建结果文件
    print(f"结果文件: {args.answers_file}")
    answers_file = args.answers_file
    
    # 读取COCO数据集,json格式
    
    data = []
    
    with open(args.coco_data_file, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Processing JSONL"):
            data.append(line)
    
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        for line in tqdm(data):
            line = json.loads(line)
            image_file = os.path.join(args.coco_image_path, line['image'])
            entity = extract_entity(line['text'])
            label = line['label']
            # 准备提示词
            prompt = args.prompt
            if entity is None:
                print("实体为空")
                continue
            
            prompt = prompt.replace('<object>', entity)
            prompt = line['text'] +  ' Answer me with yes or no.'
            a_or_an = is_a_or_an(line['text'])
            
            # 生成响应
            output_text = generate_response(model, processor, image_file, prompt, gen_config)
            response = output_text
            
            # 日志记录
            logger.info(f"[{image_file}]")
            logger.info(f"提示: {prompt}")
            logger.info(f"响应: {response}")
            
            # 保存结果
            res_dict = {"image_file": image_file, "response": response, "label":label}
            ans_file.write(json.dumps(res_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/visual/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--coco-data-file", type=str, default="../opera_log/llava-1.5/ours.jsonl", 
                       dest="coco_data_file")  # 添加dest参数指定属性名
    parser.add_argument("--coco-image-path", type=str, default="../val2014/", 
                       dest="coco_image_path")  # 添加dest参数指定属性名
    parser.add_argument(
        "--answers-file",
        type=str,
        default=None,  # 将默认值设置为 None
        dest="answers_file",
    )
    parser.add_argument("--prompt", type=str, default="Is there a/an <object> in the image? Answer me with yes or no.")
    parser.add_argument("--temperature", type=float, default=-1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--log_path", type=str, default="./qwen_coco_logs")
    parser.add_argument("--use_ConCorD", action="store_true", default=False)
    parser.add_argument("--use_dola", action="store_true", default=False)
    parser.add_argument("--start_layer", type=int, default=20)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=50)
    parser.add_argument("--intermediate_top_k", type=int, default=20)
    parser.add_argument("--num_consensus_layers", type=int, default=3)
    parser.add_argument("--structure_token_bonus", type=float, default=0.0)
    parser.add_argument("--dynamic_alpha_factor", type=float, default=0.1)
    

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
