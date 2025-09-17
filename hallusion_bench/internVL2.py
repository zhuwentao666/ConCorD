import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, GenerationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_response(model, tokenizer, image_path, question, generation_config):
    """
    Generates a response from the model given an image and a question.

    Args:
        model: The pre-trained InternVL model.
        tokenizer: The tokenizer for the model.
        image_path (str): The path to the input image file.
        question (str): The question to ask the model about the image.
        generation_config (dict): Configuration for the text generation process.

    Returns:
        str: The generated response from the model.
    """
    # Load and preprocess the image
    # Assuming max_num=12 and dtype=torch.bfloat16 based on the context
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_kwargs = {}
    generation_kwargs['generation_config'] = generation_config
    generation_kwargs['do_sample'] = generation_config.do_sample
    generation_kwargs['max_new_tokens'] = generation_config.max_new_tokens
    generation_kwargs['top_k'] = generation_config.top_k
    generation_kwargs['top_p'] = generation_config.top_p
    generation_kwargs['num_beams'] = generation_config.num_beams
    generation_kwargs['temperature'] = generation_config.temperature
    
    question = "<image>\n" + question

    # Generate the response using the model's chat method
    response = model.chat(tokenizer, pixel_values, question, generation_kwargs)

    return response

if __name__ == "__main__":
    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    path = '/visual/InternVL2-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    gen_config = GenerationConfig(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.9,
            use_deco = True,
            early_exit_layers=list(range(18, 25)),
            alpha = 0.1
        )
    image = 'val2014/COCO_val2014_000000203564.jpg'
    question = "Describe the image in detail."
    response = get_response(model, tokenizer, image, question, gen_config)
    print(response)