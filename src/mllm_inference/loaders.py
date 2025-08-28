from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor
import torch
import math



def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL3/8B/':32, 'InternVL3/38B/': 64, 'InternVL3/78B/':80}[model_name.split('--')[-1]]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def loader_qwen25vl(model_path):
    from transformers import Qwen2_5_VLForConditionalGeneration

    if '32B' not in model_path and '72B' not in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
        )   
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  
            llm_int8_enable_fp32_cpu_offload=True  
            )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto"
            )  

    tokenizer = AutoProcessor.from_pretrained(model_path)
    return model, tokenizer

def loader_internvl3(model_path):
    torch.cuda.empty_cache()
    device_map = split_model(model_path)
    if '38B' not in model_path and '78B' not in model_path:
        model = AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16, 
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map=device_map).eval()
    else:
        #Load in 8 bit for 38B and 78B
        # Define the quantization configuration
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit precision
            llm_int8_threshold=6.0,  
            llm_int8_enable_fp32_cpu_offload=True 
            )
        model = AutoModel.from_pretrained(
                            model_path,
                            quantization_config=quantization_config,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto").eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def load_model(model):
    paths = {
    'internvl3/8B/':'OpenGVLab/InternVL3-8B',
    'internvl3/38B/':'OpenGVLab/InternVL3-38B',
    'internvl3/78B/':'OpenGVLab/InternVL3-78B',
    'qwen2.5vl/7B/': 'Qwen2.5-VL-7B-Instruct/',
    'qwen2.5vl/32B/': 'Qwen2.5-VL-32B-Instruct/',
    'qwen2.5vl/72B/': 'Qwen2.5-VL-72B-Instruct/',
        }
    
    loader_map = {'internvl3': loader_internvl3, 
                  'qwen2.5vl':loader_qwen25vl,
                 }
    model, tokenizer = loader_map[model.split('/')[0]](paths[model])
    return model, tokenizer