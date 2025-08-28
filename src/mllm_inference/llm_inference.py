import os
from utils import *
import time
import mimetypes
import warnings
warnings.filterwarnings("ignore", message=r"Setting `pad_token_id` to `eos_token_id`.*", module="transformers")


try:
    from openai import OpenAI
    gpt_client = OpenAI(api_key=os.getenv('YOUR_OPENAI_API_KEY'))
except:
    pass

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.getenv("YOUR_GEMINI_API_KEY"))
except:
    pass


def generate_answer_qwen25vl(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    if image_path:
        image = Image.open(image_path)
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, {"type": "image"}, ],},]
    else: 
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, ],},]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]   
    return response

def generate_answer_qwen25vl_32B(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    dtype   = next(model.parameters()).dtype   # bfloat16 or float16
    device  = next(model.parameters()).device  
    if image_path:
        image = Image.open(image_path).to(dtype=dtype, device=device)
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, {"type": "image"}, ],},]
    else: 
        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt}, ],},]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]   
    return response


def generate_answer_internvl3(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
    pixel_values = load_image_internvl2(image_path, max_num=12).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


def generate_answer_internvl3_38B(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens=200):
    dtype   = next(model.parameters()).dtype   # bfloat16 or float16
    device  = next(model.parameters()).device 

    generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
    pixel_values = load_image_internvl2(image_path, max_num=12).to(dtype=dtype, device=device).cuda()
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response


def generate_answer_gemini(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    #Prepare input
    if image_path:
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        media_type, _ = mimetypes.guess_type(image_path)
        if media_type is None:                  
            media_type = "image/png"
    content = (genai.types.Part.from_bytes(
                data=img_bytes,
                mime_type=media_type,
                ), 
                prompt
            )
    #Generate response
    try:
        response = gemini_client.models.generate_content(contents= content, 
                                                        model= model,
                                                config={
                                                    "temperature": 0.0,
                                                    "top_p": 1,
                                                    "top_k": 1,
                                                    "max_output_tokens": max_tokens
                                                }
                                                )
        output = response.text
        usage_input = response.usage_metadata.prompt_token_count
        usage_output = response.usage_metadata.candidates_token_count
        usage = usage_input + usage_output
    except:
        output = ''
        usage = 0
    time.sleep(2)
    return output, usage



def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = gpt_client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

def generate_answer_gpt4v(image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens):
    if model=='GPT41':
        deployment_name ='gpt-4.1-2025-04-14' 
    else: 
        deployment_name = 'o3-2025-04-16'  

    content = [{"type": "input_text", "text": prompt}]
    if image_path:
        file_id = create_file(image_path)
        content += [{"type":"input_image","file_id": file_id}]
    messages=[
    { 
    "role": "user",
    "content": content,
    }

    ]
    if model=='o3':
        completion = gpt_client.responses.create(
        model=deployment_name,
        input=messages
        )     
    else:
        completion = gpt_client.responses.create(
        model=deployment_name,
        temperature=0,
        input=messages, 
        max_output_tokens=max_tokens 
        )       
    output = completion.output_text
    usage = completion.usage.total_tokens
    
    
    time.sleep(2)
    return output, usage





def generate_answer(image_path, prompt, tokenizer, image_processor, context_len, model, template, max_tokens=200):
    prompt_map = {'internvl3':generate_answer_internvl3, 'internvl3/38B/': generate_answer_internvl3_38B,
                  'internvl3/78B/': generate_answer_internvl3_38B,
                  'qwen2.5vl/32B/': generate_answer_qwen25vl_32B,  'qwen2.5vl/72B/': generate_answer_qwen25vl_32B,
                  'qwen2.5vl': generate_answer_qwen25vl, 'gemini-1.5-flash': generate_answer_gemini, 
                  'gemini-1.5-pro': generate_answer_gemini,
                   'o3': generate_answer_gpt4v, 'GPT41': generate_answer_gpt4v,
                }
    if template not in ['internvl3/38B/', 'internvl3/78B/', 'qwen2.5vl/32B/', 'qwen2.5vl/72B/']:
        generation_type = template.split('/')[0]
    else:
        generation_type = template
    answer = prompt_map[generation_type](image_path, prompt, tokenizer, image_processor, context_len, model, max_tokens)
    return answer