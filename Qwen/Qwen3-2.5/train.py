# 加载所需要的⼯具包：
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = r'D:\Qwen\Qwen3-2.5\qwen\Qwen2.5-0.5B-Instruct'


"""加载模型参数和tokenize"""
model = AutoModelForCausalLM.from_pretrained(
model_path,
torch_dtype="auto",
device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


"""定义提⽰语（prompt）"""
prompt = "简单介绍一下你自己"
messages = [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": prompt}
]


"""将提⽰语messages转换为⼀个格式化的对话字符串text，并将text转化为张量格式"""
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(text)


"""使⽤generate()函数⽣成内容，并保留⽣成的内容"""
generated_ids = model.generate(
    model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids,generated_ids)
]


"""使⽤tokenizer.batch_decode()对⽣成的ids进⾏解码，得到实际内容"""
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)