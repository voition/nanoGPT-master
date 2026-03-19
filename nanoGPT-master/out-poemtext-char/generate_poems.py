# generate_poems.py
import torch
import os
from model import GPTConfig, GPT
from contextlib import nullcontext
import tiktoken

# 配置
out_dir = 'out-poemtext-char'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# 加载模型
ckpt_path = os.path.join(out_dir, r'D:\nanoGPT-master\out-poemtext-char\ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 编码器
enc = tiktoken.get_encoding("gpt2")


# 生成函数
def generate_poem(start_text="", max_new_tokens=200, temperature=0.8, top_k=40):
    # 编码起始文本
    start_ids = enc.encode_ordinary(start_text) if start_text else []
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # 生成
    with torch.no_grad():
        with torch.cuda.amp.autocast() if device == 'cuda' else nullcontext():
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

    # 解码
    generated = enc.decode(y[0].tolist())
    return generated


# 测试不同提示
prompts = [
    "床前明月光",
    "白日依山尽",
    "春眠不觉晓",
    "明月几时有",
    "黄河远上白云间",
]

print("=" * 50)
print("诗歌生成结果")
print("=" * 50)

for prompt in prompts:
    print(f"\n📝 提示词: {prompt}")
    print("-" * 30)
    poem = generate_poem(prompt, max_new_tokens=150, temperature=0.8)
    print(poem)
    print()