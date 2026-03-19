import os
import tiktoken
import numpy as np
from zhconv import convert

# 安装：pip install zhconv

input_file_path = 'tang_poet.txt'
output_dir = os.path.dirname(__file__)

# 读取文件
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"原始文本长度: {len(data)} 字符")

# 繁体转简体
print("正在转换为简体字...")
data_simplified = convert(data, 'zh-cn')

# 数据分割
n = len(data_simplified)
train_data = data_simplified[:int(n*0.9)]
val_data = data_simplified[int(n*0.9):]

# 编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"训练集有 {len(train_ids):,} 个 token")
print(f"验证集有 {len(val_ids):,} 个 token")

# 保存
os.makedirs(output_dir, exist_ok=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

print(f"✅ 处理完成！")