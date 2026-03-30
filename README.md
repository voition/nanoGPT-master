# README

# [nanoGPT]()

⽬前市⾯上⼤多数可⽤的GPT都⾮常的庞⼤，初学者很难学习和复现。nanoGPT项⽬[1]是使⽤Pytorch对GPT的⼀个复现，
包括训练和推理，其⽬的是做到⼩巧、⼲净、可解释性并且能⽤于教育。
本案例将训练两个模型：⼀个是使⽤由58000⾸诗词构成的诗歌数据集，
训练⼀个歌词⽣成的GPT；另⼀个是使⽤约 124万个字符构成的《天⻰⼋部》⽂本，训练⼀个具有《天⻰⼋部》⻛格的GPT。

# nanoGPT项目流程

## 安装配置库

`pip install torch numpy transformers datasets tiktoken wandb tqdm`



## 下载数据集

在终端运行`python data/shakespeare_char/prepare.py`
生成数据集莎士比亚input.txt文件  



## 模型训练

在终端运行`python train.py config/train_shakespeare_char.py`  


## 运行效果

运行`python sample.py --out_dir=out-shakespeare-char`  
通过采样脚本进行抽取结果  
其结果（部分）：

```
C:\Users\28207\anaconda3\envs\aicond\python.exe D:\nanoGPT-master\out-poemtext-char\look.py 
模型 state_dict 的键（层名称）：
['transformer.wte.weight', 'transformer.wpe.weight', 'transformer.h.0.ln_1.weight', 'transformer.h.0.attn.c_attn.weight', 'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.ln_2.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_proj.weight', 'transformer.h.1.ln_1.weight', 'transformer.h.1.attn.c_attn.weight', 'transformer.h.1.attn.c_proj.weight', 'transformer.h.1.ln_2.weight', 'transformer.h.1.mlp.c_fc.weight', 'transformer.h.1.mlp.c_proj.weight', 'transformer.h.2.ln_1.weight', 'transformer.h.2.attn.c_attn.weight', 'transformer.h.2.attn.c_proj.weight', 'transformer.h.2.ln_2.weight', 'transformer.h.2.mlp.c_fc.weight', 'transformer.h.2.mlp.c_proj.weight', 'transformer.h.3.ln_1.weight', 'transformer.h.3.attn.c_attn.weight', 'transformer.h.3.attn.c_proj.weight', 'transformer.h.3.ln_2.weight', 'transformer.h.3.mlp.c_fc.weight', 'transformer.h.3.mlp.c_proj.weight', 'transformer.h.4.ln_1.weight', 'transformer.h.4.attn.c_attn.weight', 'transformer.h.4.attn.c_proj.weight', 'transformer.h.4.ln_2.weight', 'transformer.h.4.mlp.c_fc.weight', 'transformer.h.4.mlp.c_proj.weight', 'transformer.h.5.ln_1.weight', 'transformer.h.5.attn.c_attn.weight', 'transformer.h.5.attn.c_proj.weight', 'transformer.h.5.ln_2.weight', 'transformer.h.5.mlp.c_fc.weight', 'transformer.h.5.mlp.c_proj.weight', 'transformer.ln_f.weight', 'lm_head.weight']
共有 40 个参数组

优化器 state_dict 的键：
dict_keys(['state', 'param_groups'])

模型配置：
{'out_dir': 'out-poemtext-char', 'eval_interval': 250, 'log_interval': 10, 'eval_iters': 200, 'eval_only': False, 'always_save_checkpoint': False, 'init_from': 'scratch', 'wandb_log': False, 'wandb_project': 'owt', 'wandb_run_name': 'gpt2', 'dataset': 'poemtext', 'gradient_accumulation_steps': 1, 'batch_size': 32, 'block_size': 256, 'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'dropout': 0.2, 'bias': False, 'learning_rate': 0.001, 'max_iters': 5000, 'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.99, 'grad_clip': 1.0, 'decay_lr': True, 'warmup_iters': 100, 'lr_decay_iters': 5000, 'min_lr': 0.0001, 'backend': 'nccl', 'device': 'cuda', 'dtype': 'bfloat16', 'compile': False}

模型超参数：
{'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'block_size': 256, 'bias': False, 'vocab_size': 50304, 'dropout': 0.2}

当前迭代步数：5000
最佳验证损失：2.711282968521118

进程已结束，退出代码为 0

```

## 使用唐诗作为训练数据进行训练

其步骤和上述一致，将数据集修改为唐诗即可
修改参数如下：

```
out_dir = 'out-poemtext-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

always_save_checkpoint = False

dataset = 'poemtext'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 previous characters

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
```

其训练过程(部分)：  

```
4290: loss 2.4981, time 541.95ms, mfu 0.84%
iter 4300: loss 2.6002, time 540.15ms, mfu 0.85%
iter 4310: loss 2.4962, time 540.83ms, mfu 0.85%
iter 4320: loss 2.4972, time 543.07ms, mfu 0.86%
iter 4330: loss 2.5158, time 546.33ms, mfu 0.86%
iter 4340: loss 2.5619, time 537.65ms, mfu 0.87%
iter 4350: loss 2.5123, time 544.58ms, mfu 0.87%
iter 4360: loss 2.5313, time 540.60ms, mfu 0.87%
iter 4370: loss 2.5671, time 546.66ms, mfu 0.88%
iter 4380: loss 2.5436, time 541.72ms, mfu 0.88%
iter 4390: loss 2.5560, time 541.02ms, mfu 0.88%
iter 4400: loss 2.4786, time 543.43ms, mfu 0.88%
iter 4410: loss 2.5016, time 545.10ms, mfu 0.89%
iter 4420: loss 2.5575, time 543.02ms, mfu 0.89%
iter 4430: loss 2.5311, time 540.92ms, mfu 0.89%
iter 4440: loss 2.5134, time 543.99ms, mfu 0.89%
iter 4450: loss 2.5363, time 540.87ms, mfu 0.89%
iter 4460: loss 2.5488, time 543.04ms, mfu 0.89%
iter 4470: loss 2.5178, time 536.86ms, mfu 0.89%
iter 4480: loss 2.4945, time 534.60ms, mfu 0.90%
iter 4490: loss 2.5170, time 544.19ms, mfu 0.90%
step 4500: train loss 2.4598, val loss 2.7376
saving checkpoint to out-poemtext-char
iter 4500: loss 2.4889, time 60048.85ms, mfu 0.81%
iter 4510: loss 2.4686, time 541.25ms, mfu 0.82%
iter 4520: loss 2.5057, time 541.68ms, mfu 0.83%
iter 4530: loss 2.5576, time 539.76ms, mfu 0.84%
iter 4540: loss 2.5251, time 539.69ms, mfu 0.84%
iter 4550: loss 2.5172, time 540.20ms, mfu 0.85%
iter 4560: loss 2.5758, time 539.72ms, mfu 0.85%
iter 4570: loss 2.5248, time 539.98ms, mfu 0.86%
iter 4580: loss 2.5541, time 543.28ms, mfu 0.86%
iter 4590: loss 2.5575, time 540.33ms, mfu 0.87%
iter 4600: loss 2.4910, time 539.43ms, mfu 0.87%
iter 4610: loss 2.5374, time 542.08ms, mfu 0.88%
iter 4620: loss 2.4483, time 537.46ms, mfu 0.88%
iter 4630: loss 2.5038, time 543.73ms, mfu 0.88%
iter 4640: loss 2.5948, time 538.77ms, mfu 0.88%
iter 4650: loss 2.4830, time 542.26ms, mfu 0.89%
iter 4660: loss 2.5171, time 541.42ms, mfu 0.89%
iter 4670: loss 2.5283, time 541.36ms, mfu 0.89%
iter 4680: loss 2.4921, time 541.33ms, mfu 0.89%
iter 4690: loss 2.5170, time 541.20ms, mfu 0.89%
iter 4700: loss 2.4548, time 542.45ms, mfu 0.89%
iter 4710: loss 2.4984, time 539.94ms, mfu 0.90%
iter 4720: loss 2.4679, time 540.46ms, mfu 0.90%
iter 4730: loss 2.5228, time 540.32ms, mfu 0.90%
iter 4740: loss 2.5583, time 545.37ms, mfu 0.90%
step 4750: train loss 2.4507, val loss 2.7165
saving checkpoint to out-poemtext-char
iter 4750: loss 2.4873, time 60106.61ms, mfu 0.81%
iter 4760: loss 2.5049, time 535.78ms, mfu 0.82%
iter 4770: loss 2.4823, time 533.13ms, mfu 0.83%
iter 4780: loss 2.5220, time 538.70ms, mfu 0.84%
iter 4790: loss 2.5099, time 539.71ms, mfu 0.84%
iter 4800: loss 2.5359, time 541.17ms, mfu 0.85%
iter 4810: loss 2.5165, time 541.71ms, mfu 0.86%
iter 4820: loss 2.4569, time 536.21ms, mfu 0.86%
iter 4830: loss 2.5754, time 537.84ms, mfu 0.87%
iter 4840: loss 2.5272, time 541.90ms, mfu 0.87%
iter 4850: loss 2.5079, time 539.28ms, mfu 0.87%
iter 4860: loss 2.5058, time 541.74ms, mfu 0.88%
iter 4870: loss 2.4552, time 544.33ms, mfu 0.88%
iter 4880: loss 2.5224, time 544.10ms, mfu 0.88%
iter 4890: loss 2.5190, time 546.18ms, mfu 0.88%
iter 4900: loss 2.4875, time 540.73ms, mfu 0.89%
iter 4910: loss 2.5374, time 545.92ms, mfu 0.89%
iter 4920: loss 2.4895, time 545.05ms, mfu 0.89%
iter 4930: loss 2.5355, time 539.98ms, mfu 0.89%
iter 4940: loss 2.5129, time 541.46ms, mfu 0.89%
iter 4950: loss 2.4921, time 543.53ms, mfu 0.89%
iter 4960: loss 2.5099, time 541.38ms, mfu 0.89%
iter 4970: loss 2.5437, time 547.95ms, mfu 0.89%
iter 4980: loss 2.5482, time 525.29ms, mfu 0.90%
iter 4990: loss 2.5013, time 543.77ms, mfu 0.90%
step 5000: train loss 2.4428, val loss 2.7113
saving checkpoint to out-poemtext-char
iter 5000: loss 2.4821, time 59987.69ms, mfu 0.81%
```


其最后输出结果截图：  

<img width="1449" height="426" alt="Snipaste_2026-03-24_20-03-16" src="https://github.com/user-attachments/assets/e03f3afa-7845-4ff9-85b1-dfd1ba161fe0" />


# nanoGPT思考题

## 1使用《天龙八部》数据集训练一个GPT模形，并生成结果

要使用《天龙八部》txt 数据集训练 GPT 模型，首先需要将小说文本整理成适合

训练的格式。通常的步骤是：准备一个 tianlong.txt 文件，放在 data 目录下；

然后运行数据预处理脚本，将文本转换为训练所需的 token 序列，并保存为 numpy

数组。接着，修改训练配置文件中的 dataset 参数，指向新准备的数据集名称。

在配置文件中，还需要调整模型参数如 n_layer、n_head 等以适应文本规模。最

后，运行训练脚本，模型将开始学习《天龙八部》的语言风格。训练完成后，使

用采样脚本 sample.py，并加载训练好的模型权重，即可生成文本。预期生成的

内容会带有武侠小说的味道，如出现“乔峰”、“降龙十八掌”等词汇，语句风

格接近金庸原文，但故事情节可能是模型自己组合的。

## 2训练文件参数含义与修改实验

在 `config/train_poemtext_char.py `中，
我们看到的参数是用于控制 GPT 模型训练的各种超参数。理解这些参数的意义，
并通过调整它们来观察生成效果的变化，是深入掌握模型训练的关键。
下面逐一解释常见参数的含义，并给出一些可行的修改建议及其预期影响。

### 基本路径与日志参数

| 参数                     | 含义                                      | 说明与建议                                                   |
| :----------------------- | :---------------------------------------- | :----------------------------------------------------------- |
| `out_dir`                | 输出目录，用于保存模型 checkpoint、日志等 | 例如 `'out-poemtext-char'`，训练过程中生成的模型文件和中间结果将存储在此路径下。 |
| `eval_interval`          | 每多少次迭代在验证集上评估一次损失        | 如 250，频繁评估可实时监控模型，但会减慢训练。可根据总迭代次数调整（例如 `max_iters/10`）。 |
| `eval_iters`             | 评估时使用的验证集迭代次数                | 如 200，取值越大评估越准确，但耗时增加。通常 100~500 之间。  |
| `log_interval`           | 每多少次迭代打印一次训练损失              | 如 10，控制终端输出频率，不影响训练速度，但太频繁会刷屏。    |
| `always_save_checkpoint` | 是否每次评估后都保存 checkpoint           | 一般设为 `False`，仅在验证损失改善时保存，避免频繁写磁盘。调试时可临时开启。 |
| `init_from`              | 初始化模型的方式                          | 可选 `'scratch'`（从头训练）、`'resume'`（从 checkpoint 恢复）、`'gpt2*'`（加载预训练 GPT-2 权重）。 |
| `wandb_log`              | 是否启用 Weights & Biases 日志            | 若设为 `True`，需配置 `wandb_project` 等，便于可视化训练曲线。 |
| `wandb_project`          | W&B 项目名称                              | 如 `'poemtext-char'`，仅当 `wandb_log=True` 时有效。         |
| `wandb_run_name`         | W&B 运行名称                              | 可选，用于区分不同实验。                                     |

### 数据集参数

| 参数                          | 含义                                          | 说明与建议                                                   |
| :---------------------------- | :-------------------------------------------- | :----------------------------------------------------------- |
| `dataset`                     | 数据集名称                                    | 对应 `data/` 下的子目录，如 `'poemtext'`，需包含 `train.bin` 和 `val.bin` 等预处理文件。 |
| `gradient_accumulation_steps` | 梯度累积步数                                  | 用于模拟更大批次，总批次大小 = `batch_size` * `gradient_accumulation_steps`。显存不足时可增大此值。 |
| `batch_size`                  | 每个 GPU 上的批次大小（单次前向传播的样本数） | 受显存限制，通常设为 2 的幂次（如 16, 32, 64）。越大梯度估计越准，但需调整学习率。 |
| `block_size`                  | 模型处理的最大上下文长度（token 数）          | 对于字符级模型，常设为 128~512。值越大，计算量线性增长，但能捕捉更长依赖。 |

### 模型结构参数

| 参数      | 含义                                      | 说明与建议                                                   |
| :-------- | :---------------------------------------- | :----------------------------------------------------------- |
| `n_layer` | Transformer 解码器层数                    | 如 6，增加层数可提升模型容量，但参数量和计算成本也增加。小数据集宜用较少层（如 4~6）。 |
| `n_head`  | 多头注意力头数                            | 如 6，需保证 `n_embd % n_head == 0`。更多头有助于模型关注不同子空间，但过多可能冗余。 |
| `n_embd`  | 嵌入向量维度                              | 如 384，维度越高表示能力越强，但参数量爆炸（与 `n_layer` 乘积关系）。字符级任务常用 256~512。 |
| `dropout` | Dropout 概率                              | 如 0.2，用于防止过拟合。训练集较小时可适当增大（如 0.3），大模型或大数据集可减小（如 0.1）。 |
| `bias`    | 是否在 Linear 和 LayerNorm 层中使用偏置项 | 默认为 `True`，但一些实现（如 GPT-2）中某些层无偏置。关闭可略微减少参数量。 |

### 优化器与学习率参数

| 参数             | 含义                       | 说明与建议                                                   |
| :--------------- | :------------------------- | :----------------------------------------------------------- |
| `learning_rate`  | 初始学习率                 | 如 1e-3，对于小模型可稍高，但需配合预热和衰减。通常范围 1e-4 ~ 1e-3。 |
| `max_iters`      | 总训练迭代步数             | 如 5000，决定了训练时长。可根据损失曲线调整，若损失已平缓可提前停止。 |
| `weight_decay`   | AdamW 优化器的权重衰减系数 | 如 1e-1，用于正则化，防止过拟合。常用值 0.1 ~ 0.01。         |
| `beta1`          | Adam 的一阶矩衰减率        | 默认 0.9，通常保持不动。                                     |
| `beta2`          | Adam 的二阶矩衰减率        | 默认 0.99 或 0.95（小批量时可增大）。                        |
| `grad_clip`      | 梯度裁剪的最大范数         | 如 1.0，防止梯度爆炸，稳定训练。设为 0 或 `None` 表示不裁剪。 |
| `decay_lr`       | 是否使用学习率衰减         | 布尔值，通常 `True`。配合 `warmup_iters`, `lr_decay_iters`, `min_lr` 使用。 |
| `warmup_iters`   | 学习率预热步数             | 如 100，在前 `warmup_iters` 步内将学习率从 0 线性增加到 `learning_rate`，有助于稳定初始训练。 |
| `lr_decay_iters` | 学习率衰减的总步数         | 通常等于 `max_iters`，使得学习率在训练结束时降到 `min_lr`。若小于 `max_iters`，则之后学习率保持 `min_lr`。 |
| `min_lr`         | 最小学习率                 | 如 1e-4，通常为 `learning_rate/10`，衰减终止时的学习率。     |

### 系统与硬件参数

| 参数      | 含义                          | 说明与建议                                                   |
| :-------- | :---------------------------- | :----------------------------------------------------------- |
| `backend` | 分布式后端                    | 如 `'nccl'`（GPU）、`'gloo'`（CPU）。单卡训练无需关心。      |
| `device`  | 训练设备                      | 自动检测，可设为 `'cuda'`、`'cpu'` 或 `'mps'`（Apple Silicon）。 |
| `dtype`   | 数据类型                      | 如 `'float16'`、`'bfloat16'`、`'float32'`。混合精度可加速并节省显存，但需硬件支持。 |
| `compile` | 是否使用 PyTorch 2.0 编译模型 | 布尔值。开启可提速 20~30%，但首次运行有编译开销，且可能与某些操作不兼容。 |


## 3模型采样文件 `sample.py` 的推理过程

`sample.py` 脚本主要负责加载训练好的 GPT 模型权重，并根据给定的起始文本（Prompt）进行自回归生成。其核心推理流程如下：

### 1. 加载模型与编码器

*   **加载 Checkpoint**：从指定的输出目录（`out_dir`）读取训练保存的模型文件（如 `ckpt.pt`），恢复模型结构和权重。
*   **初始化编解码器**：加载对应的元数据（`meta.pkl`），初始化字符级的编码器（`encode`）和解码器（`decode`）。编码器将字符映射为整数索引，解码器则将索引还原为可读字符。

### 2. 设置生成参数

常见的采样超参数包括：

*   **`start`**：起始提示文本。若为空，通常从换行符 `\n` 或特定开始符启动。
*   **`max_new_tokens`**：限制生成的最大字符数量。
*   **`temperature`**：控制采样的随机性。
    *   $T > 1$：概率分布更平滑，生成结果更多样但也更不可控。
    *   $T < 1$：概率分布更尖锐，倾向于选择高概率字符，生成更确定。
    *   $T \to 0$：退化为贪心解码（Greedy Decoding），每次只选概率最大的词。
*   **`top_k`**：仅从概率最高的 $k$ 个候选字符中采样，过滤低概率词，提升生成质量。
*   **`top_p` (Nucleus Sampling)**：在累积概率达到 $p$ 的最小候选集内采样，可与 `top_k` 结合使用。

### 3. 推理循环（逐字符生成）

模型采用**自回归（Auto-regressive）**方式工作：每次预测一个字符，将其追加到输入序列末尾，作为下一次预测的上下文。具体步骤如下：

1.  **编码输入**：将起始文本通过编码器转换为 token 序列 $x$，形状通常为 `(1, T)`，其中 $T$ 为当前序列长度。
2.  **生成循环**：重复执行直到达到 `max_new_tokens` 或遇到停止符：
    *   **上下文截断**：若当前序列长度超过模型的 `block_size`，需截断最左侧部分，仅保留最近的 `block_size` 个 token（`idx_cond = idx[:, -block_size:]`）。
    *   **前向传播**：将处理后的序列输入模型，得到 logits，形状为 `(1, T, vocab_size)`。
    *   **提取最后一步**：仅取最后一个时间步的预测结果 `logits = logits[:, -1, :]`，并除以温度系数 `temperature`。
    *   **采样过滤**（可选）：
        *   若启用 `top_k`，将非前 $k$ 大的 logits 置为负无穷。
        *   若启用 `top_p`，对概率分布进行核采样过滤。
    *   **概率转换**：应用 `softmax` 函数将 logits 转换为概率分布。
    *   **随机采样**：根据概率分布随机抽取下一个 token 索引 `next_token`（使用 `torch.multinomial`）。
    *   **序列拼接**：将 `next_token` 拼接到原序列 $x$ 后方，形成新的输入序列。
3.  **解码输出**：循环结束后，将最终生成的完整 token 序列通过解码器还原为文本并打印或保存。

### 4. 关键实现细节

*   **无梯度计算**：推理过程包裹在 `torch.no_grad()` 上下文中，禁止计算梯度以节省显存并加速推理。
*   **上下文窗口限制**：必须严格遵守 `block_size` 限制，否则会导致模型报错或预测错误。
*   **常用采样策略组合**：
    *   `temperature=1.0, top_k=0`：纯随机采样，多样性高但可能逻辑混乱。
    *   `temperature=0.8, top_k=40`：常用平衡组合，兼顾连贯性与创造性。
    *   `temperature=0.0` (或极小值)：贪心策略，结果最确定但容易陷入重复循环。

### 5. 核心代码逻辑示例

以下是简化后的生成函数核心逻辑：

```
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. 截断输入以适应 block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # 2. 前向传播获取 logits
        logits, _ = model(idx_cond)
        
        # 3. 提取最后一个时间步并应用温度缩放
        logits = logits[:, -1, :] / temperature
        
        # 4. Top-K 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 5. 转换为概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 6. 从分布中采样下一个 token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 7. 拼接序列
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx
```

### 6. 影响生成效果的因素

*   **温度（Temperature）**：过低导致文本重复、死板；过高导致胡言乱语、语法错误。
*   **Top-K / Top-P**：设置过小会限制模型的创造力，设置过大则可能引入噪声和不相关字符。
*   **起始文本（Prompt）**：高质量的 Prompt 能有效引导模型进入特定的主题或风格。
*   **模型能力**：模型容量（参数量）和训练充分度是基础。若模型欠拟合或容量过小，即使调整采样参数也难以生成流畅文本。
    通过灵活调整 `sample.py` 中的这些参数，用户可以探索模型生成能力的边界，针对创意写作、代码生成或事实性问答等不同场景优化输出效果。