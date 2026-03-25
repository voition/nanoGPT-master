from modelscope import snapshot_download

model_dir = snapshot_download(
    'qwen/Qwen3-0.6B-Instruct',  # 改为 0.6B
    cache_dir='D:\\Qwen\\Qwen3-main'  # 路径建议用双反斜杠或正斜杠
)