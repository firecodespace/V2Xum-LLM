from huggingface_hub import snapshot_download

# Download V2Xum-LLM Stage 2 model
print("Downloading V2Xum-LLM Stage 2 model...")
snapshot_download(
    repo_id="hhua2/V2Xum-LLM",
    local_dir="checkpoints/v2xumllm-vicuna-v1-5-7b-stage2-e2",
    local_dir_use_symlinks=False
)

# Download LLaVA Stage 1 projector
print("Downloading LLaVA Stage 1 projector...")
snapshot_download(
    repo_id="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
    local_dir="checkpoints/llava-vicuna-v1-5-7b-stage1",
    allow_patterns=["mm_projector.bin"],
    local_dir_use_symlinks=False
)

print("All models downloaded successfully!")
