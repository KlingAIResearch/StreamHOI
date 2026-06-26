#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

python3 - <<'EOF'
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi

token = "YOUR_HF_TOKEN_HERE"
api = HfApi(token=token, endpoint="https://hf-mirror.com")
repo_id = "zjrao/StreamHOI"

print("==> Creating repo if not exists...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
print("==> Repo ready!")

print("==> Uploading code...")
api.upload_folder(
    folder_path="/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22",
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="code",
    ignore_patterns=[
        "logs/**",
        "StreamingCode/**",
        "videos/**",
        "vis/**",
        "wan_models/**",
        "wandb/**",
        "**/__pycache__/**",
        "**/*.pyc",
    ],
)
print("==> Code uploaded!")

print("==> Uploading conda env...")
api.upload_folder(
    folder_path="/m2v_intern/raozejing/StreamingCode/opt/conda/envs/longlive2",
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="conda_env/longlive2",
)
print("==> Conda env uploaded!")

print("==> Uploading miniconda.sh...")
api.upload_file(
    path_or_fileobj="/m2v_intern/raozejing/StreamingCode/miniconda.sh",
    path_in_repo="miniconda.sh",
    repo_id=repo_id,
    repo_type="model",
)
print("==> miniconda.sh uploaded!")

print("All done!")
EOF
