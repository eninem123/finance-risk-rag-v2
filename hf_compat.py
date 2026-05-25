# hf_compat.py → 替换为以下内容
import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import EntryNotFoundError

def cached_download(*args, **kwargs):
    """
    超级兼容旧版 sentence-transformers 的 cached_download
    支持所有调用方式
    """
    repo_id = None
    filename = None
    cache_dir = None
    revision = "main"
    local_files_only = False

    # 方式1: 位置参数 (repo_id, filename)
    if len(args) >= 1:
        repo_id = args[0]
    if len(args) >= 2:
        filename = args[1]

    # 方式2: 关键字参数
    repo_id = kwargs.get("repo_id", repo_id)
    filename = kwargs.get("filename", filename)
    cache_dir = kwargs.get("cache_dir", cache_dir)
    revision = kwargs.get("revision", revision)
    local_files_only = kwargs.get("local_files_only", local_files_only)

    if not repo_id:
        raise ValueError("repo_id is required")

    try:
        return snapshot_download(
            repo_id=repo_id,
            allow_patterns=[filename] if filename else None,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only
        )
    except Exception as e:
        raise EntryNotFoundError(f"Could not download {repo_id}/{filename}: {e}")