import os
import typing as t
from huggingface_hub import snapshot_download

def get_weight_dir(weight_dir: str, ignore_patterns: t.Optional[t.List[str]] = None, local_only: bool = False):
    if os.path.isdir(weight_dir):
        return weight_dir

    if local_only:
        raise FileNotFoundError(f"{weight_dir} does not exist or is not a directory")

    # assuming weight_dir is a huggingface model tag
    weight_dir = snapshot_download(weight_dir, ignore_patterns=ignore_patterns)
    return weight_dir
