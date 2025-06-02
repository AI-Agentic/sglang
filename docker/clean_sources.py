#!/usr/bin/env python3


# python3 clean_sources.py "$(python3 -c "import os, sglang; print(os.path.dirname(sglang.__file__))")"
"""
Usage:
    python3 clean_sources.py <folder_name>

This script deletes specific .py and .c files under the given base folder.
The base directory is taken from the first command-line argument.
"""

import os
import sys

# List of .py files to remove (paths relative to BASE_DIR)
py_files = [
    "srt/models/qwen2_classification.py",
    "srt/models/internvl.py",
    "srt/server_args.py",
    "srt/layers/token_pruning/utils.py",
    "srt/layers/token_pruning/visionzip.py",
    "srt/layers/token_pruning/patch_pruning.py",
    "srt/layers/token_pruning/fast_patch_pruning.py",
    # "srt/layers/multimodal.py", 
    "srt/managers/multimodal_processors/internvl.py",
    "srt/configs/model_config.py",
]

# List of .c files to remove (paths relative to BASE_DIR)
c_files = [
    "srt/models/qwen2_classification.c",
    "srt/models/internvl.c",
    "srt/server_args.c",
    "srt/layers/token_pruning/utils.c",
    "srt/layers/token_pruning/visionzip.c",
    "srt/layers/token_pruning/patch_pruning.c",
    "srt/layers/token_pruning/fast_patch_pruning.c",
    # "srt/layers/multimodal.c",
    "srt/managers/multimodal_processors/internvl.c",
    "srt/configs/model_config.c",
]

def remove_if_exists(base_dir, relative_path):
    """
    Given a base directory and a relative path, delete the file if it exists.
    """
    full_path = os.path.join(base_dir, relative_path)
    if os.path.isfile(full_path):
        os.remove(full_path)
        print(f"Removed: {full_path}")
    else:
        print(f"Not found (skipped): {full_path}")

def main():
    # Ensure a folder name was provided
    if len(sys.argv) != 2:
        print("Usage: python3 clean_sources.py <folder_name>")
        sys.exit(1)

    base_dir = sys.argv[1]

    # Verify that the provided folder exists and is a directory
    if not os.path.isdir(base_dir):
        print(f"Error: '{base_dir}' is not a valid directory.")
        sys.exit(1)

    # Remove all .py files
    print(f"Removing .py files under '{base_dir}':")
    for rel in py_files:
        remove_if_exists(base_dir, rel)

    # Remove all .c files
    print(f"\nRemoving .c files under '{base_dir}':")
    for rel in c_files:
        remove_if_exists(base_dir, rel)

if __name__ == "__main__":
    main()
