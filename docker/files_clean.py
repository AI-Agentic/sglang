#!/usr/bin/env python3
"""
recursive_cleanup.py

Recursively traverse a directory tree. For each directory:
  1. List all files in that directory.
  2. If you find a file named `xxx.so`, delete `xxx.py` and `xxx.c` (if they exist) in that directory.
"""

import os
import argparse

def cleanup_directory(root_path: str):
    """
    Walk through `root_path` directory tree. For each directory:
      - Scan all filenames inside.
      - If a filename ends with '.so', compute its base name (without extension).
      - Delete corresponding 'base.py' and 'base.c' in that same folder, if present.
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Print the directory being scanned
        print(f"\nScanning folder: {dirpath}")
        print(f"Contents: {filenames}")

        # Find all .so files in this directory
        so_files = [f for f in filenames if f.endswith(".so")]

        for so in so_files:
            base_name, _ = os.path.splitext(so)
            py_file = os.path.join(dirpath, base_name + ".py")
            c_file  = os.path.join(dirpath, base_name + ".c")

            # Attempt to delete the .py file if it exists
            if os.path.isfile(py_file):
                try:
                    os.remove(py_file)
                    print(f"  Deleted: {py_file}")
                except Exception as e:
                    print(f"  [Error] Could not delete {py_file}: {e}")

            # Attempt to delete the .c file if it exists
            if os.path.isfile(c_file):
                try:
                    os.remove(c_file)
                    print(f"  Deleted: {c_file}")
                except Exception as e:
                    print(f"  [Error] Could not delete {c_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively search for *.so files and delete matching *.py and *.c in the same folder."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default=".",
        help="Top-level directory to start the recursive scan (default: current directory)."
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    print(f"Starting recursive cleanup from: {root}\n")
    cleanup_directory(root)
    print("\nCleanup complete.")

