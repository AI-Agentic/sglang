# ─────────────────────────────────────────────────────────────────────────────
# File: setup.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sysconfig
from setuptools import setup, find_packages, Extension, find_namespace_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import importlib.machinery

# ----------------------------------------------------------------------
# 1. Explicitly list any exact file paths (relative to project root)
#    that we want to exclude from Cython compilation.
# ----------------------------------------------------------------------
base_dir = "sglang"

exclude_list = {
    os.path.join(base_dir, "srt", "distributed", "parallel_state.py"),
    os.path.join(base_dir, "srt", "distributed", "device_communicators", "pynccl_wrapper.py"),
    os.path.join(base_dir, "srt", "layers", "quantization", "compressed_tensors", "compressed_tensors.py"),
    os.path.join(base_dir, "srt", "layers", "quantization", "compressed_tensors", "compressed_tensors_moe.py"),
    os.path.join(base_dir, "srt", "layers", "quantization", "utils.py"),
    os.path.join(base_dir, "srt", "managers", "expert_distribution.py"),
    os.path.join(base_dir, "srt", "model_loader", "weight_utils.py"),
    os.path.join(base_dir, "srt", "speculative", "build_eagle_tree.py"),
    os.path.join(base_dir, "srt", "utils.py"),
    os.path.join(base_dir, "test", "test_utils.py")
}


extensions = []      # Will hold Cython Extension objects for modules to compile
raw_py_files = []

for root, _, files in os.walk(base_dir):  # Recurse through all subdirs :contentReference[oaicite:2]{index=2}
    for fname in files:
        if not fname.endswith(".py"):
            continue  # Skip non-Python files :contentReference[oaicite:3]{index=3}

        full_path = os.path.join(root, fname)

        if fname == "__init__.py":
            raw_py_files.append(full_path)
            continue  # Always copy __init__.py as source, never compile :contentReference[oaicite:4]{index=4}

        if full_path in exclude_list:
            raw_py_files.append(full_path)
            continue  # Copy excluded files as source, never compile :contentReference[oaicite:5]{index=5}

        # Otherwise, register this .py as a Cython extension to compile to .so
        rel_path = os.path.relpath(full_path, base_dir)
        module_name = rel_path[:-3].replace(os.path.sep, ".")  # e.g. "sglang.subpkg.module" :contentReference[oaicite:6]{index=6}

        ext = Extension(
            name=module_name,
            sources=[full_path],
        )
        extensions.append(ext) 


ext_modules = cythonize(
    extensions,
    compiler_directives={"language_level": "3"},  # Enforce Python 3 syntax :contentReference[oaicite:8]{index=8}
)

package_data = {}
for full_path in raw_py_files:
    rel_path = os.path.relpath(full_path, base_dir)  # e.g. "sglang/srt/utils.py"
    dir_name, fname = os.path.split(rel_path)  # dir_name="sglang/srt", fname="utils.py"
    pkg_name = dir_name.replace(os.path.sep, ".")  # e.g. "sglang.srt" :contentReference[oaicite:9]{index=9}

    if pkg_name not in package_data:
        package_data[pkg_name] = []
    package_data[pkg_name].append(fname)  # Only store the filename, not full path :contentReference[oaicite:10]{index=10}


# ----------------------------------------------------------------------
# 6. Call setup(), using cythonize() on our Extension list,
#    and hooking in our custom commands.
# ----------------------------------------------------------------------
setup(
    name="sglang",
    version="0.4.6.post5",
    python_requires=">=3.10,<3.11",
    packages=find_namespace_packages(include=["sglang", "sglang.*"]),
    ext_modules=ext_modules,
    include_package_data=True,
    package_data=package_data, 
    exclude_package_data={ '': ['*.py'] }, 
    zip_safe=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 
# python3 setup.py bdist_wheel

