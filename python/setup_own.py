# setup.py

import os
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext
# ------------------------------------------------------------
# 1. List all modules (package paths) that should be compiled
#    into shared-object (.so) files. Each tuple contains:
#      - The module’s import path (with dots)
#      - The corresponding source .py file path (with slashes)
# ------------------------------------------------------------
to_compile = [
    ("sglang.srt.models.qwen2_classification", "sglang/srt/models/qwen2_classification.py"),
    ("sglang.srt.models.internvl",              "sglang/srt/models/internvl.py"),
    ("sglang.srt.server_args",                  "sglang/srt/server_args.py"),
    ("sglang.srt.layers.token_pruning.utils",        "sglang/srt/layers/token_pruning/utils.py"),
    ("sglang.srt.layers.token_pruning.visionzip",     "sglang/srt/layers/token_pruning/visionzip.py"),
    ("sglang.srt.layers.token_pruning.patch_pruning", "sglang/srt/layers/token_pruning/patch_pruning.py"),
    ("sglang.srt.layers.token_pruning.fast_patch_pruning", "sglang/srt/layers/token_pruning/fast_patch_pruning.py"),
    # ("sglang.srt.layers.multimodal",             "sglang/srt/layers/multimodal.py"),
    ("sglang.srt.managers.multimodal_processors.internvl", "sglang/srt/managers/multimodal_processors/internvl.py"),
    ("sglang.srt.configs.model_config",          "sglang/srt/configs/model_config.py"),
]

# ------------------------------------------------------------
# 2. Create a list of Extension() objects, one per module to compile.
#    Each Extension takes:
#      - The fully qualified module name (e.g. "sglang.srt.models.qwen2_classification")
#      - A list containing the path to its .py source file.
# ------------------------------------------------------------
extensions = [
    Extension(module_name, [source_path])
    for module_name, source_path in to_compile
]

class build_ext(_build_ext):
    def get_ext_filename(self, ext_name):
        # Let setuptools compute the normal filename:
        full_path = super().get_ext_filename(ext_name)
        # If it ends with “.so”, strip off the ABI/tag portion:
        if full_path.endswith(".so"):
            # `ext_name` is something like "sglang.srt.models.internvl"
            module_basename = ext_name.split(".")[-1] + ".so"
            # Get the directory part of full_path:
            directory = os.path.dirname(full_path)
            return os.path.join(directory, module_basename)
        return full_path


setup(
    name="sglang",
    version="0.4.6.dev5",
    python_requires=">=3.10,<3.11",
    packages=find_packages(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},  # Use Python 3 syntax
    ),
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    zip_safe=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# 
# python3 setup_own.py bdist_wheel



