# ─────────────────────────────────────────────────────────────────────────────
# File: setup.py
# ─────────────────────────────────────────────────────────────────────────────

from setuptools import setup, find_packages
from Cython.Build import cythonize
import os
import sysconfig
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.build_ext import build_ext as _build_ext
import importlib.machinery

# ----------------------------------------------------------------------
# 1. collect_modules: walk through base_dir and collect all .py files
#    (excluding __init__.py), converting them to dotted module names.
#    Skip anything under the "test" subdirectory entirely.
# ----------------------------------------------------------------------
def collect_modules(base_dir: str):
    modules = []
    for root, dirs, files in os.walk(base_dir):
        # If we're inside a test directory, skip it altogether
        if os.path.relpath(root, base_dir).startswith("test"):
            continue

        for file in files:
            if not file.endswith(".py") or file == "__init__.py":
                continue
            module_path = os.path.join(root, file)
            # Remove only the trailing .py extension
            no_ext, _ = os.path.splitext(module_path)
            # Convert file path to dotted module name
            module_name = no_ext.replace(os.sep, ".")
            modules.append(module_name)
    return modules

# ----------------------------------------------------------------------
# 2. Generate all candidate .py file paths from the collected module names
# ----------------------------------------------------------------------
base_dir = "sglang"
all_exts = [
    module.replace(".", os.sep) + ".py"
    for module in collect_modules(base_dir)
]

# ----------------------------------------------------------------------
# 3. Define a set of paths that should NOT be compiled into .so.
#    All files in tests are effectively excluded by collect_modules already,
#    but keep this as a safety net for any extra test files.
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 4. Build the list of extensions to compile via Cython, 
#    excluding those in exclude_list.
# ----------------------------------------------------------------------
extensions = [
    ext
    for ext in all_exts
    if ext not in exclude_list
]

print("=== Extensions to compile (Cython) ===")
for e in extensions:
    print("   ", e)
print("======================================")

# ----------------------------------------------------------------------
# 5. CustomBuildPy: override find_package_modules so that:
#    - if a .py file has already been compiled into a .so, skip copying the .py.
#    - __init__.py is always preserved.
#    - skip any .c files entirely (we only want the .so).
# ----------------------------------------------------------------------
class CustomBuildPy(_build_py):
    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []

        for root, _, files in os.walk(package_dir):
            # Skip anything under "test" directories
            if os.path.relpath(root, package_dir).startswith("test"):
                continue

            for file in files:
                # Skip .c files completely
                if file.endswith(".c"):
                    continue

                if not file.endswith(".py"):
                    continue

                module_path = os.path.join(root, file)
                module_name = os.path.relpath(module_path, package_dir).replace(".py", "")

                # Always include __init__.py
                if file == "__init__.py":
                    filtered_modules.append((package, module_name, module_path))
                    continue

                # If a corresponding .so exists, skip copying the .py
                compiled_path = os.path.splitext(module_path)[0] + ext_suffix
                if os.path.exists(compiled_path):
                    continue

                # Otherwise, keep this .py module
                filtered_modules.append((package, module_name, module_path))

        return filtered_modules

# ----------------------------------------------------------------------
# 6. CustomBuildExt: override get_ext_filename to remove the
#    ".cpython-310-...-x86_64-linux-gnu.so" suffix, leaving just ".so".
# ----------------------------------------------------------------------
class CustomBuildExt(_build_ext):
    def get_ext_filename(self, fullname):
        # fullname might be "sglang.api"
        fname = super().get_ext_filename(fullname)
        # Example: "sglang/api.cpython-310-x86_64-linux-gnu.so"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            if fname.endswith(suffix):
                # Strip off the long suffix and append ".so"
                base = fname[: -len(suffix)]
                return base + ".so"
        return fname

# ----------------------------------------------------------------------
# 7. Call setup()
#    - Exclude the "sglang.test" package so nothing under test/
#      is packaged in the wheel.
# ----------------------------------------------------------------------
setup(
    name="sglang",
    version="0.4.6.post5",
    python_requires=">=3.10,<3.11",       # Only support Python 3.10
    # Exclude all test packages
    packages=find_packages(exclude=["sglang.test", "sglang.test.*"]),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        build_dir="build/cython",          # Place all generated .c files under build/cython/
    ),
    zip_safe=False,
    cmdclass={
        "build_py": CustomBuildPy,
        "build_ext": CustomBuildExt,
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# 
# python3 setup.py bdist_wheel

