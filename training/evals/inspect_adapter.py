"""Show the keys actually written to the adapter file.

Passing an explicit state_dict to save_pretrained can skip PEFT's usual key
rewriting, and a malformed key would make a vLLM load failure look like an
architecture mismatch when it is really a packaging bug. Separate the two
before drawing any conclusion.
"""
import sys

from safetensors import safe_open

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/lukeb/brittain4/adapters/dummy-r8/adapter_model.safetensors"

with safe_open(path, framework="pt") as f:
    keys = sorted(f.keys())
    print("tensors:", len(keys))
    for k in keys[:6]:
        print("   ", k, tuple(f.get_slice(k).get_shape()))
    print("...")
    print("contains '.default.':", any(".default." in k for k in keys))
