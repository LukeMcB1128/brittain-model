"""Does this vLLM build still ship bitsandbytes serving?

If it does not, the 'train nf4 / serve nf4' option is off the table and the
train/serve quantization question has to be answered a different way.
"""
import importlib.util

for mod in (
    "vllm.model_executor.layers.quantization.bitsandbytes",
    "vllm.model_executor.layers.quantization.awq",
    "vllm.model_executor.layers.quantization.gptq",
    "vllm.model_executor.layers.quantization.compressed_tensors",
):
    print("%-62s %s" % (mod, importlib.util.find_spec(mod) is not None))

print()
from vllm.model_executor.layers.quantization import QuantizationMethods, get_quantization_config
import typing

print("declared methods:", sorted(typing.get_args(QuantizationMethods)))
print()
for name in ("bitsandbytes", "awq_marlin", "gptq_marlin", "compressed-tensors"):
    try:
        get_quantization_config(name)
        print("%-20s resolvable: True" % name)
    except Exception as e:  # noqa: BLE001
        print("%-20s resolvable: False  (%s)" % (name, type(e).__name__))
