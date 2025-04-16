import os
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)

"""
radix_sort (keys (int64), values (int32), n_bits) -> sorted_keys, sorted_values
NOTE: the input may be modified and the output may be the same object as the input.
"""
radix_sort = load(
    name="radix_sort",
    sources=[os.path.join(module_path, "radix_sort.cu")],
).radix_sort
