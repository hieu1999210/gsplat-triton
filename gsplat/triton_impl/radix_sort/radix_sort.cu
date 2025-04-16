#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <vector>

namespace extension_cpp {

std::tuple<torch::Tensor, torch::Tensor>
radix_sort(const at::Tensor keys, const at::Tensor values, const int n_bits) {
    TORCH_CHECK(keys.sizes() == values.sizes());

    TORCH_CHECK(keys.dtype() == torch::kInt64);
    TORCH_CHECK(values.dtype() == torch::kInt32);

    TORCH_CHECK(keys.is_contiguous());
    TORCH_CHECK(values.is_contiguous());

    TORCH_INTERNAL_ASSERT(keys.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);

    at::Tensor keys_sorted = torch::empty_like(keys);
    at::Tensor values_sorted = torch::empty_like(values);

    cub::DoubleBuffer<int64_t> d_keys(
        keys.data_ptr<int64_t>(), keys_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        values.data_ptr<int32_t>(), values_sorted.data_ptr<int32_t>()
    );

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    auto length = keys.sizes()[0];

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_keys, d_values, length
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_keys, d_values, length, 0, n_bits
    );
    cudaFree(d_temp_storage);

    switch (d_keys.selector) {
    case 0:
        keys_sorted = keys;
        break;
    case 1:
        break;
    }
    switch (d_values.selector) {
    case 0:
        values_sorted = values;
        break;
    case 1:
        break;
    }

    return std::make_tuple(keys_sorted, values_sorted);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("radix_sort", &radix_sort); }
} // namespace extension_cpp
