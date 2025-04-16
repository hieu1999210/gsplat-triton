import torch
from einops import repeat


def rand_sym_mat(B, D, device, dtype, eps=1e-3):
    M = torch.rand(B, D, D, device=device, dtype=dtype)
    I = repeat(torch.eye(D, device=device, dtype=dtype), "... -> b ...", b=B)

    M = M @ M.transpose(-1, -2) + (eps * I)
    return M


def check_all_close(actual, expected, atol=None, rtol=None):
    THRESHOLD = {
        torch.float16: (1e-3, 1e-5),
        torch.bfloat16: (1.6e-2, 1e-5),
        torch.float32: (1.3e-6, 1e-5),
        torch.float64: (1e-7, 1e-7),
    }
    assert (
        actual.device == expected.device
    ), f"a device {actual.device} != b device {expected.device}"
    assert (
        actual.shape == expected.shape
    ), f"a shape {actual.shape} != b shape {expected.shape}"
    if atol is None:
        if actual.dtype in THRESHOLD:
            rtol, atol = THRESHOLD[actual.dtype]
        else:
            rtol, atol = 0, 0
    nan_masks = torch.isnan(actual) | torch.isnan(expected)
    actual = actual[~nan_masks]
    expected = expected[~nan_masks]
    diff = torch.abs(actual - expected)
    failed = diff > atol + rtol * torch.abs(expected)
    if failed.any() or nan_masks.any():
        n = failed.sum() + nan_masks.sum()
        max_abs = diff.max()
        max_rel = ((diff[failed] - atol) / expected[failed]).max()
        print(f"\tNaNs: {nan_masks.sum()} / {nan_masks.numel()}")
        print(f"\tMismatched {n} / {failed.numel()} ({n/failed.numel()*100:.1f}%)")
        print(f"\tMax abs diff: {max_abs} (atol={atol})")
        print(f"\tMax rel diff: {max_rel} (rtol={rtol})")
        return False
    print("\tAll close")
    return True
