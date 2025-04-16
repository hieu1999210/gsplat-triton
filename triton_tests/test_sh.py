import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("sh_degree", list(range(5)))
def test_sh(sh_degree: int):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.triton_impl._wrapper import spherical_harmonics

    torch.manual_seed(43)
    device = "cuda"
    N = (129, 31, 200)
    max_deg = 4
    C = 3
    coeffs = torch.randn(*N, (max_deg + 1) ** 2, C, device=device)
    dirs = torch.randn(*N, C, device=device)
    coeffs.requires_grad = True
    dirs.requires_grad = True
    # masks = torch.randn(*N, device=device) > 0

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    # colors[~masks] = 0
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    v_coeffs, v_dirs = torch.autograd.grad(
        (colors * v_colors).sum(), (coeffs, dirs), allow_unused=True
    )
    _v_coeffs, _v_dirs = torch.autograd.grad(
        (_colors * v_colors).sum(), (coeffs, dirs), allow_unused=True
    )
    torch.testing.assert_close(v_coeffs, _v_coeffs, rtol=1e-4, atol=1e-4)
    if sh_degree > 0:
        torch.testing.assert_close(v_dirs, _v_dirs, rtol=1e-4, atol=1e-4)
