import math
import os

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


# @pytest.mark.parametrize("channels", [4])
# @pytest.mark.parametrize("calc_compensations", [True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [4, 8])
@pytest.mark.parametrize("calc_compensations", [True, False])
def test_rasterize_to_pixels(test_data, channels: int, calc_compensations):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.triton_impl._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels,
    )

    torch.manual_seed(50)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    C = len(Ks)
    colors = torch.rand(C, len(means), channels, device=device)
    colors[:, :, :3] = test_data["colors"]
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)
    # backgrounds = None

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means,
        None,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        calc_compensations=calc_compensations,
    )
    opacities = opacities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)
    assert not torch.isnan(means2d).any()
    assert not torch.isnan(conics).any()
    assert not torch.isnan(colors).any()
    assert not torch.isnan(opacities).any()
    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # torch
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
    )

    render_colors, render_alphas = rasterize_to_pixels(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        width,
        height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        backgrounds,
    )

    torch.testing.assert_close(_render_alphas, render_alphas)
    torch.testing.assert_close(_render_colors, render_colors)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )

    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )

    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=5e-3, atol=5e-3)
