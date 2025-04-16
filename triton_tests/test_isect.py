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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [4, 64])
def test_isect(test_data, channels: int):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.triton_impl._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
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

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, None, quats, scales, viewmats, Ks, width, height
    )
    opacities = opacities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _flatten_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height, sort=True
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, C, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(flatten_ids, _flatten_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)
