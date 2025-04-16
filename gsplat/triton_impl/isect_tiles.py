import math
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int32, Int64
from torch import Tensor

from gsplat.triton_impl.radix_sort import radix_sort


@torch.no_grad()
def isect_tiles(
    means2d: Union[Float[Tensor, "C N 2"], Float[Tensor, "nnz 2"]],
    radii: Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
    depths: Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    block_size: int = 256,
) -> Tuple[
    Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
    Int64[Tensor, "n_insects"],
    Int32[Tensor, "n_insects"],
]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        n_gaussians = nnz
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz,), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras
        N = 0  # dummy value

    else:
        C, N, _ = means2d.shape
        n_gaussians = C * N
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N), radii.size()
        assert depths.shape == (C, N), depths.size()
        # dummy values
        camera_ids = torch.empty(1, device=device, dtype=torch.int32)
    means2d = means2d.contiguous()
    radii = radii.contiguous()
    depths = depths.contiguous()

    # step 1 get tile per gauss, offsets(cumsum of tiles per gauss)
    n_blocks = int(math.ceil(n_gaussians / block_size))
    # the maximum number of tiles intersected by a gaussian in one side
    tiles_per_gauss = torch.zeros(n_gaussians, dtype=torch.int32, device=device)
    get_tile_per_gauss_kernel[n_blocks,](
        tile_size,
        tile_width,
        tile_height,
        n_gaussians,
        means2d,
        radii,
        tiles_per_gauss,
        block_size,
    )

    # step 2 write isec_id (tile_id and depth) and flatten_ids (gaussian id)
    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss, dim=0, dtype=torch.int64)
    n_isects = cum_tiles_per_gauss[-1].item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)
    n_bit_tile_id = (tile_height * tile_width - 1).bit_length()
    n_bit_cam_id = (C - 1).bit_length()
    assert n_bit_tile_id + n_bit_cam_id <= 32, "tile_id and cam_id exceed 32 bits"
    get_isect_ids_kernel[n_gaussians, 1, 1](
        packed,
        tile_size,
        tile_width,
        tile_height,
        N,
        camera_ids,
        means2d,
        radii,
        depths,
        cum_tiles_per_gauss,
        flatten_ids,
        isect_ids,
        n_bit_tile_id,
    )

    if sort:
        isect_ids, flatten_ids = radix_sort(
            isect_ids, flatten_ids, 32 + n_bit_tile_id + n_bit_cam_id
        )

    if not packed:
        tiles_per_gauss = tiles_per_gauss.view(C, N)
    return tiles_per_gauss, isect_ids, flatten_ids


@triton.heuristics({"num_warps": lambda args: max(1, args["BLOCK_SIZE"] // 32)})
@triton.jit
def get_tile_per_gauss_kernel(
    tile_size,  # int32
    tile_width,  # int32
    tile_height,  # int32
    n_gaussian,  # int32 C*N or nnz
    means2D_ptr,  # float32 [C N 2] or [nnz, 2]
    radii_ptr,  # int32 [C N] or [nnz]
    tiles_per_gauss_ptr,  # int32 [C N] or [nnz]
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calculate the number of tiles intersected by each Gaussian.

    args:
        tile_size (int32): in pixel
        tile_width (int32): image width in #tiles
        tile_height (int32): image height in #tiles
        n_gaussian (int32): number of Gaussians C*N or nnz
        means2D (float32 [C N 2] or [nnz, 2])
        radii (int32 [C N] or [nnz])

    returns
        tiles_per_gauss (int32 [C N] or [nnz]) number of tiles intersected by each Gaussian
        max_n_tile_per_side (int32) maximum number of tiles intersected by a Gaussian in one side
    """
    ptr_offsets = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
    masks = ptr_offsets < n_gaussian
    means2D_x = tl.load(means2D_ptr + ptr_offsets * 2, mask=masks, other=0.0)
    means2D_y = tl.load(means2D_ptr + ptr_offsets * 2 + 1, mask=masks, other=0.0)
    radii = tl.load(radii_ptr + ptr_offsets, mask=masks, other=0.0)

    tile_min_x, tile_max_x, tile_min_y, tile_max_y = get_isec_tile(
        means2D_x, means2D_y, radii, tile_size, tile_width, tile_height
    )
    masks &= radii > 0
    n_tiles = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y)
    tl.store(tiles_per_gauss_ptr + ptr_offsets, n_tiles, mask=masks)


# hacking to pass num warps to triton
@triton.heuristics(values={"num_warps": lambda args: 1})
@triton.jit
def get_isect_ids_kernel(
    packed: tl.constexpr,  # bool
    tile_size,  # int32
    tile_width,  # int32
    tile_height,  # int32
    N,  # int32 number of gaussians per camera if packed==False
    camera_ids_ptr,  # int64 [nnz]
    means2D_ptr,  # float32 [C N 2] or [nnz, 2]
    radii_ptr,  # int32 [C N] or [nnz]
    depths_ptr,  # float32 [C N] or [nnz]
    cum_tiles_per_gauss_ptr,  # int64 [C*N] or [nnz]
    flatten_ids_ptr,  # int32 [n_isects]
    isect_ids_ptr,  # int64 [n_isects]
    tile_n_bits,  # int32
):
    """
    compute intersection ids (isect_ids) and corresponding gaussian ids (flatten_ids)
        isect_id: |camera_id|tile_id|depth (n_cam_bits|n_tile_bit|32)
                    cam_id in [0, n_cameras)
                    tile_id in [0, tile_width*tile_height)
                    depth in [0, 2^32)
        flatten_ids: gaussian ids in range [0, n_gaussian)

    args:
        packed (bool): if True, gaussian are packed in [nnz] else [C N]
        tile_size (int32): in pixel
        tile_width (int32): image width in #tiles
        tile_height (int32): image height in #tiles
        n_gaussian (int32): number of Gaussians C*N or nnz
        N (int32): number of gaussians per camera if packed==False
        camera_ids (int64 [nnz]): camera ids if packed==True
        means2D (float32 [C N 2] or [nnz, 2])
        radii (int32 [C N] or [nnz])
        depths (float32 [C N] or [nnz])
        cum_tiles_per_gauss (int64 [C*N] or [nnz])
        tile_n_bits (int32): number of bits to represent tile_id
        n_isect_tile_side (int32): max number of tiles intersected by a gaussian in one side
    return
        isect_ids (int64 [n_isects])
        flatten_ids (int32 [n_isects])
    """
    idx = tl.program_id(0)

    # load centers and radius of a batch of gaussians
    radii = tl.load(radii_ptr + idx)
    if radii <= 0:
        return
    means2D_x = tl.load(means2D_ptr + idx * 2)
    means2D_y = tl.load(means2D_ptr + idx * 2 + 1)

    # get range of tiles intersected by each gaussian
    tile_min_x, tile_max_x, tile_min_y, tile_max_y = get_isec_tile(
        means2D_x, means2D_y, radii, tile_size, tile_width, tile_height
    )
    # write cam_id to isec_id
    if packed:
        cam_id = tl.load(camera_ids_ptr + idx).cast(tl.int64)
    else:
        cam_id = tl.floor(idx / N).cast(tl.int64)
    cam_id = cam_id << (tile_n_bits + 32)

    depths = tl.load(depths_ptr + idx).cast(tl.int32, bitcast=True).cast(tl.int64)
    if idx == 0:
        cur_idx = tl.cast(0, tl.int64)
    else:
        cur_idx = tl.load(cum_tiles_per_gauss_ptr + idx - 1)

    x_size = tile_max_x - tile_min_x
    for i in tl.range(tile_min_y, tile_max_y):
        for j in tl.range(tile_min_x, tile_max_x):
            tile_id = (i * tile_width + j).cast(tl.int64) << 32
            tile_idx = cur_idx + j - tile_min_x + (i - tile_min_y) * x_size
            isect_id = cam_id | tile_id | depths
            tl.store(isect_ids_ptr + tile_idx, isect_id)
            tl.store(flatten_ids_ptr + tile_idx, idx.cast(tl.int32))


@triton.jit
def get_isec_tile(
    px,  # float32 [B]
    py,  # float32 [B]
    radii,  # int32 [B]
    t_size,  # int32
    t_w,  # int32 tile width
    t_h,  # int32 tile height
):
    """
    return tile_min_x, tile_max_x, tile_min_y, tile_max_y [B]
    min is inclusive, max is exclusive
    """
    tile_min_x = tl.minimum(
        tl.maximum(tl.floor((px - radii) / t_size).cast(tl.int32), 0), t_w
    )
    tile_max_x = tl.minimum(
        tl.maximum(tl.ceil((px + radii) / t_size).cast(tl.int32), 0), t_w
    )

    tile_min_y = tl.minimum(
        tl.maximum(tl.floor((py - radii) / t_size).cast(tl.int32), 0), t_h
    )
    tile_max_y = tl.minimum(
        tl.maximum(tl.ceil((py + radii) / t_size).cast(tl.int32), 0), t_h
    )

    return tile_min_x, tile_max_x, tile_min_y, tile_max_y
