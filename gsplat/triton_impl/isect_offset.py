import torch
import triton
import triton.language as tl
from jaxtyping import Int32
from torch import Tensor


@torch.no_grad()
def get_isect_offsets(
    isect_ids: Int32[Tensor, "n_isects"],
    C: int,
    tile_width: int,
    tile_height: int,
) -> Int32[Tensor, "n_tiles"]:
    """
    get the index of the first intersection for each tile in the isects array
    for example:
        with isect_ids = [1, 1, 1, 3, 3, 5, 5]
        and 7 tiles in total, we have
        isect_offsets = [0, 0, 3, 3, 5, 5, 7]
    """
    n_tiles = tile_height * tile_width
    n_bit_tile = (n_tiles - 1).bit_length()
    n_isects = isect_ids.size(0)
    tiles_offset = torch.zeros(
        C, tile_height, tile_width, dtype=torch.int32, device=isect_ids.device
    )
    isect_ids = isect_ids.contiguous()

    get_isect_offsets_kernel[
        n_isects,
    ](isect_ids, tiles_offset, n_isects, C, n_tiles, n_bit_tile)
    return tiles_offset


#! this is way slower than cuda implementation
# as there is high level of warp convergence in thread based implementation
# while in triton, each program/warp only process one tile
@triton.heuristics(values={"num_warps": lambda args: 1})
@triton.jit
def get_isect_offsets_kernel(
    isect_ids_ptr, tiles_offset_ptr, n_isects, C, n_tiles, n_bit_tile
):
    idx = tl.program_id(0)
    if idx == 0:
        return

    cur_isect_id = tl.load(isect_ids_ptr + idx) >> 32
    cur_cam_id = cur_isect_id >> n_bit_tile
    cur_tile_id = cur_isect_id & ((1 << n_bit_tile) - 1)
    cur_id = cur_cam_id * n_tiles + cur_tile_id

    if idx == n_isects - 1:
        for i in tl.range(cur_id + 1, C * n_tiles):
            tl.store(tiles_offset_ptr + i, n_isects.cast(tl.int32))

    prev_isect_id = tl.load(isect_ids_ptr + idx - 1) >> 32
    if prev_isect_id != cur_isect_id:
        prev_cam_id = prev_isect_id >> n_bit_tile
        prev_tile_id = prev_isect_id & ((1 << n_bit_tile) - 1)
        prev_id = prev_cam_id * n_tiles + prev_tile_id
        for j in tl.range(prev_id + 1, cur_id + 1):
            tl.store(tiles_offset_ptr + j, idx.cast(tl.int32))
