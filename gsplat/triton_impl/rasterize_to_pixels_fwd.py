from typing import Optional, Union

import torch
import triton
import triton.language as tl
from jaxtyping import Bool, Float, Int
from torch import Tensor
from triton.language.extra import libdevice

from gsplat.triton_impl.utils import TRITON_MAX_NUMEL, is_power_of_two


@triton.heuristics(
    {
        "num_warps": lambda args: max(
            1, min(2, args["GAUSSIANS_BATCH"]) * args["tile_size"] ** 2 // 32
        )
    }
)
@triton.jit
def rasterize_to_pixels_fwd_kernel(
    C,
    n_gaussians,
    n_isects,
    # gaussian parameter
    means2D_ptr,  # float32 [C, N, 2] or [nnz, 2]
    conics_ptr,  # float32 [C, N, 3] or [nnz, 3]
    colors_ptr,  # float32 [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    opacities_ptr,  # float32 [C, N] or [nnz]
    bg_ptr,  # float32 [C, COLOR_DIM],
    has_background: tl.constexpr,
    masks_ptr,  # bool [C tile_height tile_width],
    has_masks: tl.constexpr,
    # shape
    image_width,
    image_height,
    tile_size: tl.constexpr,
    tile_width,
    tile_height,
    # intersection
    tile_offsets_ptr,  #: int32 [C tile_height tile_width],
    flatten_ids_ptr,  #: int32 [n_insects],
    # outputs
    render_colors_ptr,  #: float32 [C image_height image_width COLOR_DIM],
    render_alphas_ptr,  #: float32 [C image_height image_width 1],
    last_ids_ptr,  #: float32 [C image_height image_width],
    COLOR_DIM: tl.constexpr,
    GAUSSIANS_BATCH: tl.constexpr,
):
    # the program is launched with a 3D grid block [C, tile_height, tile_width]
    tile_id = tl.program_id(1) * tile_width + tl.program_id(2)
    cam_id = tl.program_id(0)

    # check if the tile is masked
    if has_masks:
        mask = tl.load(masks_ptr + cam_id * tile_height * tile_width + tile_id)
        if mask:
            return

    # pixel coordinate in range [0, height] x [0, width]
    px = (tl.arange(0, tile_size) + tl.program_id(2) * tile_size).cast(tl.float32) + 0.5
    py = (tl.arange(0, tile_size) + tl.program_id(1) * tile_size).cast(tl.float32) + 0.5
    done = (py[:, None] >= image_height) | (px[None, :] >= image_width)

    # get gaussian index range
    tile_offset_ptr = tile_offsets_ptr + cam_id * tile_height * tile_width + tile_id
    range_start = tl.load(tile_offset_ptr)
    if cam_id == C - 1 and tile_id == tile_height * tile_width - 1:
        range_end = n_isects
    else:
        range_end = tl.load(tile_offset_ptr + 1)
    num_batches = tl.cdiv(range_end - range_start, GAUSSIANS_BATCH)

    # variables to update in the loop
    c_log_T = tl.zeros((tile_size, tile_size), dtype=tl.float32)
    last_ids = tl.zeros((tile_size, tile_size), dtype=tl.int32)
    render_colors = tl.zeros((tile_size, tile_size, COLOR_DIM), dtype=tl.float32)

    batch_id = 0
    running = True
    while running and batch_id < num_batches:
        # get indices of the gaussian batch
        start = range_start + batch_id * GAUSSIANS_BATCH
        end = min(range_end, start + GAUSSIANS_BATCH)
        gid_offsets = start + tl.arange(0, GAUSSIANS_BATCH)
        g_masks = gid_offsets < end
        g_ids = tl.load(
            flatten_ids_ptr + gid_offsets,
            mask=g_masks,
            other=n_gaussians,
        )

        # load means2D
        means2D_x = tl.load(means2D_ptr + g_ids * 2, g_masks, 0.0)
        means2D_y = tl.load(means2D_ptr + g_ids * 2 + 1, g_masks, 0.0)

        # load conics
        conics_xx = tl.load(conics_ptr + g_ids * 3, g_masks, 0.0)[:, None, None]
        conics_xy = tl.load(conics_ptr + g_ids * 3 + 1, g_masks, 0.0)[:, None, None]
        conics_yy = tl.load(conics_ptr + g_ids * 3 + 2, g_masks, 0.0)[:, None, None]

        # load opacities
        opacs = tl.load(opacities_ptr + g_ids, g_masks, 0.0)

        # compute alpha of the gaussian
        ## get delta
        delta_x = means2D_x[:, None, None] - px[None, None, :]
        delta_y = means2D_y[:, None, None] - py[None, :, None]

        ## get sigma = delta.T @ conics @ delta * 0.5
        sigma = (
            0.5 * (conics_xx * delta_x * delta_x + conics_yy * delta_y * delta_y)
            + conics_xy * delta_x * delta_y
        )
        alpha = tl.clamp(
            opacs[:, None, None] * libdevice.fast_expf(-sigma),
            min=0.0,
            max=0.999,
        )

        # ignore some gaussians with small alpha or negative sigma
        skip = (alpha < 1.0 / 255.0) | (sigma < 0.0) | done[None, :, :]
        alpha = tl.where(skip, 0.0, alpha)

        # cumulate T using log
        log_1_m_alpha = libdevice.fast_logf(1.0 - alpha)
        log_T = c_log_T[None, :, :] + tl.cumsum(log_1_m_alpha, axis=0)

        # stop accumulating once T <= 1e-4 EXCLUSIVE
        small_T = log_T <= -9.21034  # log(1e-4)
        skip |= small_T
        done |= tl.max(small_T, axis=0) > 0

        # color blending
        vis = libdevice.fast_expf(log_T - log_1_m_alpha) * alpha
        vis = tl.where(skip, 0.0, vis)

        color_offsets = g_ids[:, None] * COLOR_DIM + tl.arange(0, COLOR_DIM)[None, :]
        tl.max_contiguous(tl.multiple_of(color_offsets, (COLOR_DIM, 1)), (COLOR_DIM, 1))
        color = tl.load(
            colors_ptr + color_offsets,
            mask=g_masks[:, None],
            other=0.0,
        )
        render_colors += tl.sum(color[:, None, None, :] * vis[:, :, :, None], axis=0)

        # update log_T for the next batch
        log_1_m_alpha = tl.where(skip, 0.0, log_1_m_alpha)
        c_log_T += tl.sum(log_1_m_alpha, axis=0)

        # update the last_ids as the last gaussian in the batch
        keep = 1 - skip.cast(tl.int32)
        has_any = tl.max(keep, axis=0) > 0
        batch_last_ids = tl.argmax(tl.cumsum(keep, axis=0), axis=0)
        last_ids = tl.where(has_any, start + batch_last_ids, last_ids)

        # terminate if all pixels are done
        if tl.sum(done) == tile_size * tile_size:
            running = False
        batch_id += 1

    # blend with background
    render_alphas = 1.0 - libdevice.fast_expf(c_log_T)
    if has_background:
        bg_color = tl.load(bg_ptr + cam_id * COLOR_DIM + tl.arange(0, COLOR_DIM))
        render_colors += (1.0 - render_alphas)[:, :, None] * bg_color[None, None, :]

    # store the result
    pixels_x = tl.arange(0, tile_size) + tl.program_id(2) * tile_size
    pixels_y = tl.arange(0, tile_size) + tl.program_id(1) * tile_size
    tl.max_contiguous(tl.multiple_of(pixels_x, tile_size), tile_size)
    tl.max_contiguous(tl.multiple_of(pixels_y, tile_size), tile_size)

    pixel_mask = (pixels_y[:, None] < image_height) & (pixels_x[None, :] < image_width)
    pixels_offsets = (
        cam_id * image_height * image_width
        + pixels_y[:, None] * image_width
        + pixels_x[None, :]
    )
    tl.store(
        render_colors_ptr
        + pixels_offsets[:, :, None] * COLOR_DIM
        + tl.arange(0, COLOR_DIM),
        mask=pixel_mask[:, :, None],
        value=render_colors,
    )
    tl.store(
        render_alphas_ptr + pixels_offsets,
        mask=pixel_mask,
        value=render_alphas,
    )
    tl.store(
        last_ids_ptr + pixels_offsets,
        mask=pixel_mask,
        value=last_ids,
    )


@torch.no_grad()
def rasterize_to_pixels_fwd(
    means2D: Union[Float[Tensor, "C N 2"], Float[Tensor, "nnz 2"]],
    conics: Union[Float[Tensor, "C N 3"], Float[Tensor, "nnz 3"]],
    colors: Union[Float[Tensor, "C N COLOR_DIM"], Float[Tensor, "nnz COLOR_DIM"]],
    opacities: Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
    backgrounds: Optional[Float[Tensor, "C COLOR_DIM"]],
    masks: Optional[Bool[Tensor, "C tile_height tile_width"]],
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_offsets: Int[Tensor, "C tile_height tile_width"],
    flatten_ids: Int[Tensor, "n_insects"],
    GAUSSIANS_BATCH: int = 8,
):
    assert is_power_of_two(tile_size)
    color_dim = colors.shape[-1]
    assert is_power_of_two(color_dim)
    assert color_dim * tile_size * tile_size * GAUSSIANS_BATCH <= TRITON_MAX_NUMEL
    # GAUSSIANS_BATCH = TRITON_MAX_NUMEL // color_dim // tile_size // tile_size
    device = means2D.device
    float_type = means2D.dtype

    C, tile_height, tile_width = tile_offsets.size()
    n_isecs = flatten_ids.shape[0]
    n_gaussians = means2D.numel() // 2

    means2D = means2D.contiguous()
    conics = conics.contiguous()
    colors = colors.contiguous()
    opacities = opacities.contiguous()
    tile_offsets = tile_offsets.contiguous()

    # create flag and dummy tensors for background
    has_background = backgrounds is not None
    if not has_background:
        backgrounds = torch.empty(1, device=device, dtype=float_type)
    else:
        backgrounds = backgrounds.contiguous()

    # create flag and dummy tensors for masks
    has_masks = masks is not None
    if not has_masks:
        masks = torch.empty(1, device=device, dtype=torch.bool)
    else:
        masks = masks.contiguous()

    # create output tensors
    render_colors = torch.empty(
        (C, image_height, image_width, color_dim),
        device=device,
        dtype=float_type,
    )
    last_ids = torch.empty(
        (C, image_height, image_width), device=device, dtype=torch.int32
    )
    render_alphas = torch.empty(
        (C, image_height, image_width, 1), device=device, dtype=float_type
    )

    grid = (C, tile_height, tile_width)
    rasterize_to_pixels_fwd_kernel[grid](
        C,
        n_gaussians,
        n_isecs,
        means2D,
        conics,
        colors,
        opacities,
        backgrounds,
        has_background,
        masks,
        has_masks,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        tile_offsets,
        flatten_ids,
        render_colors,
        render_alphas,
        last_ids,
        color_dim,
        GAUSSIANS_BATCH,
    )
    return render_colors, render_alphas, last_ids
