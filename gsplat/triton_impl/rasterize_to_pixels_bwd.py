from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from jaxtyping import Bool, Float, Int
from torch import Tensor

from gsplat.triton_impl.utils import TRITON_MAX_NUMEL, is_power_of_two


@triton.heuristics(
    {
        "num_warps": lambda args: max(
            1, min(2, args["GAUSSIANS_BATCH"]) * args["tile_size"] ** 2 // 32
        )
    }
)
@triton.jit
def rasterize_to_pixels_bwd_kernel(
    # gaussian parameters
    n_gaussians,
    means2D_ptr,  # float32 [C, N, 2] or [nnz, 2]
    conics_ptr,  # float32 [C, N, 3] or [nnz, 3]
    colors_ptr,  # float32 [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    opacities_ptr,  # float32 [C, N] or [nnz]
    bg_ptr,  # float32 [C, COLOR_DIM],
    has_background: tl.constexpr,  # bool
    masks_ptr,  # bool [C tile_height tile_width],
    has_masks: tl.constexpr,  # bool
    # shape
    C,
    image_width,
    image_height,
    tile_size: tl.constexpr,
    tile_width,
    tile_height,
    # intersections
    n_isects,
    tile_offsets_ptr,  #: int32 [C tile_height tile_width],
    flatten_ids_ptr,  #: int32 [n_insects],
    # forward outputs
    render_alphas_ptr,  #: float32 [C image_height image_width 1],
    last_ids_ptr,  #: float32 [C image_height image_width],
    # grad outputs
    v_render_colors_ptr,  #: float32 [C image_height image_width COLOR_DIM],
    v_render_alphas_ptr,  #: float32 [C image_height image_width 1],
    # grad inputs
    v_means2D_abs_ptr,  #: float32 [C, N, 2] or [nnz, 2]
    v_means2D_ptr,  #: float32 [C, N, 2] or [nnz, 2]
    v_conics_ptr,  #: float32 [C, N, 3] or [nnz, 3]
    v_colors_ptr,  #: float32 [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    v_opacities_ptr,  #: float32 [C, N] or [nnz]
    absgrad,  # bool
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

    # pixel indices
    pixels_x = tl.arange(0, tile_size) + tl.program_id(2) * tile_size
    pixels_y = tl.arange(0, tile_size) + tl.program_id(1) * tile_size
    tl.max_contiguous(tl.multiple_of(pixels_x, tile_size), tile_size)
    tl.max_contiguous(tl.multiple_of(pixels_y, tile_size), tile_size)

    # pixel coordinates
    px = pixels_x.cast(tl.float32) + 0.5
    py = pixels_y.cast(tl.float32) + 0.5

    # pixel offsets to load data
    pixels_offsets = (
        cam_id * image_height * image_width
        + pixels_y[:, None] * image_width
        + pixels_x[None, :]
    )
    pixel_mask = (pixels_y[:, None] < image_height) & (pixels_x[None, :] < image_width)

    # get gaussian index range
    tile_offset_ptr = tile_offsets_ptr + cam_id * tile_height * tile_width + tile_id
    range_start = tl.load(tile_offset_ptr)
    if cam_id == C - 1 and tile_id == tile_height * tile_width - 1:
        range_end = n_isects
    else:
        range_end = tl.load(tile_offset_ptr + 1)

    # load tile last ids
    tile_last_ids = tl.load(last_ids_ptr + pixels_offsets, mask=pixel_mask, other=0)

    # update range end
    tile_last_id = tl.max(tile_last_ids)
    range_end = min(range_end, tile_last_id + 1)
    num_batches = tl.cdiv(range_end - range_start, GAUSSIANS_BATCH)

    # constant variable per tile
    Dralpha = tl.load(v_render_alphas_ptr + pixels_offsets, mask=pixel_mask, other=0.0)
    Drcolors = tl.load(
        v_render_colors_ptr
        + pixels_offsets[:, :, None] * COLOR_DIM
        + tl.arange(0, COLOR_DIM)[None, None, :],
        mask=pixel_mask[:, :, None],
        other=0.0,
    )
    T_final = 1.0 - tl.load(
        render_alphas_ptr + pixels_offsets, mask=pixel_mask, other=0.0
    )
    if has_background:
        bg_color = tl.load(bg_ptr + cam_id * COLOR_DIM + tl.arange(0, COLOR_DIM))

    # variables that need prepare from far to near
    ## c_log_T is updated from far to near
    ## by gradually removing far gaussian
    c_log_T = libdevice.fast_logf(T_final)

    ##  render_color * d_render_color
    ## where render_color is accumulated from far to near
    c_rcolor_Drcolor = tl.zeros((tile_size, tile_size), dtype=tl.float32)

    # batch_id = num_batches
    # while batch_id > 0:
    #     batch_id -= 1
    for i in tl.range(num_batches):
        # reverse loop
        batch_id = num_batches - 1 - i

        # get indices of the gaussian batch
        start = range_start + batch_id * GAUSSIANS_BATCH
        end = min(range_end, start + GAUSSIANS_BATCH)
        gid_offsets = start + tl.arange(0, GAUSSIANS_BATCH)
        g_ids = tl.load(
            flatten_ids_ptr + gid_offsets,
            mask=gid_offsets < end,
            other=n_gaussians,
        )
        g_ids = tl.flip(g_ids, dim=0)
        # NOTE ordering gaussian from far to near
        g_mask = g_ids < n_gaussians

        # gaussian parameters
        ## means2D
        means2D_x = tl.load(means2D_ptr + g_ids * 2, mask=g_mask, other=0.0)
        means2D_y = tl.load(means2D_ptr + g_ids * 2 + 1, mask=g_mask, other=0.0)
        ## conics
        conics_xx = tl.load(conics_ptr + g_ids * 3, g_mask, 0.0)[:, None, None]
        conics_xy = tl.load(conics_ptr + g_ids * 3 + 1, g_mask, 0.0)[:, None, None]
        conics_yy = tl.load(conics_ptr + g_ids * 3 + 2, g_mask, 0.0)[:, None, None]
        ## opacities
        opacs = tl.load(opacities_ptr + g_ids, g_mask, 0.0)
        ## colors
        color_offsets = g_ids[:, None] * COLOR_DIM + tl.arange(0, COLOR_DIM)[None, :]
        tl.max_contiguous(tl.multiple_of(color_offsets, (COLOR_DIM, 1)), (COLOR_DIM, 1))
        gcolors = tl.load(
            colors_ptr + color_offsets,
            mask=g_mask[:, None],
            other=0.0,
        )

        # skip gaussian that has continuous id > last_id
        cont_ids = start + tl.flip(tl.arange(0, GAUSSIANS_BATCH), dim=0)
        skip = cont_ids[:, None, None] > tile_last_ids[None, :, :]

        # compute alpha
        delta_x = means2D_x[:, None, None] - px[None, None, :]
        delta_y = means2D_y[:, None, None] - py[None, :, None]
        sigma = (
            0.5 * conics_xx * delta_x * delta_x
            + 0.5 * conics_yy * delta_y * delta_y
            + conics_xy * delta_x * delta_y
        )
        alpha = opacs[:, None, None] * libdevice.fast_expf(-sigma)

        # ignore some gaussians with small alpha or negative sigma
        skip |= (alpha < 1.0 / 255.0) | (sigma < 0.0)
        alpha = tl.where(skip, 0.0, alpha)

        # clip alpha by 0.999
        # clipped items dont have gradients
        Dalpha_skip = (alpha > 0.999) | skip
        alpha = tl.clamp(alpha, 0.0, 0.999)

        # compute T for current batch
        log_1_m_alpha = libdevice.fast_logf(1.0 - alpha)
        # NOTE log_T is exclusive, i.e not include the current gaussian
        log_T = c_log_T[None, :, :] - tl.cumsum(log_1_m_alpha, 0)
        T = libdevice.fast_expf(log_T)

        # update c_log_T for next batch
        c_log_T -= tl.sum(log_1_m_alpha, 0)

        # cumulate d_gcolors from this tile
        # d_gcolor = d_rcolors*T*alpha
        tl.atomic_add(
            v_colors_ptr
            + g_ids[:, None] * COLOR_DIM
            + tl.arange(0, COLOR_DIM)[None, :],
            mask=g_mask[:, None],
            val=tl.sum(
                tl.reshape(
                    (T * alpha)[:, :, :, None] * Drcolors[None, :, :, :],
                    GAUSSIANS_BATCH,
                    tile_size * tile_size,
                    COLOR_DIM,
                ),
                1,
            ),
        )

        # for other gradient the color dimension can be reduced
        ## d_render_color * gaussian color
        ## but this expand batch dimension
        gcolor_Drcolor = tl.sum(Drcolors[None, :, :, :] * gcolors[:, None, None, :], 3)

        # get d_render_color * render_color
        # render_color = sum_gauss(T*alpha*g_color)
        # summation from the current gaussian to the last
        _rcolor_Drcolor = gcolor_Drcolor * T * alpha
        rcolor_Drcolor = c_rcolor_Drcolor + tl.cumsum(_rcolor_Drcolor, 0)
        # update c_rcolor_Drcolor for next batch
        c_rcolor_Drcolor += tl.sum(_rcolor_Drcolor, 0)

        # the total gradient of alpha is from
        # both gradient of render_alpha and render_color
        # Dalpha = Dralpha * pD(ralpha/alpha) + Drcolor * pD(rcolor/alpha)
        # where
        # pD(rcolor/alpha) = (T*gcolor - rcolor - T_final*background)/(1-alpha)
        # ( first term is the current gaussian,
        #   second term is (mainly) the gaussian after,
        #   third term is the background
        # )
        # pD(ralpha/alpha) = T_final/(1-alpha)

        Dalpha = libdevice.fast_expf(-log_1_m_alpha) * (
            T_final * Dralpha + T * gcolor_Drcolor - rcolor_Drcolor
        )

        if has_background:
            bg_term = tl.sum(bg_color[None, None, :] * Drcolors, 2) * T_final
            Dalpha -= libdevice.fast_expf(-log_1_m_alpha) * bg_term

        Dalpha = tl.where(Dalpha_skip, 0.0, Dalpha)

        # opacities gradient
        # pD(alpha/opacity) = exp(-sigma)
        tl.atomic_add(
            v_opacities_ptr + g_ids,
            mask=g_mask,
            val=tl.sum(
                tl.reshape(
                    Dalpha * libdevice.fast_expf(-sigma),
                    GAUSSIANS_BATCH,
                    tile_size * tile_size,
                ),
                1,
            ),
        )

        alpha_Dalpha = alpha * Dalpha

        # means2D gradient
        # pD(alpha/means2D) = -alpha*conic*delta
        conic_delta_x = conics_xx * delta_x + conics_xy * delta_y
        conic_delta_y = conics_xy * delta_x + conics_yy * delta_y

        Dmeans2D_x = tl.reshape(
            -alpha_Dalpha * conic_delta_x, GAUSSIANS_BATCH, tile_size * tile_size
        )
        Dmeans2D_y = tl.reshape(
            -alpha_Dalpha * conic_delta_y, GAUSSIANS_BATCH, tile_size * tile_size
        )

        tl.atomic_add(v_means2D_ptr + g_ids * 2, val=tl.sum(Dmeans2D_x, 1))
        tl.atomic_add(v_means2D_ptr + g_ids * 2 + 1, val=tl.sum(Dmeans2D_y, 1))

        if absgrad:
            tl.atomic_add(
                v_means2D_abs_ptr + g_ids * 2,
                mask=g_mask,
                val=tl.sum(tl.abs(Dmeans2D_x), 1),
            )
            tl.atomic_add(
                v_means2D_abs_ptr + g_ids * 2 + 1,
                mask=g_mask,
                val=tl.sum(tl.abs(Dmeans2D_y), 1),
            )

        # conics gradient
        # pD(alpha/conics) = -0.5*alpha*delta*delta.T
        # 0.5 is multiplied to to delta*delta.T
        Dconics_xx = tl.sum(
            tl.reshape(
                -0.5 * alpha_Dalpha * delta_x * delta_x,
                GAUSSIANS_BATCH,
                tile_size * tile_size,
            ),
            1,
        )
        tl.atomic_add(
            v_conics_ptr + g_ids * 3,
            mask=g_mask,
            val=Dconics_xx,
        )

        Dconics_xy = tl.sum(
            tl.reshape(
                -alpha_Dalpha * delta_x * delta_y,
                GAUSSIANS_BATCH,
                tile_size * tile_size,
            ),
            1,
        )
        tl.atomic_add(
            v_conics_ptr + g_ids * 3 + 1,
            mask=g_mask,
            val=Dconics_xy,
        )

        Dconics_yy = tl.sum(
            tl.reshape(
                -0.5 * alpha_Dalpha * delta_y * delta_y,
                GAUSSIANS_BATCH,
                tile_size * tile_size,
            ),
            1,
        )
        tl.atomic_add(
            v_conics_ptr + g_ids * 3 + 2,
            mask=g_mask,
            val=Dconics_yy,
        )


@torch.no_grad()
def rasterize_to_pixels_bwd(
    # gaussian parameters
    means2D: Union[Float[Tensor, "C N 2"], Float[Tensor, "nnz 2"]],
    conics: Union[Float[Tensor, "C N 3"], Float[Tensor, "nnz 3"]],
    colors: Union[Float[Tensor, "C N COLOR_DIM"], Float[Tensor, "nnz COLOR_DIM"]],
    opacities: Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
    backgrounds: Optional[Float[Tensor, "C COLOR_DIM"]],
    masks: Optional[Bool[Tensor, "C tile_height tile_width"]],
    # shape
    image_width: int,
    image_height: int,
    tile_size: int,
    # intersections
    tile_offsets: Int[Tensor, "C tile_height tile_width"],
    flatten_ids: Int[Tensor, "n_insects"],
    # forward output
    render_alphas: Float[Tensor, "C image_height image_width 1"],
    last_ids: Int[Tensor, "C image_height image_width"],
    # output gradients
    v_render_colors: Float[Tensor, "C image_height image_width COLOR_DIM"],
    v_render_alphas: Float[Tensor, "C image_height image_width 1"],
    absgrad: bool = False,
    GAUSSIANS_BATCH: int = 128,
) -> Tuple[
    Union[Float[Tensor, "C N 2"], Float[Tensor, "nnz 2"]],
    Union[Float[Tensor, "C N 3"], Float[Tensor, "nnz 3"]],
    Union[Float[Tensor, "C N COLOR_DIM"], Float[Tensor, "nnz COLOR_DIM"]],
    Union[Float[Tensor, "C N"], Float[Tensor, "nnz"]],
]:
    """
    return v_means2D, v_conics, v_colors, v_opacities
    """
    assert is_power_of_two(tile_size)
    color_dim = colors.shape[-1]
    assert is_power_of_two(color_dim)
    assert color_dim * tile_size * tile_size * GAUSSIANS_BATCH <= TRITON_MAX_NUMEL
    # GAUSSIANS_BATCH = TRITON_MAX_NUMEL // color_dim // tile_size // tile_size

    C, tile_height, tile_width = tile_offsets.size()
    n_isecs = flatten_ids.shape[0]
    n_gaussians = means2D.numel() // 2

    means2D = means2D.contiguous()
    conics = conics.contiguous()
    colors = colors.contiguous()
    opacities = opacities.contiguous()
    tile_offsets = tile_offsets.contiguous()
    v_render_colors = v_render_colors.contiguous()
    v_render_alphas = v_render_alphas.contiguous()

    # create flag and dummy tensors for background
    has_background = backgrounds is not None
    if not has_background:
        backgrounds = torch.empty(1, device=means2D.device, dtype=means2D.dtype)
    else:
        backgrounds = backgrounds.contiguous()

    # create flag and dummy tensors for masks
    has_masks = masks is not None
    if not has_masks:
        masks = torch.empty(1, device=means2D.device, dtype=torch.bool)
    else:
        masks = masks.contiguous()

    # create output tensors
    v_means2d_abs = (
        torch.zeros_like(means2D)
        if absgrad
        else torch.empty(1, device=means2D.device, dtype=means2D.dtype)
    )
    v_means2D = torch.zeros_like(means2D)
    v_conics = torch.zeros_like(conics)
    v_colors = torch.zeros_like(colors)
    v_opacities = torch.zeros_like(opacities)

    grid = (C, tile_height, tile_width)
    rasterize_to_pixels_bwd_kernel[grid](
        ## gaussian parameters
        n_gaussians,
        means2D,
        conics,
        colors,
        opacities,
        backgrounds,
        has_background,
        masks,
        has_masks,
        ## shape
        C,
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        ## intersections
        n_isecs,
        tile_offsets,
        flatten_ids,
        render_alphas,
        last_ids,
        ## grad outputs
        v_render_colors,
        v_render_alphas,
        ## grad inputs
        v_means2d_abs,
        v_means2D,
        v_conics,
        v_colors,
        v_opacities,
        ## others
        absgrad,
        color_dim,
        GAUSSIANS_BATCH,
    )
    if not absgrad:
        v_means2d_abs = None
    return v_means2d_abs, v_means2D, v_conics, v_colors, v_opacities
