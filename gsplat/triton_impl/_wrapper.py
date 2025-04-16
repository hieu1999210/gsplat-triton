import functools
import importlib
import os
import sys
import warnings
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal

from gsplat.triton_impl.utils import is_power_of_two

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_triton_kernel(module_name, func_name=None, cache=True):
    if func_name is None:
        func_name = module_name
    if cache:
        return _load_triton_kernel_cache(module_name, func_name)
    return _load_triton_kernel(module_name, func_name)


@functools.cache
def _load_triton_kernel_cache(module_name, func_name):
    return _load_triton_kernel(module_name, func_name)


def _load_triton_kernel(module_name, func_name):
    """
    assume file name is the same as the function name
    """
    filepath = os.path.join(CURRENT_DIR, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, func_name)


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
        block_size: int = 8,
    ) -> Tuple[Tensor, Tensor]:
        """
        #! tile size must be constant during the whole run time
        """
        render_colors, render_alphas, last_ids = load_triton_kernel(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            block_size,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.block_size = block_size

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
        # v_last_ids: Tensor,
    ):

        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        block_size = ctx.block_size

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = load_triton_kernel("rasterize_to_pixels_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors,
            v_render_alphas,
            absgrad,
            block_size,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None
        # n_nan = torch.isnan(v_means2d).sum()
        # if n_nan > 0:
        #     warnings.warn(f"v_means2d contains {n_nan} NaN")
        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    block_size: int = 8,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if not is_power_of_two(channels):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds if backgrounds is not None else None,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        absgrad,
        block_size,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        block_size: int = 256,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if camera_model != "pinhole":
            raise NotImplementedError(f"Unsupported camera model: {camera_model}")

        assert (covars is None) and (quats is not None) and (scales is not None)
        radii, means2d, depths, conics, compensations = load_triton_kernel(
            "fused_projection_fwd"
        )(
            means3D=means,
            quats=quats,
            scales=scales,
            viewmats=viewmats,
            Ks=Ks,
            image_width=width,
            image_height=height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=calc_compensations,
            block_size=block_size,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.block_size = block_size

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_raddii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        block_size = ctx.block_size

        v_means, v_quats, v_scales, v_viewmats = load_triton_kernel(
            "fused_projection_bwd"
        )(
            # fwd inputs
            means3D=means,
            quats=quats,
            scales=scales,
            viewmats=viewmats,
            Ks=Ks,
            image_width=width,
            image_height=height,
            eps2d=eps2d,
            # fwd outputs
            radii=radii,
            conics=conics,
            compensations=compensations,
            # fwd grad outputs
            v_means2D=v_means2d,
            v_depths=v_depths,
            v_conics=v_conics,
            v_compensations=v_compensations,
            viewmats_requires_grad=ctx.needs_input_grad[4],
            block_size=block_size,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        # n_nan = torch.isnan(v_means).sum()
        # if n_nan > 0:
        #     warnings.warn(f"v_means contains {n_nan} NaN")
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    block_size: int = 256,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()

    assert covars is None
    assert quats is not None, "covars or quats is required"
    assert scales is not None, "covars or scales is required"
    assert quats.size() == (N, 4), quats.size()
    assert scales.size() == (N, 3), scales.size()

    assert ~sparse_grad and ~packed, "packed is not supported yet"

    return _FullyFusedProjection.apply(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model,
        block_size,
    )


def isect_tiles(*args, **kwargs):
    return load_triton_kernel("isect_tiles")(*args, **kwargs)


def isect_offset_encode(*args, **kwargs):
    return load_triton_kernel("isect_offset", "get_isect_offsets")(*args, **kwargs)


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx,
        sh_degree: int,
        dirs: Tensor,
        coeffs: Tensor,
        masks: Tensor,
        block_size: int = None,
    ) -> Tensor:
        colors = load_triton_kernel("sh_fwd", "sh_to_color_fwd")(
            sh_degree, dirs, coeffs, block_size
        )
        if masks is not None:
            colors[~masks] = 0
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.block_size = block_size
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        block_size = ctx.block_size

        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = load_triton_kernel("sh_bwd", "sh_to_color_bwd")(
            sh_degree,
            dirs,
            coeffs,
            v_colors,
            compute_v_dirs,
            block_size,
        )
        if masks is not None:
            v_coeffs[~masks] = 0
            if compute_v_dirs:
                v_dirs[~masks] = 0
        return None, v_dirs, v_coeffs, None, None


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
    block_size: int = None,
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
    return _SphericalHarmonics.apply(degrees_to_use, dirs, coeffs, masks, block_size)
