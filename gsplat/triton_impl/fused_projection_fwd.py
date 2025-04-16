import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int32
from torch import Tensor

from gsplat.triton_impl.cam_proj import _persp_proj
from gsplat.triton_impl.quat_scale_to_covar import _quat_scale_to_covar
from gsplat.triton_impl.transform import _transform_cov3D, _transform_v3D
from gsplat.triton_impl.util_kernels import _add_blur, _inverse_sym_mat2


@triton.heuristics({"num_warps": lambda args: max(1, args["BLOCK_SIZE"] // 32)})
@triton.jit
def fused_projection_fwd_kernel(
    N,
    means3D_ptr,  # float32 [N, 3]
    quats_ptr,  # float32 [N, 4]
    scales_ptr,  # float32 [N, 3]
    Ks_ptr,  # float32 [C, 3, 3]
    w2cs_ptr,  # float32 [C, 4, 4]
    image_width,  # int32
    image_height,  # int32
    eps2d,  # float32
    near_plane,  # float32
    far_plane,  # float32
    radius_clip,  # float32
    calc_compensations,  # bool
    # outputs
    means2D_ptr,  # float32 [C, N, 2]
    conics_ptr,  # float32 [C, N, 3]
    depths_ptr,  # float32 [C, N]
    radii_ptr,  # int32 [C, N]
    compensations_ptr,  # float32 [C, N]
    BLOCK_SIZE: tl.constexpr,
):
    """
    compute means2D, conics, depths, radii, compensations
    from 3DGS parameters and camera parameters

    means3D -> means2D: mean3D_world -> mean3D_cam -> mean2D
    quat_scale -> conics: quat_scale -> covar3D_world -> covar3D_cam -> covar2D -> conics (inverse of covar)
    depths is from means3D_cam
    compensations is from covar2D-to-conics inversion
    radii is from conics
    """
    cid = tl.program_id(1)
    g_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g_masks = g_offsets < N

    # 1)
    ##################### covar3d world ###################
    # q, s -> covar3D_world 'c3'
    #######################################################
    ## 1.1) load quats
    quats_ptr = quats_ptr + g_offsets * 4
    q0 = tl.load(quats_ptr, g_masks, 1.0)
    q1 = tl.load(quats_ptr + 1, g_masks, 0.0)
    q2 = tl.load(quats_ptr + 2, g_masks, 0.0)
    q3 = tl.load(quats_ptr + 3, g_masks, 0.0)
    ## 1.2) load scales
    scales_ptr = scales_ptr + g_offsets * 3
    s0 = tl.load(scales_ptr, g_masks, 1.0)
    s1 = tl.load(scales_ptr + 1, g_masks, 1.0)
    s2 = tl.load(scales_ptr + 2, g_masks, 1.0)

    ## 1.3) quat_scale -> covar3D_world
    # fmt: off
    (
        c3_xx, c3_xy, c3_xz, 
        c3_yy, c3_yz, c3_zz,
    ) = _quat_scale_to_covar(
        q0, q1, q2, q3, 
        s0, s1, s2
    )
    # fmt: on
    ############## end 1) covar3D world #################

    # 2)
    ##################### world to cam ####################
    # means3D_world -> means3D_cam 'c_m3'
    # save depths
    # covar3D_world -> covar3D_cam 'c_c3'
    #######################################################
    ## 2.1) load means3D world: 'm3'
    means3D_ptr = means3D_ptr + g_offsets * 3
    m3_x = tl.load(means3D_ptr, g_masks, 0.0)
    m3_y = tl.load(means3D_ptr + 1, g_masks, 0.0)
    m3_z = tl.load(means3D_ptr + 2, g_masks, 0.0)

    ## 2.2) load w2cs: R, T
    w2cs_ptr = w2cs_ptr + cid * 16
    Rxx = tl.load(w2cs_ptr + 0)
    Rxy = tl.load(w2cs_ptr + 1)
    Rxz = tl.load(w2cs_ptr + 2)

    Ryx = tl.load(w2cs_ptr + 4)
    Ryy = tl.load(w2cs_ptr + 5)
    Ryz = tl.load(w2cs_ptr + 6)

    Rzx = tl.load(w2cs_ptr + 8)
    Rzy = tl.load(w2cs_ptr + 9)
    Rzz = tl.load(w2cs_ptr + 10)

    Tx = tl.load(w2cs_ptr + 3)
    Ty = tl.load(w2cs_ptr + 7)
    Tz = tl.load(w2cs_ptr + 11)

    ## 2.3 mean3D world to cam: m3 -> c_m3
    # fmt: off
    c_m3_x, c_m3_y, c_m3_z = _transform_v3D(
        m3_x, m3_y, m3_z,
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
        Tx, Ty, Tz,
    )
    # fmt: on

    ## 2.4 save depths: c_m3_z
    depths_ptr = depths_ptr + cid * N + g_offsets
    tl.store(depths_ptr, c_m3_z, g_masks)
    # masking by depth, true mean ignored
    keep_masks = (c_m3_z > near_plane) & (c_m3_z < far_plane) & g_masks
    # while masked gaussian is ignored, the computation still goes on
    # set z to avoid nan in perspective projection
    # c_m3_z = tl.where(keep_masks, c_m3_z, 1.0)

    ## 2.5) covar3D world to cam space: c3 -> c_c3
    # fmt: off
    (
        c_c3_xx, c_c3_xy, c_c3_xz, 
        c_c3_yy, c_c3_yz, c_c3_zz,
    ) = _transform_cov3D(
        c3_xx, c3_xy, c3_xz, 
        c3_yy, c3_yz, c3_zz,
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    )
    # fmt: on
    ############ end 2) world to cam #####################

    # 3)
    ##################### cam_projection #################
    # means3D_cam -> means2D: c_m3 -> m2
    # covar3D_cam -> covar2D: c_c3 -> c2
    ######################################################

    ## 3.1) load Ks: fx, fy, cx, cy
    Ks_ptr = Ks_ptr + cid * 9
    fx = tl.load(Ks_ptr)
    fy = tl.load(Ks_ptr + 4)
    cx = tl.load(Ks_ptr + 2)
    cy = tl.load(Ks_ptr + 5)

    ## 3.2) cam_projection: 'm2', 'c2'
    # fmt: off
    (
        m2_x, m2_y, 
        cov2_xx, cov2_xy, cov2_yy,
    ) = _persp_proj(
        # means 3D cam space
        c_m3_x, c_m3_y, c_m3_z,
        # covar 3D cam space
        c_c3_xx, c_c3_xy, c_c3_xz, 
        c_c3_yy, c_c3_yz, c_c3_zz,
        fx, fy, cx, cy, 
        image_width, image_height,
    )
    # fmt: on

    ## 3.3) save means2D
    means2D_ptr = means2D_ptr + cid * N * 2 + g_offsets * 2
    tl.store(means2D_ptr, m2_x, keep_masks)
    tl.store(means2D_ptr + 1, m2_y, keep_masks)
    ############## end 3) cam_projection ################

    # 4)
    ################ covar 2D to conics #################
    # add blur to avoid singular covar
    # inverse covar2D_blur to conics2d
    # compute radii = 3 * max_scale
    #####################################################

    ## 4.1) add blur
    ##      NOTE: only cov2_xx, cov2_yy are modified
    det, compensations, cov2_xx, cov2_yy = _add_blur(cov2_xx, cov2_xy, cov2_yy, eps2d)
    # det should be positive after adding blur, so mask out the exceptions
    keep_masks &= det > 0.0

    ## 4.2) save compensations
    if calc_compensations:
        tl.store(compensations_ptr + cid * N + g_offsets, compensations, keep_masks)

    ## 4.3) inverse covar2_blur to conics2d
    conic_xx, conic_xy, conic_yy = _inverse_sym_mat2(cov2_xx, cov2_xy, cov2_yy)

    ## 4.4) save conics
    conics_ptr = conics_ptr + cid * N * 3 + g_offsets * 3
    tl.store(conics_ptr, conic_xx, keep_masks)
    tl.store(conics_ptr + 1, conic_xy, keep_masks)
    tl.store(conics_ptr + 2, conic_yy, keep_masks)

    ## 4.5) compute radii = 3 * max_scale
    ## the max variance direction is the eigen vector of the covariance matrix and the corresponding eigen value is the variance along that direction
    ## now we solve the eigen value of the covariance matrix
    ## lambda^2 - (cov_xx + cov_yy) * lambda + det = 0
    b = 0.5 * (cov2_xx + cov2_yy)
    v1 = b + tl.sqrt(tl.maximum(0.01, b * b - det))
    radii = tl.ceil(3.0 * tl.sqrt(v1))
    keep_masks &= (
        (radii > radius_clip)
        & (m2_x + radii > 0)
        & (m2_x - radii < image_width)
        & (m2_y + radii > 0)
        & (m2_y - radii < image_height)
    )
    radii = radii.cast(tl.int32)
    radii = tl.where(keep_masks, radii, 0)

    ## 4.6) save radii
    radii_ptr = radii_ptr + cid * N + g_offsets
    tl.store(radii_ptr, radii, g_masks)
    ############### end 4) covar 2D to conics ############


@torch.no_grad()
def fused_projection_fwd(
    means3D: Float[Tensor, "N 3"],
    quats: Float[Tensor, "N 4"],
    scales: Float[Tensor, "N 3"],
    viewmats: Float[Tensor, "C 4 4"],
    Ks: Float[Tensor, "C 3 3"],
    image_width: int,
    image_height: int,
    eps2d: float,
    near_plane: float,
    far_plane: float,
    radius_clip: float,
    calc_compensations: bool,
    block_size: int = 256,
) -> Tuple[
    Int32[Tensor, "C N"],
    Float[Tensor, "C N 2"],
    Float[Tensor, "C N"],
    Float[Tensor, "C N 3"],
    Optional[Float[Tensor, "C N"]],
]:
    """
    project 3D gaussian to 2D gaussian

    return radii, means2D, depths, conics, compensations
    """
    device = means3D.device
    float_dtype = means3D.dtype
    C = viewmats.shape[0]
    N = means3D.shape[0]

    means3D = means3D.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()

    # allocate output tensors
    means2D = torch.empty((C, N, 2), device=device, dtype=float_dtype)
    conics = torch.empty((C, N, 3), device=device, dtype=float_dtype)
    depths = torch.empty((C, N), device=device, dtype=float_dtype)
    radii = torch.empty((C, N), device=device, dtype=torch.int32)

    if calc_compensations:
        compensations = torch.zeros((C, N), device=device, dtype=float_dtype)
    else:
        compensations = torch.empty(1, device=device, dtype=float_dtype)

    # launch triton kernel

    grid = (int(math.ceil(N / block_size)), C)
    fused_projection_fwd_kernel[grid](
        N,
        means3D,
        quats,
        scales,
        Ks,
        viewmats,
        image_width,
        image_height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        # outputs
        means2D,
        conics,
        depths,
        radii,
        compensations,
        block_size,
    )
    if not calc_compensations:
        compensations = None
    return radii, means2D, depths, conics, compensations
