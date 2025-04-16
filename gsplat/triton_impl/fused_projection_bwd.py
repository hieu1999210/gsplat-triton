import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from jaxtyping import Float, Int32
from torch import Tensor

from gsplat.triton_impl.cam_proj import _persp_proj_vjp
from gsplat.triton_impl.quat_scale_to_covar import (
    _quat_scale_to_covar,
    _quat_scale_to_covar_vjp,
)
from gsplat.triton_impl.transform import (
    _transform_cov3D,
    _transform_cov3D_vjp,
    _transform_v3D,
    _transform_v3D_vjp,
)
from gsplat.triton_impl.util_kernels import _add_blur_vjp, _inverse_sym_mat2_vjp


@triton.heuristics({"num_warps": lambda args: max(1, args["BLOCK_SIZE"] // 32)})
@triton.jit
def fused_projection_bwd_kernel(
    N,
    # fwd inputs
    means3D_ptr,  # float32 [N, 3]
    quats_ptr,  # float32 [N, 4]
    scales_ptr,  # float32 [N, 3]
    Ks_ptr,  # float32 [C, 3, 3]
    w2cs_ptr,  # float32 [C, 4, 4]
    image_width,  # int32
    image_height,  # int32
    calc_compensations,  # bool
    compensations_ptr,  # float32 [C, N]
    eps2d,  # float32
    # fwd outputs
    radii_ptr,  # int32 [C, N]
    conics_ptr,  # float32 [C, N, 3]
    # grad outputs
    v_means2D_ptr,  # float32 [C, N, 2]
    v_conics_ptr,  # float32 [C, N, 3]
    v_depths_ptr,  # float32 [C, N]
    v_compensations_ptr,  # float32 [C, N] or None
    # grad inputs
    v_means3D_ptr,  # float32 [N, 3]
    v_quats_ptr,  # float32 [N, 4]
    v_scales_ptr,  # float32 [N, 3]
    v_w2cs_ptr,  # float32 [C, 4, 4]
    viewmats_requires_grad,
    BLOCK_SIZE: tl.constexpr,
):
    """
    compute gradients of means3D, quats, scales, w2cs
    from gradients of means2D, conics, depths, compensations

    NOTE: for gaussian which has zero radius,
    we assume all the corresponding given gradients are zero
    so no need to use radii to mask the gradients
    """
    cid = tl.program_id(1)
    g_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    g_masks = g_offsets < N

    # masking computation where radii is zero
    g_masks = tl.load(radii_ptr + cid * N + g_offsets, g_masks, 0) > 0

    # 1)
    ######### backward cov2D inversion ##########
    # v_conics, v_comp -> v_cov2D               #
    #############################################
    ## 1.1) load conics
    conics_ptr = conics_ptr + cid * N * 3 + g_offsets * 3
    conics_xx = tl.load(conics_ptr, g_masks, 1.0)
    conics_xy = tl.load(conics_ptr + 1, g_masks, 0.0)
    conics_yy = tl.load(conics_ptr + 2, g_masks, 1.0)

    ## 1.2) load grad conics
    v_conics_ptr = v_conics_ptr + cid * N * 3 + g_offsets * 3
    v_conics_xx = tl.load(v_conics_ptr, g_masks, 0.0)
    v_conics_yy = tl.load(v_conics_ptr + 2, g_masks, 0.0)

    # NOTE: grad of conics_xy from outside is the sum of
    # grad conic_xy and grad conic_yx here we split them back
    # and use a variable v_conics_xy to represent both v_conics_xy and v_conics_yx
    v_conics_xy = tl.load(v_conics_ptr + 1, g_masks, 0.0) * 0.5

    ## 1.3) compute grad cov2D from grad conics
    v_cov2_xx, v_cov2_xy, v_cov2_yy = _inverse_sym_mat2_vjp(
        # fmt: off
        v_conics_xx, v_conics_xy, v_conics_yy, 
        conics_xx, conics_xy, conics_yy,
        # fmt: on
    )
    ## 1.4) compute grad cov2D from compensations
    if calc_compensations:
        comp = tl.load(compensations_ptr + cid * N + g_offsets, g_masks, 1.0)
        v_comp = tl.load(v_compensations_ptr + cid * N + g_offsets, g_masks, 0.0)
        _v_cov2_xx, _v_cov2_xy, _v_cov2_yy = _add_blur_vjp(
            conics_xx, conics_xy, conics_yy, comp, v_comp, eps2d
        )
        v_cov2_xx += _v_cov2_xx
        v_cov2_xy += _v_cov2_xy
        v_cov2_yy += _v_cov2_yy
    ######### end backward cov2D  inversion #########

    # 2)
    ############# backward camera projection ##############
    # v_means2D, v_cov2D -> v_c_means3D, v_c_cov3D
    #######################################################
    #  2.1) recompute means and cov3D in camera space
    #       (means, quats, scales, w2cs -> c_m3, c_c3)
    #  2.2) compute grad of means3D and cov3D in camera space
    #       from grad of means2D and cov2D
    #       (c_m3, c_c3, Ks, v_means2D, v_cov2D -> v_c_m3, v_c_c3)

    ## 2.1.1) load means 3D in world space
    m3_ptr = means3D_ptr + g_offsets * 3
    m3_x = tl.load(m3_ptr, g_masks, 0.0)
    m3_y = tl.load(m3_ptr + 1, g_masks, 0.0)
    m3_z = tl.load(m3_ptr + 2, g_masks, 0.0)
    ## 2.1.2) load quat
    q_ptr = quats_ptr + g_offsets * 4
    q0 = tl.load(q_ptr, g_masks, 1.0)
    q1 = tl.load(q_ptr + 1, g_masks, 0.0)
    q2 = tl.load(q_ptr + 2, g_masks, 0.0)
    q3 = tl.load(q_ptr + 3, g_masks, 0.0)
    ## 2.1.3) load scale
    s_ptr = scales_ptr + g_offsets * 3
    s0 = tl.load(s_ptr, g_masks, 1.0)
    s1 = tl.load(s_ptr + 1, g_masks, 1.0)
    s2 = tl.load(s_ptr + 2, g_masks, 1.0)

    ## 2.1.4) load w2cs
    w2cs_ptr = w2cs_ptr + cid * 16
    Rxx = tl.load(w2cs_ptr)
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

    ## 2.1.5) get cov3D in camera space
    # fmt: off
    (
        # cov 3d world
        c3_xx, c3_xy, c3_xz,
        c3_yy, c3_yz, c3_zz,
    ) = _quat_scale_to_covar(
        q0, q1, q2, q3,
        s0, s1, s2,
    )
    (
        # cov3D cam
        c_c3_xx, c_c3_xy, c_c3_xz,
        c_c3_yy, c_c3_yz, c_c3_zz,
    ) = _transform_cov3D(
        # cov3D world
        c3_xx, c3_xy, c3_xz,
        c3_yy, c3_yz, c3_zz,
        # R
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    )
    ## 2.1.6) get means3D in camera space
    c_m3_x, c_m3_y, c_m3_z = _transform_v3D(
        m3_x, m3_y, m3_z,
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
        Tx, Ty, Tz,
    )
    # fmt: on

    ## 2.2.1) load intrinsics
    Ks_ptr = Ks_ptr + cid * 9
    fx = tl.load(Ks_ptr)
    fy = tl.load(Ks_ptr + 4)
    cx = tl.load(Ks_ptr + 2)
    cy = tl.load(Ks_ptr + 5)

    ## 2.2.2) load grad means2D
    v_means2D_ptr = v_means2D_ptr + cid * N * 2 + g_offsets * 2
    v_m2_x = tl.load(v_means2D_ptr, g_masks, 0.0)
    v_m2_y = tl.load(v_means2D_ptr + 1, g_masks, 0.0)

    ## 2.2.3 compute grad of means3D and cov3D in camera space
    # fmt: off
    # NOTE clipping c_m3_z to avoid division by zero 
    # while the output grad is zero, the input values need clipped to avoid nan
    # to get zero grad
    # c_m3_z = tl.where(g_masks, c_m3_z, 1.0)
    (
        # grad means 3D cam space
        v_c_m3_x, v_c_m3_y, v_c_m3_z,
        # grad cov 3D cam space
        v_c_c3_xx, v_c_c3_xy, v_c_c3_xz,
        v_c_c3_yy, v_c_c3_yz, v_c_c3_zz,
    ) = _persp_proj_vjp(
        # mean3D cam space
        c_m3_x, c_m3_y, c_m3_z,
        # cov 3D cam space
        c_c3_xx, c_c3_xy, c_c3_xz,
        c_c3_yy, c_c3_yz, c_c3_zz,
        # cam intrinsics
        fx, fy, cx, cy,
        image_width, image_height,
        # grad means 2D
        v_m2_x, v_m2_y,
        # grad cov 2D
        v_cov2_xx, v_cov2_xy, v_cov2_yy,
    )
    # fmt: on
    ######### end backward camera projection #########

    # 3)
    ###### compute grad means3D cam space from grad depth #######
    v_depths = tl.load(v_depths_ptr + cid * N + g_offsets, g_masks, 0.0)
    v_c_m3_z += v_depths

    # 4)
    ###################### backward w2cs ######################
    # v_c_means3D, v_c_cov3D -> v_R, v_T, v_means3D, v_cov30
    ##########################################################

    ## 4.1) compute grad of means3D in world space and grad of w2cs
    ##      from grad of means3D in camera space
    # fmt: off
    (
        # grad means3D world space
        v_m3_x, v_m3_y, v_m3_z,
        # grad translation
        v_Tx, v_Ty, v_Tz,
        # grad rotation matrix
        v_Rxx, v_Rxy, v_Rxz,
        v_Ryx, v_Ryy, v_Ryz,
        v_Rzx, v_Rzy, v_Rzz,
    )= _transform_v3D_vjp(
        # rotation matrix
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
        # mean3D world
        m3_x, m3_y, m3_z,
        # grad means3D cam space
        v_c_m3_x, v_c_m3_y, v_c_m3_z,
        reduce=True,
    )
    # fmt: on

    ## 4.2) save grad mean3D in world space,
    ##      atomic add from all camera
    v_means3D_ptr += g_offsets * 3
    tl.atomic_add(v_means3D_ptr, v_m3_x, mask=g_masks)
    tl.atomic_add(v_means3D_ptr + 1, v_m3_y, mask=g_masks)
    tl.atomic_add(v_means3D_ptr + 2, v_m3_z, mask=g_masks)

    ## 4.3) save grad of translation
    ##      atomic add from all gaussian
    if viewmats_requires_grad:
        v_w2cs_ptr += cid * 16
        tl.atomic_add(v_w2cs_ptr + 3, v_Tx)
        tl.atomic_add(v_w2cs_ptr + 7, v_Ty)
        tl.atomic_add(v_w2cs_ptr + 11, v_Tz)

    ## 4.4) compute grad of cov3D in world space and grad of R
    ##      from grad of cov3D in camera space
    # fmt: off
    (
        v_c3_xx, v_c3_xy, v_c3_xz,
        v_c3_yy, v_c3_yz, v_c3_zz,
        # grad R
        _v_Rxx, _v_Rxy, _v_Rxz,
        _v_Ryx, _v_Ryy, _v_Ryz,
        _v_Rzx, _v_Rzy, _v_Rzz,
    ) = _transform_cov3D_vjp(
        # R
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
        # cov3D world space
        c3_xx, c3_xy, c3_xz,
        c3_yy, c3_yz, c3_zz,
        # grad cov3D cam space
        v_c_c3_xx, v_c_c3_xy, v_c_c3_xz,
        v_c_c3_yy, v_c_c3_yz, v_c_c3_zz,
        reduce=True,
    )
    # fmt: on
    if viewmats_requires_grad:
        v_Rxx += _v_Rxx
        v_Rxy += _v_Rxy
        v_Rxz += _v_Rxz

        v_Ryx += _v_Ryx
        v_Ryy += _v_Ryy
        v_Ryz += _v_Ryz

        v_Rzx += _v_Rzx
        v_Rzy += _v_Rzy
        v_Rzz += _v_Rzz

        ## 4.5) save grad of R
        ##      atomic add from all gaussian
        # no need to update pointer as it is already updated in 4.4
        tl.atomic_add(v_w2cs_ptr, v_Rxx)
        tl.atomic_add(v_w2cs_ptr + 1, v_Rxy)
        tl.atomic_add(v_w2cs_ptr + 2, v_Rxz)

        tl.atomic_add(v_w2cs_ptr + 4, v_Ryx)
        tl.atomic_add(v_w2cs_ptr + 5, v_Ryy)
        tl.atomic_add(v_w2cs_ptr + 6, v_Ryz)

        tl.atomic_add(v_w2cs_ptr + 8, v_Rzx)
        tl.atomic_add(v_w2cs_ptr + 9, v_Rzy)
        tl.atomic_add(v_w2cs_ptr + 10, v_Rzz)
    ##################### end backward w2cs ######################

    # 5)
    ##################### backward quat scale ###################
    # v_cov3D -> v_S, v_q
    #############################################################
    # fmt: off
    (
        v_q0, v_q1, v_q2, v_q3,
        v_s0, v_s1, v_s2,
    ) = _quat_scale_to_covar_vjp(
        # quaternion
        q0, q1, q2, q3,
        # scale
        s0, s1, s2,
        # grad cov3D
        v_c3_xx, v_c3_xy, v_c3_xz,
        v_c3_yy, v_c3_yz, v_c3_zz,
    )
    # fmt: on
    ## save grad of quat scale
    ## atomic add from all camera

    v_quats_ptr += g_offsets * 4
    tl.atomic_add(v_quats_ptr, v_q0, mask=g_masks)
    tl.atomic_add(v_quats_ptr + 1, v_q1, mask=g_masks)
    tl.atomic_add(v_quats_ptr + 2, v_q2, mask=g_masks)
    tl.atomic_add(v_quats_ptr + 3, v_q3, mask=g_masks)

    v_scales_ptr += g_offsets * 3
    tl.atomic_add(v_scales_ptr, v_s0, mask=g_masks)
    tl.atomic_add(v_scales_ptr + 1, v_s1, mask=g_masks)
    tl.atomic_add(v_scales_ptr + 2, v_s2, mask=g_masks)

    ######### end backward quat scale #########


@torch.no_grad()
def fused_projection_bwd(
    # fwd inputs
    means3D: Float[Tensor, "N 3"],
    quats: Float[Tensor, "N 4"],
    scales: Float[Tensor, "N 3"],
    viewmats: Float[Tensor, "C 4 4"],  # w2cs
    Ks: Float[Tensor, "C 3 3"],
    image_width: int,
    image_height: int,
    eps2d: float,
    # fwd outputs
    radii: Int32[Tensor, "C N"],
    conics: Float[Tensor, "C N 3"],
    compensations: Optional[Float[Tensor, "C N"]],
    # fwd grad outputs
    v_means2D: Float[Tensor, "C N 2"],
    v_depths: Float[Tensor, "C N"],
    v_conics: Float[Tensor, "C N 3"],
    v_compensations: Optional[Float[Tensor, "C N"]],
    viewmats_requires_grad: bool,
    block_size: int = 256,
) -> Tuple[
    Float[Tensor, "N 3"],
    Float[Tensor, "N 4"],
    Float[Tensor, "N 3"],
    Float[Tensor, "C 4 4"],
]:
    """
    compute gradients of means3D, quats, scales, w2cs
    from gradients of means2D, conics, depths, compensations
    """
    device = means3D.device
    float_dtype = means3D.dtype
    C = Ks.size(0)
    N = means3D.size(0)

    means3D = means3D.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    radii = radii.contiguous()
    conics = conics.contiguous()
    v_means2D = v_means2D.contiguous()
    v_depths = v_depths.contiguous()
    v_conics = v_conics.contiguous()

    if compensations is not None:
        calc_compensations = True
        compensations = compensations.contiguous()
        v_compensations = v_compensations.contiguous()
    else:
        calc_compensations = False
        compensations = torch.empty(1, device=device, dtype=float_dtype)
        v_compensations = torch.empty(1, device=device, dtype=float_dtype)

    # allocate output tensors
    v_means3D = torch.zeros_like(means3D)
    v_quats = torch.zeros_like(quats)
    v_scales = torch.zeros_like(scales)

    if viewmats_requires_grad:
        v_viewmats = torch.zeros_like(viewmats)
    else:
        v_viewmats = torch.empty(1, device=device, dtype=float_dtype)

    grid = (int(math.ceil(N / block_size)), C)
    fused_projection_bwd_kernel[grid](
        N,
        # fwd inputs
        means3D,
        quats,
        scales,
        Ks,
        viewmats,
        image_width,
        image_height,
        calc_compensations,
        compensations,
        eps2d,
        # fwd outputs
        radii,
        conics,
        # grad outputs
        v_means2D,
        v_conics,
        v_depths,
        v_compensations,
        # grad inputs
        v_means3D,
        v_quats,
        v_scales,
        v_viewmats,
        viewmats_requires_grad,
        block_size,
    )
    if not viewmats_requires_grad:
        v_viewmats = None
    return v_means3D, v_quats, v_scales, v_viewmats
