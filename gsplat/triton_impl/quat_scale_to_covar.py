import triton
import triton.language as tl
from triton.language.extra import libdevice
from gsplat.triton_impl.util_kernels import _matmul3x3


@triton.jit
def _quat_scale_to_covar(
    # fmt: off
    # quaternion float32 [B]
    q0, q1, q2, q3,
    # scale float32 [B]
    s0, s1, s2,
    # fmt: on
):
    """
    compute covariance matrix from quaternion and scale

    R = quat2R(q)
    S = diag(s)
    C = R @ S @ S^T @ R^T

    args:
        q: float32 [B, 4]
        s: float32 [B, 3]

    return
        C3: float32 [B, 3, 3] symmetric matrices
    """
    (
        # fmt: off
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz,
        # fmt: on
    ) = _quat_to_R(q0, q1, q2, q3)
    # R @ S
    RS_xx = Rxx * s0
    RS_xy = Rxy * s1
    RS_xz = Rxz * s2
    RS_yx = Ryx * s0
    RS_yy = Ryy * s1
    RS_yz = Ryz * s2
    RS_zx = Rzx * s0
    RS_zy = Rzy * s1
    RS_zz = Rzz * s2

    # C = RS @ RS^T
    # fmt: off
    (
        c3_xx, c3_xy, c3_xz,
        c3_yx, c3_yy, c3_yz,
        c3_zx, c3_zy, c3_zz,
    ) = _matmul3x3(
        RS_xx, RS_xy, RS_xz,
        RS_yx, RS_yy, RS_yz,
        RS_zx, RS_zy, RS_zz,

        RS_xx, RS_yx, RS_zx,
        RS_xy, RS_yy, RS_zy,
        RS_xz, RS_yz, RS_zz,
    )
    # fmt: on
    return c3_xx, c3_xy, c3_xz, c3_yy, c3_yz, c3_zz


@triton.jit
def _quat_scale_to_covar_vjp(
    # fmt: off
    # quaternion float32 [B]
    q0, q1, q2, q3,
    # scale float32 [B]
    s0, s1, s2,
    # grad covar 3D float32 [B]
    d_c3_xx, d_c3_xy, d_c3_xz,
             d_c3_yy, d_c3_yz,
                      d_c3_zz,
    # fmt: on
):
    """
    compute grad q, grad s from grad covar 3D

    C = RS @ RS^T
    D_RS = (D_C + D_C^T) @ RS = 2*D_C @ RS
    D_R = D_RS @ S^T
    D_S = R^T @ D_RS

    args:
        q: float32 [B, 4]
        s: float32 [B, 3]
        d_C: float32 [B, 3, 3]

    return:
        d_q: float32 [B, 4]
        d_s: float32 [B, 3]
    """
    (
        # fmt: off
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz,
        # fmt: on
    ) = _quat_to_R(q0, q1, q2, q3)

    # D_RS = 2 * D_C @ RS
    d_c3_yx = d_c3_xy
    d_c3_zx = d_c3_xz
    d_c3_zy = d_c3_yz
    # fmt: off
    (
        D_RS_xx, D_RS_xy, D_RS_xz,
        D_RS_yx, D_RS_yy, D_RS_yz,
        D_RS_zx, D_RS_zy, D_RS_zz,
    ) = _matmul3x3(
        d_c3_xx, d_c3_xy, d_c3_xz,
        d_c3_yx, d_c3_yy, d_c3_yz,
        d_c3_zx, d_c3_zy, d_c3_zz,

        # 2*RS
        2.0*Rxx*s0, 2.0*Rxy*s1, 2.0*Rxz*s2,
        2.0*Ryx*s0, 2.0*Ryy*s1, 2.0*Ryz*s2,
        2.0*Rzx*s0, 2.0*Rzy*s1, 2.0*Rzz*s2,
    )


    d_q0, d_q1, d_q2, d_q3 = _quat_to_R_vjp(
        # quat
        q0, q1, q2, q3,
        # D_R = D_RS @ S
        D_RS_xx * s0, D_RS_xy * s1, D_RS_xz * s2,
        D_RS_yx * s0, D_RS_yy * s1, D_RS_yz * s2,
        D_RS_zx * s0, D_RS_zy * s1, D_RS_zz * s2,
    )
    # fmt: on
    # D_S = R^T @ D_RS
    D_s0 = Rxx * D_RS_xx + Ryx * D_RS_yx + Rzx * D_RS_zx
    D_s1 = Rxy * D_RS_xy + Ryy * D_RS_yy + Rzy * D_RS_zy
    D_s2 = Rxz * D_RS_xz + Ryz * D_RS_yz + Rzz * D_RS_zz
    return (
        # fmt: off
        d_q0, d_q1, d_q2, d_q3,
        D_s0, D_s1, D_s2,
        # fmt: on
    )


@triton.jit
def _quat_to_R(q0, q1, q2, q3):
    """
    convert quaternion to rotation matrix
    NOTE: q is first normalized

    args:
        q: float32 [B, 4] quaternion
    return:
        R: float32 [B, 3, 3]

    CONVENTION:
    q is in the form of (w, x, y, z)
        q0 = cos(theta/2)
        q1 = sin(theta/2) * v_x
        q2 = sin(theta/2) * v_y
        q3 = sin(theta/2) * v_z
    R is left-multiplying the vector
    y = R @ x is the rotation of x
    """
    # normalize quaternion
    inv_norm = libdevice.rsqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    q0 = q0 * inv_norm
    q1 = q1 * inv_norm
    q2 = q2 * inv_norm
    q3 = q3 * inv_norm

    x2 = q1 * q1
    y2 = q2 * q2
    z2 = q3 * q3

    xy = q1 * q2
    xz = q1 * q3
    yz = q2 * q3

    xw = q1 * q0
    yw = q2 * q0
    zw = q3 * q0

    Rxx = 1 - 2 * (y2 + z2)
    Rxy = 2 * (xy - zw)
    Rxz = 2 * (xz + yw)

    Ryx = 2 * (xy + zw)
    Ryy = 1 - 2 * (x2 + z2)
    Ryz = 2 * (yz - xw)

    Rzx = 2 * (xz - yw)
    Rzy = 2 * (yz + xw)
    Rzz = 1 - 2 * (x2 + y2)
    return (
        # fmt: off
        Rxx, Rxy, Rxz, 
        Ryx, Ryy, Ryz, 
        Rzx, Rzy, Rzz,
        # fmt: on
    )


@triton.jit
def _quat_to_R_vjp(
    # fmt: off
    # quaternion float32 [B]
    q0, q1, q2, q3,
    # grad R float32 [B]
    d_Rxx, d_Rxy, d_Rxz,
    d_Ryx, d_Ryy, d_Ryz,
    d_Rzx, d_Rzy, d_Rzz,
):
    """
    compute grad of q from grad of R

    args:
        q: float32 [B, 4]
        d_R: float32 [B, 3, 3]

    return:
        d_q: float32 [B, 4]
    """
    inv_norm = libdevice.rsqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
    w = q0 * inv_norm
    x = q1 * inv_norm
    y = q2 * inv_norm
    z = q3 * inv_norm

    dRzy_m_dRyz = d_Rzy - d_Ryz
    dRxz_m_dRzx = d_Rxz - d_Rzx
    dRyx_m_dRxy = d_Ryx - d_Rxy
    dRxy_p_dRyx = d_Rxy + d_Ryx
    dRxz_p_dRzx = d_Rxz + d_Rzx
    dRyz_p_dRzy = d_Ryz + d_Rzy
    # fmt: off
    dw = 2.0 * (
          x * dRzy_m_dRyz 
        + y * dRxz_m_dRzx 
        + z * dRyx_m_dRxy
    )
    dx = 2.0 * (
         -2.0 * x * (d_Ryy + d_Rzz)
              + y * dRxy_p_dRyx
              + z * dRxz_p_dRzx
              + w * dRzy_m_dRyz
    )
    dy = 2.0 * (
                x * dRxy_p_dRyx
         -2.0 * y * (d_Rxx + d_Rzz)
              + z * dRyz_p_dRzy
              + w * dRxz_m_dRzx
    )
    dz = 2.0 * (
                x * dRxz_p_dRzx
              + y * dRyz_p_dRzy
         -2.0 * z * (d_Rxx + d_Ryy)
              + w * dRyx_m_dRxy
    )
    # fmt: on

    # qn = q * inv_norm = (w,x,y,z)
    # d_q = (d_qn - qn^T * d_qn * qn) * inv_norm
    qn_d_qn = w * dw + x * dx + y * dy + z * dz
    d_q0 = (dw - w * qn_d_qn) * inv_norm
    d_q1 = (dx - x * qn_d_qn) * inv_norm
    d_q2 = (dy - y * qn_d_qn) * inv_norm
    d_q3 = (dz - z * qn_d_qn) * inv_norm
    return d_q0, d_q1, d_q2, d_q3
