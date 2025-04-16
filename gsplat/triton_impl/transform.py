import triton
import triton.language as tl

from gsplat.triton_impl.util_kernels import _matmul3x3


@triton.jit
def _transform_v3D(
    # fmt: off
    # vector
    Vx, Vy, Vz,
    # rotation matrix
    Rxx,Rxy,Rxz,
    Ryx,Ryy,Ryz,
    Rzx,Rzy,Rzz,
    # translation vector
    Tx,Ty,Tz,
    # fmt: on
):
    """
    compute U = R * V + T

    args:
        R: [3, 3]
        V: [B, 3]
        T: [3]

    return
        U: [B, 3]

    """
    Ux = Rxx * Vx + Rxy * Vy + Rxz * Vz + Tx
    Uy = Ryx * Vx + Ryy * Vy + Ryz * Vz + Ty
    Uz = Rzx * Vx + Rzy * Vy + Rzz * Vz + Tz
    return Ux, Uy, Uz


@triton.jit
def _transform_v3D_vjp(
    # fmt: off
    # rotation matrix
    Rxx,Rxy,Rxz,
    Ryx,Ryy,Ryz,
    Rzx,Rzy,Rzz,
    # fwd input vector
    Vx, Vy, Vz,
    # grad fwd output vector
    D_Ux, D_Uy, D_Uz,
    # fmt: on
    reduce=False,
):
    """
    compute D_V, D_R, D_T from D_U
    U = R * V + T
    D_V = R^T * D_U
    D_T = D_U
    D_R = D_U * V^T

    args:
        R: [3, 3]
        V: [B, 3]
        D_U: [B, 3]
        reduce: if True, reduce the gradients of the rotation matrix

    return
        D_V: [B, 3]
        D_T: [3] if reduce else [B, 3, 3]
        D_R: [3, 3] if reduce else [B, 3, 3]

    """
    D_Vx = Rxx * D_Ux + Ryx * D_Uy + Rzx * D_Uz
    D_Vy = Rxy * D_Ux + Ryy * D_Uy + Rzy * D_Uz
    D_Vz = Rxz * D_Ux + Ryz * D_Uy + Rzz * D_Uz

    D_Rxx = D_Ux * Vx
    D_Rxy = D_Ux * Vy
    D_Rxz = D_Ux * Vz

    D_Ryx = D_Uy * Vx
    D_Ryy = D_Uy * Vy
    D_Ryz = D_Uy * Vz

    D_Rzx = D_Uz * Vx
    D_Rzy = D_Uz * Vy
    D_Rzz = D_Uz * Vz

    D_Tx = D_Ux
    D_Ty = D_Uy
    D_Tz = D_Uz

    if reduce:
        D_Rxx = tl.sum(D_Rxx, axis=0)
        D_Rxy = tl.sum(D_Rxy, axis=0)
        D_Rxz = tl.sum(D_Rxz, axis=0)

        D_Ryx = tl.sum(D_Ryx, axis=0)
        D_Ryy = tl.sum(D_Ryy, axis=0)
        D_Ryz = tl.sum(D_Ryz, axis=0)

        D_Rzx = tl.sum(D_Rzx, axis=0)
        D_Rzy = tl.sum(D_Rzy, axis=0)
        D_Rzz = tl.sum(D_Rzz, axis=0)

        D_Tx = tl.sum(D_Tx, axis=0)
        D_Ty = tl.sum(D_Ty, axis=0)
        D_Tz = tl.sum(D_Tz, axis=0)

    return (
        # fmt: off
        # grad input vector
        D_Vx, D_Vy, D_Vz,
        # grad translation vector
        D_Tx, D_Ty, D_Tz,
        # grad rotation matrix
        D_Rxx, D_Rxy, D_Rxz,
        D_Ryx, D_Ryy, D_Ryz,
        D_Rzx, D_Rzy, D_Rzz,
        # fmt: on
    )


@triton.jit
def _transform_cov3D(
    # fmt: off
    # covariance matrix
    Cxx, Cxy, Cxz,
         Cyy, Cyz,
              Czz,
    # rotation matrix
    Rxx, Rxy, Rxz,
    Ryx, Ryy, Ryz,
    Rzx, Rzy, Rzz,
    # fmt: on
):
    """
    compute S = R * C * R^T

    args:
        R: [3, 3]
        C: [B, 3, 3]

    return S: [B, 3, 3]
        Sxx, Sxy, Sxz,
             Syy, Syz,
                  Szz

    NOTE: while C is symmetric, we treat it as a general matrix in gradient computation
    i.e. Cxy and Cyx are 2 independent variables that happen to have the same values
    """
    Cyx = Cxy
    Czx = Cxz
    Czy = Cyz
    # fmt: off
    (
        RC_xx, RC_xy, RC_xz, 
        RC_yx, RC_yy, RC_yz, 
        RC_zx, RC_zy, RC_zz,
    ) = _matmul3x3(
        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,

        Cxx, Cxy, Cxz,
        Cyx, Cyy, Cyz,
        Czx, Czy, Czz,
    )
    (
        Sxx, Sxy, Sxz, 
        Syx, Syy, Syz, 
        Szx, Szy, Szz,
    ) = _matmul3x3(
        RC_xx, RC_xy, RC_xz,
        RC_yx, RC_yy, RC_yz,
        RC_zx, RC_zy, RC_zz,

        Rxx, Ryx, Rzx,
        Rxy, Ryy, Rzy,
        Rxz, Ryz, Rzz,
    )
    # fmt: on
    return Sxx, Sxy, Sxz, Syy, Syz, Szz


@triton.jit
def _transform_cov3D_vjp(
    # fmt: off
    # rotation matrix
    Rxx, Rxy, Rxz,
    Ryx, Ryy, Ryz,
    Rzx, Rzy, Rzz,
    # covariance matrix (fwd input)
    Cxx, Cxy, Cxz,
         Cyy, Cyz,
              Czz,
    # grad fwd output
    D_Sxx, D_Sxy, D_Sxz,
           D_Syy, D_Syz, 
                  D_Szz,
    reduce=False,
    # fmt: on
):
    """
    compute D_C, D_R from D_S
    S = R * C * R^T
    D_C = R^T * D_S * R
    D_R = D_S * R * C^T + D_S^T * R * C
        = 2 * D_S * R * C due to the symmetry of C and D_S

    args:
        R: [3, 3]
        C: [B, 3, 3]
        D_S: [B, 3, 3]

    return
        D_C: [B, 3, 3]
        D_R: [3, 3] if reduce else [B, 3, 3]

    NOTE: while some matrices are symmetric,
    we treat their entries as independent variables even if they have the same values
    i.e. Cxy and Cyx are independent variables that happen to have the same values and same gradients

    """
    D_Syx = D_Sxy
    D_Szx = D_Sxz
    D_Szy = D_Syz
    # fmt: off
    # D_S @ R
    (
        D_SRxx, D_SRxy, D_SRxz,
        D_SRyx, D_SRyy, D_SRyz,
        D_SRzx, D_SRzy, D_SRzz,
    ) = _matmul3x3(
        D_Sxx, D_Sxy, D_Sxz,
        D_Syx, D_Syy, D_Syz,
        D_Szx, D_Szy, D_Szz,

        Rxx, Rxy, Rxz,
        Ryx, Ryy, Ryz,
        Rzx, Rzy, Rzz,
    )
    # D_C = R^T @ (D_S @ R)
    (
        D_Cxx, D_Cxy, D_Cxz,
        D_Cyx, D_Cyy, D_Cyz,
        D_Czx, D_Czy, D_Czz,
    ) = _matmul3x3(
        Rxx, Ryx, Rzx,
        Rxy, Ryy, Rzy,
        Rxz, Ryz, Rzz,

        D_SRxx, D_SRxy, D_SRxz,
        D_SRyx, D_SRyy, D_SRyz,
        D_SRzx, D_SRzy, D_SRzz,
    )

    # D_R = 2*(D_S @ R) @ C
    Cyx = Cxy
    Czx = Cxz
    Czy = Cyz
    (
        D_Rxx, D_Rxy, D_Rxz,
        D_Ryx, D_Ryy, D_Ryz,
        D_Rzx, D_Rzy, D_Rzz,
    ) = _matmul3x3(
        2.0*D_SRxx, 2.0*D_SRxy, 2.0*D_SRxz,
        2.0*D_SRyx, 2.0*D_SRyy, 2.0*D_SRyz,
        2.0*D_SRzx, 2.0*D_SRzy, 2.0*D_SRzz,

        Cxx, Cxy, Cxz,
        Cyx, Cyy, Cyz,
        Czx, Czy, Czz,
    )
    # fmt: on
    if reduce:
        D_Rxx = tl.sum(D_Rxx, axis=0)
        D_Rxy = tl.sum(D_Rxy, axis=0)
        D_Rxz = tl.sum(D_Rxz, axis=0)

        D_Ryx = tl.sum(D_Ryx, axis=0)
        D_Ryy = tl.sum(D_Ryy, axis=0)
        D_Ryz = tl.sum(D_Ryz, axis=0)

        D_Rzx = tl.sum(D_Rzx, axis=0)
        D_Rzy = tl.sum(D_Rzy, axis=0)
        D_Rzz = tl.sum(D_Rzz, axis=0)

    return (
        # fmt: off
        D_Cxx, D_Cxy, D_Cxz,
               D_Cyy, D_Cyz,
                      D_Czz,

        D_Rxx, D_Rxy, D_Rxz,
        D_Ryx, D_Ryy, D_Ryz,
        D_Rzx, D_Rzy, D_Rzz,
        # fmt: on
    )
