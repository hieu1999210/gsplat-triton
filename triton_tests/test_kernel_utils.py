import triton
import triton.language as tl


@triton.jit
def load_mat3x3(offsets, ptr, masks):
    """
    default identity
    ptr: Float [N 3 3]
    """
    xx = tl.load(ptr + offsets * 9, mask=masks, other=1.0)
    xy = tl.load(ptr + offsets * 9 + 1, mask=masks, other=0.0)
    xz = tl.load(ptr + offsets * 9 + 2, mask=masks, other=0.0)
    yx = tl.load(ptr + offsets * 9 + 3, mask=masks, other=0.0)
    yy = tl.load(ptr + offsets * 9 + 4, mask=masks, other=1.0)
    yz = tl.load(ptr + offsets * 9 + 5, mask=masks, other=0.0)
    zx = tl.load(ptr + offsets * 9 + 6, mask=masks, other=0.0)
    zy = tl.load(ptr + offsets * 9 + 7, mask=masks, other=0.0)
    zz = tl.load(ptr + offsets * 9 + 8, mask=masks, other=1.0)
    return xx, xy, xz, yx, yy, yz, zx, zy, zz


@triton.jit
def load_sym_mat3(offsets, ptr, masks):
    """
    default identity
    ptr: Float [N 3 3]
    """
    xx = tl.load(ptr + offsets * 9, mask=masks, other=1.0)
    xy = tl.load(ptr + offsets * 9 + 1, mask=masks, other=0.0)
    xz = tl.load(ptr + offsets * 9 + 2, mask=masks, other=0.0)
    yy = tl.load(ptr + offsets * 9 + 4, mask=masks, other=1.0)
    yz = tl.load(ptr + offsets * 9 + 5, mask=masks, other=0.0)
    zz = tl.load(ptr + offsets * 9 + 8, mask=masks, other=1.0)
    return xx, xy, xz, yy, yz, zz


@triton.jit
def load_sym_mat2(offsets, ptr, masks):
    """
    default identity
    ptr: Float [N 2 2]
    """
    xx = tl.load(ptr + offsets * 4, mask=masks, other=1.0)
    xy = tl.load(ptr + offsets * 4 + 1, mask=masks, other=0.0)
    yy = tl.load(ptr + offsets * 4 + 3, mask=masks, other=1.0)
    return xx, xy, yy


@triton.jit
def load_vec2(offsets, ptr, masks, default=0.0):
    """
    default vec 0
    ptr: Float [N 2]

    """
    x = tl.load(ptr + offsets * 2, mask=masks, other=default)
    y = tl.load(ptr + offsets * 2 + 1, mask=masks, other=default)
    return x, y


@triton.jit
def load_vec3(offsets, ptr, masks, default=0.0):
    """
    default vec 0
    ptr: Float [N 3]

    """
    x = tl.load(ptr + offsets * 3, mask=masks, other=default)
    y = tl.load(ptr + offsets * 3 + 1, mask=masks, other=default)
    z = tl.load(ptr + offsets * 3 + 2, mask=masks, other=default)
    return x, y, z


@triton.jit
def load_vec4(offsets, ptr, masks):
    """
    default vec [1 0 0 0]
    ptr: Float [N 4]

    """
    x0 = tl.load(ptr + offsets * 4, mask=masks, other=1.0)
    x1 = tl.load(ptr + offsets * 4 + 1, mask=masks, other=0.0)
    x2 = tl.load(ptr + offsets * 4 + 2, mask=masks, other=0.0)
    x3 = tl.load(ptr + offsets * 4 + 3, mask=masks, other=0.0)
    return x0, x1, x2, x3


@triton.jit
def store_vec2(offsets, ptr, x, y, masks):
    """
    ptr: Float [N 2]
    """
    tl.store(ptr + offsets * 2, x, mask=masks)
    tl.store(ptr + offsets * 2 + 1, y, mask=masks)


@triton.jit
def store_vec3(offsets, ptr, x, y, z, masks):
    """
    ptr: Float [N 3]
    """
    tl.store(ptr + offsets * 3, x, mask=masks)
    tl.store(ptr + offsets * 3 + 1, y, mask=masks)
    tl.store(ptr + offsets * 3 + 2, z, mask=masks)


@triton.jit
def store_vec4(offsets, ptr, x0, x1, x2, x3, masks):
    """
    ptr: Float [N 4]
    """
    tl.store(ptr + offsets * 4, x0, mask=masks)
    tl.store(ptr + offsets * 4 + 1, x1, mask=masks)
    tl.store(ptr + offsets * 4 + 2, x2, mask=masks)
    tl.store(ptr + offsets * 4 + 3, x3, mask=masks)


@triton.jit
def store_sym_mat2(offsets, ptr, xx, xy, yy, masks):
    """
    ptr: Float [N 2 2]
    """
    tl.store(ptr + offsets * 4, xx, mask=masks)
    tl.store(ptr + offsets * 4 + 1, xy, mask=masks)
    tl.store(ptr + offsets * 4 + 2, xy, mask=masks)
    tl.store(ptr + offsets * 4 + 3, yy, mask=masks)


@triton.jit
def store_sym_mat3(offsets, ptr, xx, xy, xz, yy, yz, zz, masks):
    """
    ptr: Float [N 3 3]
    """
    tl.store(ptr + offsets * 9, xx, mask=masks)
    tl.store(ptr + offsets * 9 + 1, xy, mask=masks)
    tl.store(ptr + offsets * 9 + 2, xz, mask=masks)
    tl.store(ptr + offsets * 9 + 3, xy, mask=masks)
    tl.store(ptr + offsets * 9 + 4, yy, mask=masks)
    tl.store(ptr + offsets * 9 + 5, yz, mask=masks)
    tl.store(ptr + offsets * 9 + 6, xz, mask=masks)
    tl.store(ptr + offsets * 9 + 7, yz, mask=masks)
    tl.store(ptr + offsets * 9 + 8, zz, mask=masks)


@triton.jit
def store_mat3x3(offsets, ptr, xx, xy, xz, yx, yy, yz, zx, zy, zz, masks):
    """
    ptr: Float [N 3 3]
    """
    tl.store(ptr + offsets * 9, xx, mask=masks)
    tl.store(ptr + offsets * 9 + 1, xy, mask=masks)
    tl.store(ptr + offsets * 9 + 2, xz, mask=masks)
    tl.store(ptr + offsets * 9 + 3, yx, mask=masks)
    tl.store(ptr + offsets * 9 + 4, yy, mask=masks)
    tl.store(ptr + offsets * 9 + 5, yz, mask=masks)
    tl.store(ptr + offsets * 9 + 6, zx, mask=masks)
    tl.store(ptr + offsets * 9 + 7, zy, mask=masks)
    tl.store(ptr + offsets * 9 + 8, zz, mask=masks)
