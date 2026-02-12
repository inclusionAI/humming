import torch
import cuda.bindings.driver as cbd


tmap_type_map = {
    torch.int8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def make_tma_desc(
    t: torch.Tensor,
    smem_dims,
    gmem_dims=None,
    gmem_outer_strides=None,
    swizzle_bytes: int = 0,
) -> cbd.CUtensorMap:
    if not isinstance(smem_dims[0], cbd.cuuint32_t):
        smem_dims = tuple(cbd.cuuint32_t(x) for x in smem_dims)
    if gmem_dims is None:
        gmem_dims = tuple(cbd.cuuint64_t(x) for x in t.shape[::-1])
    if gmem_outer_strides is None:
        size = t.element_size()
        gmem_outer_strides = [t.stride(i) * size for i in range(t.ndim - 1)]
        gmem_outer_strides = tuple(cbd.cuuint64_t(x) for x in gmem_outer_strides)
        gmem_outer_strides = gmem_outer_strides[::-1]
    elif gmem_outer_strides and not isinstance(gmem_outer_strides[0], cbd.cuuint64_t):
        gmem_outer_strides = tuple(cbd.cuuint64_t(x) for x in gmem_outer_strides)

    assert len(smem_dims) == t.ndim
    assert len(gmem_dims) == t.ndim
    assert len(gmem_outer_strides) == t.ndim - 1

    element_strides = tuple(cbd.cuuint32_t(1) for _ in range(t.ndim))
    tensor_dtype = tmap_type_map[t.dtype]
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        t.ndim,
        t.data_ptr(),
        gmem_dims,
        gmem_outer_strides,
        smem_dims,
        element_strides,
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type_map[swizzle_bytes],
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f"Failed to encode tensor map: {repr(res)}")
    return tensor_map
