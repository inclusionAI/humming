import triton
from humming import dtypes
from humming.layer import HummingLayer
from humming.utils.test import generate_random_inputs, generate_random_weight


a_dtypes = [dtypes.float16, dtypes.float8e4m3]
b_dtypes = [dtypes.uint3, dtypes.uint4, dtypes.uint5]


def run_example(group_size, a_dtype, b_dtype):
    layer = HummingLayer(
        shape_n=8192,
        shape_k=8192,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype=dtypes.float16,
        bs_dtype=dtypes.float16,
        input_scale_group_size=0,
        weight_scale_group_size=group_size,
        mma_type="wgmma",
    ).cuda()

    random_weight_data = generate_random_weight(
        n=layer.shape_n,
        k=layer.shape_k,
        group_size=layer.weight_scale_group_size,
        dtype=layer.b_dtype,
        scale_dtype=layer.bs_dtype,
    )

    _, _, weight, weight_scale, _, _ = random_weight_data
    _, _, inputs, input_scale = generate_random_inputs(8192, layer.shape_k, dtype=layer.a_dtype)

    layer.load_weight(weight=weight, weight_scale=weight_scale)
    layer.finish_load()

    def run_fp16():
        layer(
            inputs=inputs,
            input_scale=input_scale,
            block_shape=(64, 256, 32),
            warp_shape=(64, 64, 32),
            use_warp_spec=False,
            use_tma=False,
            use_cp_async=True,
            use_mbarrier=False,
            num_stages=3,
            num_ctas_per_sm=2,
        )

    def run_fp8():
        layer(
            inputs=inputs,
            input_scale=input_scale,
            block_shape=(64, 128, 128),
            warp_shape=(64, 32, 128),
            use_warp_spec=False,
            use_tma=False,
            use_cp_async=True,
            use_mbarrier=False,
            num_stages=4,
            num_ctas_per_sm=3,
        )

    if a_dtype == dtypes.float16:
        t = triton.testing.do_bench_cudagraph(run_fp16, rep=50)
    else:
        t = triton.testing.do_bench_cudagraph(run_fp8, rep=50)

    tflops = 8192 * 8192 * 8192 * 2 / t / 1e9
    tflops = round(tflops, 2)
    print(group_size, a_dtype, b_dtype, tflops)


for group_size in [0, 128]:
    for a_dtype in a_dtypes:
        for b_dtype in b_dtypes:
            run_example(group_size, a_dtype, b_dtype)


# H20 TFLOPS: 148 (FP16) / 296 (FP8)
# 0 float16 uint3 140.77
# 0 float16 uint4 140.71
# 0 float16 uint5 140.77
# 0 float8e4m3 uint3 282.12
# 0 float8e4m3 uint4 282.16
# 0 float8e4m3 uint5 281.54
# 128 float16 uint3 140.51
# 128 float16 uint4 140.51
# 128 float16 uint5 140.47
# 128 float8e4m3 uint3 275.24
# 128 float8e4m3 uint4 275.48
# 128 float8e4m3 uint5 275.24
