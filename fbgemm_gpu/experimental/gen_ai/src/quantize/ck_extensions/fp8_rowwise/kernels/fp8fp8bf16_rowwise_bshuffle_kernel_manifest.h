#pragma once
#include <cstdlib>

#include <ATen/ATen.h>


at::Tensor
fp8fp8bf16_rowwise_bpreshuffle_256x12x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_bpreshuffle_256x16x128x256_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_1x8192x3584(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_1x8192x1024(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);
