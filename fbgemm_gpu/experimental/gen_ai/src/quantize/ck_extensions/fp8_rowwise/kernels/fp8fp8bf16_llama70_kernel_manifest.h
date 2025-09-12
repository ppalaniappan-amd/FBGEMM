#pragma once
#include <cstdlib>

#include <ATen/ATen.h>


at::Tensor
fp8fp8bf16_rowwise_1x1280x8192(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_1x7168x8192(
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


at::Tensor
fp8fp8bf16_rowwise_128x1280x8192(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_2048x7168x8192(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);


at::Tensor
fp8fp8bf16_rowwise_2048x8192x3584(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);