/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
 
#include "fp8_rowwise_common_testing.h"

at::Tensor
fp8fp8bf16_rowwise_bpreshuffle_256x12x64x256_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {

  // The smallest kernel we have available. Works well for memory bound shapes.
  return f8f8bf16_rowwise_wrapper<
      256,
      16,
      64,
      256,
      16,
      16,
      1,
      1,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 16, 1, 16>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 8);
}

