/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "fp8_rowwise_common_testing.h"

// Config for shape 1X1280X8192
at::Tensor
fp8fp8bf16_rowwise_256x256x192x128_16x16_8x6_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  return f8f8bf16_rowwise_wrapper<
      256,// block size
      16, // Mblock
      64, // nblock
      256, // kblock
      16, // wave tile M
      16, // wave tile N
      1, // wave map M 
      1, // wave map N 
      S<16, 16, 1>, // ABLOCK_TRANSFER
      S<16, 16, 1>, // BBLOCK_TRANSFER
      S<1, 16, 1, 16>, // CBLOCK_TRANSFER
      S<4, 4, 1>, // CBLOCK_SPV
      2, // CSHUFFLE_MX_PER_WAVE_PERSHUFFLE
      1, // CSHUFFLE_NX_PER_WAVE_PERSHUFFLE
      ck::BlockGemmPipelineScheduler::Intrawave, // loop sched
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 1);
}
