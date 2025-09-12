
#include "fp8_rowwise_common_testing.h"

at::Tensor
fp8fp8bf16_rowwise_2048x7168x8192(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {

  // The smallest kernel we have available. Works well for memory bound shapes.
  return f8f8bf16_rowwise_wrapper<
      256,// block size
      256, // Mblock
      224, // nblock
      128, // kblock
      16, // wave tile M
      16, // wave tile N
      8, // wave map M 
      7, // wave map N 
      S<8, 32, 1>, // ABLOCK_TRANSFER
      S<8, 32, 1>, // BBLOCK_TRANSFER
      S<1, 64, 1, 4>, // CBLOCK_TRANSFER
      S<8, 8, 1>, // CBLOCK_SPV
      2, // CSHUFFLE_MX_PER_WAVE_PERSHUFFLE
      1, // CSHUFFLE_NX_PER_WAVE_PERSHUFFLE
      ck::BlockGemmPipelineScheduler::Intrawave, // loop sched
      ck::BlockGemmPipelineVersion::v3>(XQ, WQ, x_scale, w_scale, Y, 1);
}

