
#include "fp8_rowwise_common_testing.h"

at::Tensor
fp8fp8bf16_rowwise_1x7168x8192(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {

  // The smallest kernel we have available. Works well for memory bound shapes.
  return f8f8bf16_rowwise_wrapper<
      256,// block size
      16, // Mblock
      128, // nblock
      256, // kblock
      16, // wave tile M
      16, // wave tile N
      1, // wave map M 
      2, // wave map N 
      S<16, 16, 1>, // ABLOCK_TRANSFER
      S<16, 16, 1>, // BBLOCK_TRANSFER
      S<1, 16, 1, 16>, // CBLOCK_TRANSFER
      S<8, 8, 1>, // CBLOCK_SPV
      1, // CSHUFFLE_MX_PER_WAVE_PERSHUFFLE
      2, // CSHUFFLE_NX_PER_WAVE_PERSHUFFLE
      ck::BlockGemmPipelineScheduler::Intrawave, // loop sched
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 8);
}

