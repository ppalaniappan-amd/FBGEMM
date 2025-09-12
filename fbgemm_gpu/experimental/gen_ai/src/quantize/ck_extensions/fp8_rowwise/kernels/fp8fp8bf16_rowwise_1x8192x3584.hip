
#include "fp8_rowwise_common_testing.h"

at::Tensor
fp8fp8bf16_rowwise_1x8192x3584(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {

  // The smallest kernel we have available. Works well for memory bound shapes.
  return f8f8bf16_rowwise_wrapper<
      128,// block size
      16, // Mblock
      32, // nblock
      128, // kblock
      16, // wave tile M
      16, // wave tile N
      1, // wave map M 
      1, // wave map N 
      S<8, 16, 1>, // ABLOCK_TRANSFER
      S<8, 16, 1>, // BBLOCK_TRANSFER
      S<1, 16, 1, 8>, // CBLOCK_TRANSFER
      S<4, 4, 1>, // CBLOCK_SPV
      1, // CSHUFFLE_MX_PER_WAVE_PERSHUFFLE
      1, // CSHUFFLE_NX_PER_WAVE_PERSHUFFLE
      ck::BlockGemmPipelineScheduler::Intrawave, // loop sched
      ck::BlockGemmPipelineVersion::v1>(XQ, WQ, x_scale, w_scale, Y, 4);
}

