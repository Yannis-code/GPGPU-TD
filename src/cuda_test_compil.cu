
#include "cuda_test_compil.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void doNothing()
{
  [[maybe_unused]] int i = blockIdx.x * blockDim.x + threadIdx.x;
  [[maybe_unused]] int j = blockIdx.y * blockDim.y + threadIdx.y;
}

} // namespace

void runOnGPU()
{
  doNothing<<<1, 10>>>();
  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
  }
}
