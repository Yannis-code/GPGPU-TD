
#include "cpp_TD1.hpp"
#include "cuda_TD1.hpp"
#include "cpp_TD2.hpp"
#include "cuda_TD2.hpp"

int main(int, char*[])
{
  runOnCPU();
  runOnGPU();
  return 0;
}