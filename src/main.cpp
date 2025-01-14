
#include "cpp_test_compil.hpp"
#include "cuda_test_compil.hpp"

int main(int, char*[])
{
  runOnCPU();
  runOnGPU();
  return 0;
}