
#include "cuda_TP3_glob.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void check_prime_glob(int* InOutTab, int sizeTab, int* PrimeTab)
{
	int idGlobal =
		threadIdx.x
		+ blockIdx.x * blockDim.x;

	if (idGlobal < sizeTab)
	{
		InOutTab[idGlobal] = 0;
		for (int i = 0; i < sizeTab; i++)
		{
			if (InOutTab[idGlobal] == PrimeTab[i])
			{
				InOutTab[idGlobal] = 1;
				break;
			}
		}
	}
}

} // namespace

bool isPrimeGlob(int number) {

	if (number < 2) return false;
	if (number == 2) return true;
	if (number % 2 == 0) return false;
	for (int i = 3; (i * i) <= number; i += 2) {
		if (number % i == 0) return false;
	}
	return true;
}

void check_prime_glob_GPU()
{
	int SizeTab;
	int RandomMax;

	std::cout << "Enter SizeTab : ";
	std::cin >> SizeTab;
	std::cout << std::endl;

	std::cout << "Enter RandomMax : ";
	std::cin >> RandomMax;
	std::cout << std::endl;


	dim3 Bloc(32, 32);
	dim3 Grille(1000, 1000);

	int* TabRandom = new int[SizeTab];
	for (int i = 0; i < SizeTab; i++)
	{
		TabRandom[i] = rand() % RandomMax;
	}

	// Tab of all possible prime numbers < randomMax
	int* TabPrime = new int[sqrt(RandomMax) + 1];
	int j = 0;
	for (int i = 2; i < RandomMax; i++)
	{
		if (isPrimeGlob(i))
		{
			TabPrime[j] = i;
			j++;
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int* TabRandomGPU;
	int* TabPrimeGPU;
	auto err = cudaMalloc(&TabRandomGPU, SizeTab * sizeof(int));
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaMalloc(&TabPrimeGPU, SizeTab * sizeof(int));
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	err = cudaMemcpy(TabRandomGPU, TabRandom, SizeTab * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaMemcpy(TabPrimeGPU, TabPrime, SizeTab * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	cudaEventRecord(start);
	check_prime_glob << <Grille, Bloc >> > (TabRandomGPU, SizeTab, TabPrimeGPU);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << "Elapsed time: " << elapsed << "ms" << std::endl;

	// destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	err = cudaGetLastError();
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	err = cudaMemcpy(TabRandom, TabRandomGPU, SizeTab * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	delete[] TabRandom;
	delete[] TabPrime;
	err = cudaFree(TabRandomGPU);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaFree(TabPrimeGPU);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

}
