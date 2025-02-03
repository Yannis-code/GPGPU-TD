
#include "cuda_TP3_glob.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void check_prime_shrd(int* InOutTab, int InSizeTab, int* InPrimeTab, int InNbPrimeTab)
{
	extern __shared__ int PrimeTabShrd[];
	int idGlobal =
		threadIdx.x
		+ blockIdx.x * blockDim.x;

	// copy PrimeTab to shared memory
	if (threadIdx.x < InNbPrimeTab)
	{
		PrimeTabShrd[threadIdx.x] = InPrimeTab[threadIdx.x];
	}
	__syncthreads();

	if (idGlobal < InSizeTab)
	{
		InOutTab[idGlobal] = 0;
		for (int i = 0; i < InNbPrimeTab; i++)
		{
			if (InOutTab[idGlobal] == PrimeTabShrd[i])
			{
				InOutTab[idGlobal] = 1;
				break;
			}
		}
	}
}

} // namespace

bool isPrimeShrd(int number) {

	if (number < 2) return false;
	if (number == 2) return true;
	if (number % 2 == 0) return false;
	for (int i = 3; (i * i) <= number; i += 2) {
		if (number % i == 0) return false;
	}
	return true;
}

void check_prime_shrd_GPU()
{
	// Récupérer les valeurs de SizeTab et RandomMax
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
	int SizeTabPrime = sqrt(RandomMax) + 1;
	int* TabPrime = new int[SizeTabPrime];
	int j = 0;
	for (int i = 2; i < RandomMax; i++)
	{
		if (isPrimeShrd(i))
		{
			TabPrime[j] = i;
			j++;
		}
	}

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

	cudaEvent_t start, stop;
	err = cudaEventCreate(&start);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaEventCreate(&stop);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	// run kernel and measure time elapsed
	err = cudaEventRecord(start);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	check_prime_shrd <<<Grille, Bloc, SizeTabPrime * sizeof(int) >>> (TabRandomGPU, SizeTab, TabPrimeGPU, SizeTabPrime);
	
	err = cudaGetLastError();
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	err = cudaEventRecord(stop);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	float elapsed;
	err = cudaEventElapsedTime(&elapsed, start, stop);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	std::cout << "Elapsed time: " << elapsed << "ms" << std::endl;

	// destroy events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	

	err = cudaMemcpy(TabRandom, TabRandomGPU, SizeTab * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;

	delete[] TabRandom;
	delete[] TabPrime;
	err = cudaFree(TabRandomGPU);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	err = cudaFree(TabPrimeGPU);
	if (err != cudaSuccess) std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
}
