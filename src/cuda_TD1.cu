
#include "cuda_TD1.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void nx2_plus_ny(int n, int m, int* TabX, int* TabY, int* TabOut, int TailleTab)
{
	int idGlobal =
		threadIdx.x // <= 0 - 511
		+ blockIdx.x * blockDim.x // x * 512
		+ blockIdx.y * blockDim.x * gridDim.x; // y * 512 * 1000

	if (idGlobal < TailleTab)
		TabOut[idGlobal] = n * TabX[idGlobal] * TabX[idGlobal] + m * TabY[idGlobal];
}

} // namespace

void nx2_plus_ny_GPU()
{
    int n;
	int m;
	std::cin >> n;
	std::cin >> m;
    int TailleTab;
	std::cin >> TailleTab;
	fdim Grille(1000, 1000); // <= Définir la taille de la grille en fonction de la taille du tableau
	int* TabX = new int[TailleTab];
	int* TabY = new int[TailleTab];
	int* TabOut = new int[TailleTab];
	// random fill tabX and tabY
	for (int i = 0; i < TailleTab; i++)
	{
		TabX[i] = rand() % 100;
		TabY[i] = rand() % 100;
	}
	nx2_plus_ny_GPU << <Grille, 512 >> > (n, m, TabX, TabY, TabOut, TailleTab);
	delete[] TabX;
	delete[] TabY;
	delete[] TabOut;

	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
}
