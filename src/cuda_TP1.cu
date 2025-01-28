
#include "cuda_TP1.hpp"

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
	// Récupérer les valeurs de n, m et TailleTab
    int n;
	int m;
    int TailleTab;
	std::cout << "Enter n : ";
	std::cin >> n;
	std::cout << std::endl;
	std::cout << "Enter m : ";
	std::cin >> m;
	std::cout << std::endl;
	std::cout << "Enter TailleTab : ";
	std::cin >> TailleTab;
	std::cout << std::endl;

	// Définition de la grille
	dim3 Grille(1000, 1000); // <= Définir la taille de la grille en fonction de la taille du tableau

	// Allocation des tableaux CPU
	int* TabX = new int[TailleTab];
	int* TabY = new int[TailleTab];
	int* TabOut = new int[TailleTab];

	// Remplissage des tableaux CPU avec des valeurs aléatoires
	for (int i = 0; i < TailleTab; i++)
	{
		TabX[i] = rand() % 100;
		TabY[i] = rand() % 100;
	}

	// Allocation des tableaux GPU
	int* TabXGPU;
	int* TabYGPU;
	int* TabOutGPU;
	auto err = cudaMalloc(&TabXGPU, TailleTab * sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMalloc(&TabYGPU, TailleTab * sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMalloc(&TabOutGPU, TailleTab * sizeof(int));
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	// Affichage les 5 premières valeurs de TabX et TabY
	std::cout << std::endl << "5 premières valeurs de TabX et TabY"  << std::endl;
	std::cout << "TabX: ";
	for (int i = 0; i < 5; i++)
	{
		std::cout << TabX[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "TabY: ";
	for (int i = 0; i < 5; i++)
	{
		std::cout << TabY[i] << " ";
	}
	std::cout << std::endl << "=================" << std::endl;

	// Affichage les 5 dernière valeurs de TabX et TabY
	std::cout << "5 dernière valeurs de TabX et TabY" << std::endl;
	std::cout << "TabX: ";
	for (int i = TailleTab - 5; i < TailleTab; i++)
	{
		std::cout << TabX[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "TabY: ";
	for (int i = TailleTab - 5; i < TailleTab; i++)
	{
		std::cout << TabY[i] << " ";
	}
	std::cout << std::endl;

	// Copie des tableaux CPU vers les tableaux GPU
	err = cudaMemcpy(TabXGPU, TabX, TailleTab * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMemcpy(TabYGPU, TabY, TailleTab * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMemcpy(TabOutGPU, TabOut, TailleTab * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	// Appel du kernel 
	nx2_plus_ny << <Grille, 512 >> > (n, m, TabXGPU, TabYGPU, TabOutGPU, TailleTab);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	// Copie du tableau GPU vers le tableau CPU
	err = cudaMemcpy(TabOut, TabOutGPU, TailleTab * sizeof(int), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	// Libération de la mémoire des tableaux GPU
	err = cudaFree(TabXGPU);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaFree(TabYGPU);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaFree(TabOutGPU);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	// Affichage les 5 premières valeurs de TabOut
	std::cout << std::endl << "5 premières valeurs de TabOut" << std::endl;
	std::cout << "TabOut: ";
	for (int i = 0; i < 5; i++)
	{
		std::cout << TabOut[i] << " ";
	}
	std::cout << std::endl << "=================" << std::endl;

	// Affichage les 5 dernière valeurs de TabOut
	std::cout << "5 dernière valeurs de TabOut" << std::endl;
	std::cout << "TabOut: ";
	for (int i = TailleTab - 5; i < TailleTab; i++)
	{
		std::cout << TabOut[i] << " ";
	}
	std::cout << std::endl;

	// Libération de la mémoire des tableaux CPU
	delete[] TabX;
	delete[] TabY;
	delete[] TabOut;
}
