
#include "cuda_TP2.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void convert_greyscale(char*** Img, int width, int height, char** ImgOut)
{
	int idX =
		threadIdx.x // 0-32
		+ blockIdx.x * blockDim.x; // x * 512

	int idY =
		threadIdx.y // 0-32
		+ blockIdx.y * blockDim.y; // y * 512

	if (idX < width && idY < height)
	{
		ImgOut[idY][idX] = 0.299 * Img[idY][idX][0] + 0.587 * Img[idY][idX][1] + 0.114 * Img[idY][idX][2];
	}
}

} // namespace

void convert_greyscale_GPU()
{
	int height;
	int width;

	std::cin >> height;
	std::cin >> width;

	dim3 Bloc(32, 32);
	dim3 Grille(1000, 1000);

	char*** Img = new char** [height];
	char** ImgOut = new char* [height];
	for (int i = 0; i < height; i++)
	{
		Img[i] = new char* [width];
		ImgOut[i] = new char[width];
		for (int j = 0; j < width; j++)
		{
			Img[i][j] = new char[3];
			for (int k = 0; k < 3; k++)
			{
				Img[i][j][k] = i * k % 256;
			}
		}
	}

	char* Img = new char[height * width * 4];
	char* ImgOut = new char[height * width];
	for (int i = 0; i < height * width * 4; i++)
	{
		Img[i] = i % 256;
	}
	
	char* ImgGPU;
	char* ImgOutGPU;
	auto err = cudaMalloc(&ImgGPU, height * width * 4 * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMalloc(&ImgOutGPU, height * width * sizeof(char));
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	err = cudaMemcpy(ImgGPU, Img, height * width * 4 * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::
	}
	err = cudaMemcpy(ImgOutGPU, ImgOut, height * width * sizeof(char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}

	convert_greyscale << <Grille, Bloc >> > (Img, width, height, ImgOut);

	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
}
