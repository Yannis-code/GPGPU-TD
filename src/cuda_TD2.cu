
#include "cuda_TD2.hpp"

#include <cuda_runtime.h>

#include <iostream>

namespace {

__global__ void convert_greyscale(char*** Img, int width, int height, char** ImgOut)
{
	int idX =
		threadId.x // 0-32
		+ blockIdx.x * blockDim.x; // x * 512

	int idY =
		threadId.y // 0-32
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

	fdim Bloc(32, 32);
	fdim Grille(1000, 1000);

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

	convert_greyscale << <Grille, Bloc >> > (Img, width, height, ImgOut);

	auto err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cerr << "Error on runOnGPU: " << cudaGetErrorString(err) << std::endl;
	}
}
