/*

K- Means Clustering in GPU
@Author Midhun Harikumar
@Date   10/23/2015


Function performs K- Means clustering on different Data point and Cluster inputs
for the CUDA capable NVIDIA GPU



*/





#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "KMeans.h"


__constant__ Vector2 clusterD[3];








__global__ void findDistances(Datapoint *dataD, int n, int k){

	double dist[3] = { 5000, 500, 5000 };
	double distmin = 5000;
	int minval = 0;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int tid = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	//int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	/*int tid = blockIdx.x * blockDim.x * blockDim.y
		+ threadIdx.y * blockDim.x + threadIdx.x;*/

	if (tid < n){

		dist[0] = clusterD[0].distSq(dataD[tid].p);
		dist[1] = clusterD[1].distSq(dataD[tid].p);
		dist[2] = clusterD[2].distSq(dataD[tid].p);

		for (int i = 0; i < 3; i++){

			if (distmin>dist[i]){
				minval = i;

				distmin = dist[i];
			}


		}

		if (minval != dataD[tid].cluster){
			dataD[tid].cluster = minval;

			dataD[tid].altered = true;
		}
		else{
			dataD[tid].altered = false;
		}




	}//if loopClose




}








bool KMeansGPU(Datapoint* data, long n, Vector2* clusters, int k){


	cudaError_t status;

	int data_bytes = sizeof(Datapoint)*n;
	int cluster_bytes = sizeof(Vector2)*k;
	int altered_data_cnt = 1555;
	int clusterCount = 0;
	Datapoint* dataD;




	// Allocate memory on device
	cudaMalloc((void**)&dataD, data_bytes);
	cudaMemcpyToSymbol(clusterD, clusters, cluster_bytes);
	cudaMemcpyFromSymbol(clusters, clusterD, cluster_bytes);

	// Copy data to allocated memory	
	cudaMemcpy(dataD, data, data_bytes, cudaMemcpyHostToDevice);

	// Set Grid and block Dimensions
	dim3 dimblock(16, 16, 1);
	dim3 dimgrid;
	dimgrid.x = ceil(sqrt((float)n / 256));
	dimgrid.y = ceil(sqrt((float)n / 256));

	// Check state	
	while (altered_data_cnt != 0){
		altered_data_cnt = 0;
		// Find distances 
		findDistances << <dimgrid, dimblock >> >(dataD, n, k);
		cudaThreadSynchronize();


		// Copy Data back
		cudaMemcpy(data, dataD, data_bytes, cudaMemcpyDeviceToHost);



		for (int i = 0; i < k; i++){

			for (int j = 0; j < n; j++){

				if (data[j].cluster == i){

					clusters[i].x += data[j].p.x;
					clusters[i].y += data[j].p.y;
					clusterCount++;
				}

			}
			if (clusterCount != 0){
				clusters[i].x /= clusterCount;
				clusters[i].y /= clusterCount;
			}
			clusterCount = 0;
		}

		for (int i = 0; i < n; i++){

			if (data[i].altered){
				altered_data_cnt++;
				data[i].altered = false;
			}

		}

		cudaMemcpy(dataD, data, data_bytes, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(clusterD, clusters, cluster_bytes);

	}









	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed: " << cudaGetErrorString(status) <<
			std::endl;
		cudaFree(dataD);


		return false;
	}




	cudaFree(dataD);


	return true;
}