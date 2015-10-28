/*

K- Means Clustering in CPU
@Author Midhun Harikumar
@Date   10/23/2015


Function performs K- Means clustering on different Data point and Cluster inputs



*/




#include<iostream>
#include"KMeans.h"

#define DATA_SIZE (1<<24)

float dist[3] = { 0, 0, 0 };


int findMin(){
	float min = dist[0];
		int minpos=0;

	for(int i=1;i<3;i++){

		if(min>dist[i]){
			minpos = i;
			min = dist[i];
		}

	}


	return minpos;
}




void KMeansCPU(Datapoint* data, long n, Vector2* clusters, int k){

	
	int clusterCount=0;
	int tempmin;
	int altered_data_cnt = 1555;
	while (altered_data_cnt !=0){
		altered_data_cnt = 0;
		for (int i = 0; i < n; i++){

			for (int j = 0; j < k; j++){
				dist[j] = clusters[j].distSq(data[i].p);
			}

			tempmin = findMin();
			if (tempmin != data[i].cluster){
				data[i].cluster = tempmin;
				data[i].altered = true;
			}
			else{
				data[i].altered = false;
				
			}


		}

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

	}


	

}	







