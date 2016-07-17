#include "Population.h"
#include "quantities.h"
#include "conv_common.h"

////////////////////////////////////////////////////////////////////////////////
// Utilities and system includes
////////////////////////////////////////////////////////////////////////////////

#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
#include <Image.h>

#include <string>
#include <fstream>
#include <iostream>

#include <npp.h>
#include <nppdefs.h>
#include <cublas_v2.h>

#include <helper_cuda.h>
#include <shrQATest.h>
#include <cuda_runtime.h>
#include <helper_functions.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/stream_accessor.hpp"

// Constructor
/////////////////////////////////////////////////////////////////////////////////////////////

Population::Population() {
	imageW = 0;
	imageH = 0;
}

// General utility functions
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::call(std::string filename, std::string functionName, void (Population::*funcToCall)(Filters *,float *,bool), int iterations, bool warmUpIt, StopWatchInterface *timer, bool bEvent, Filters *f) {
	
	float tot_elapsed_time = 0, elapsed_time = 0;   // timing variable
	double timeIN = 0, timeOUT = 0;

	if (warmUpIt) {
		for (int i = 0; i < iterations; i++) {
			if (i == 1) {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				timeIN = 0;
				tot_elapsed_time = 0;
			} 
			(this->*funcToCall)(f, &elapsed_time, bEvent);
			tot_elapsed_time += elapsed_time;
		}
	} else {
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		for (int i = 0; i < iterations; i++) {
			(this->*funcToCall)(f, &elapsed_time, bEvent);
			tot_elapsed_time += elapsed_time;
		}
    }

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	timeOUT = sdkGetTimerValue(&timer);
	std::cout << "FUNCTION: " << functionName << std::endl;
	printf("timers: \t timeOUT = %.5f ms \n \t\t timeIN = %.5f ms \n", timeOUT / (double)iterations, timeIN / (double)iterations);
	printf("events: \t elapsed_time = %.5f ms \n\n", tot_elapsed_time / (double)iterations);
}
void Population::callAll(std::string filename, std::string *functionName, void (Population::**funcToCall)(Filters *,float *,bool), int _nFunc, int iterations, bool warmUpIt, StopWatchInterface *timer, bool bEvent, bool bTimer, Filters *f) {
	
	float *timeFuncEvent, elapsed_time = 0;
	float timeTotSumFuncEvent = 0;
	
	double *timeFuncTimer;
	double timeTotTimer = 0;

	if (bTimer) {

		timeFuncTimer = new double[_nFunc];
		for (int j=0; j<_nFunc; j++) 
				timeFuncTimer[j] = 0;

		if (bEvent) {
		
			timeFuncEvent = new float[_nFunc];
			for (int j=0; j<_nFunc; j++) 
				timeFuncEvent[j] = 0;

			if (warmUpIt)
				for (int i = -1; i < iterations; i++) {
					if (i == 0) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&timer);
						sdkStartTimer(&timer);
						for (int j=0; j<_nFunc; j++) {
							timeFuncEvent[j] = 0;
							timeFuncTimer[j] = 0;
						}
					} 
					for (int j=0; j<_nFunc; j++) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&funcTimer);
						sdkStartTimer(&funcTimer);
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						checkCudaErrors(cudaDeviceSynchronize());
						sdkStopTimer(&funcTimer);
						timeFuncTimer[j] += sdkGetTimerValue(&funcTimer);
						timeFuncEvent[j] += elapsed_time;
					}
				}
			else {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				for (int i = 0; i < iterations; i++)
					for (int j=0; j<_nFunc; j++) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&funcTimer);
						sdkStartTimer(&funcTimer);
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						checkCudaErrors(cudaDeviceSynchronize());
						sdkStopTimer(&funcTimer);
						timeFuncTimer[j] += sdkGetTimerValue(&funcTimer);
						timeFuncEvent[j] += elapsed_time;
					}
			}

			checkCudaErrors(cudaDeviceSynchronize());
			sdkStopTimer(&timer);
			timeTotTimer = sdkGetTimerValue(&timer);
			printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
			for (int j=0; j<nFunc; j++) 
				timeTotSumFuncEvent += timeFuncEvent[j];
			printf("\t timeTotSumFuncEvent = %.5f ms \n", timeTotSumFuncEvent / (double)iterations);
			for (int j=0; j<nFunc; j++) {
				std::cout << "FUNCTION[" << j << "]: " << functionName[j] << std::endl;
				printf("\t\t timeFuncEvent[%d] = %.5f ms \n\n", j, timeFuncEvent[j] / (double)iterations);
				printf("\t\t timeFuncTimer[%d] = %.5f ms \n\n", j, timeFuncTimer[j] / (double)iterations);
			}

			std::ofstream f_output;
			f_output.open(filename);
			
			f_output << timeTotTimer / (double)iterations << "  ";
			f_output << timeTotSumFuncEvent / (double)iterations << "  ";
			for (int j=0; j<nFunc; j++) {
				f_output << timeFuncEvent[j] / (double)iterations << "  ";
				f_output << timeFuncTimer[j] / (double)iterations << "  ";
			}
		
			f_output.close();
		
			delete[] timeFuncEvent;
		}
		else {

			if (warmUpIt)
				for (int i = -1; i < iterations; i++) {
					if (i == 0) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&timer);
						sdkStartTimer(&timer);
						for (int j=0; j<_nFunc; j++) 
							timeFuncTimer[j] = 0;
						} 
					for (int j=0; j<_nFunc; j++) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&funcTimer);
						sdkStartTimer(&funcTimer);
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						checkCudaErrors(cudaDeviceSynchronize());
						sdkStopTimer(&funcTimer);
						timeFuncTimer[j] += sdkGetTimerValue(&funcTimer);
					}
				}
			else {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				for (int i = 0; i < iterations; i++)
					for (int j=0; j<_nFunc; j++) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&funcTimer);
						sdkStartTimer(&funcTimer);
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						checkCudaErrors(cudaDeviceSynchronize());
						sdkStopTimer(&funcTimer);
						timeFuncTimer[j] += sdkGetTimerValue(&funcTimer);
					}
			}

			checkCudaErrors(cudaDeviceSynchronize());
			sdkStopTimer(&timer);
			timeTotTimer = sdkGetTimerValue(&timer);
			printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
			for (int j=0; j<nFunc; j++) {
				std::cout << "FUNCTION[" << j << "]: " << functionName[j] << std::endl;
				printf("\t\t timeFuncTimer[%d] = %.5f ms \n\n", j, timeFuncTimer[j] / (double)iterations);
			}

			std::ofstream f_output;
			f_output.open(filename);
			
			f_output << timeTotTimer / (double)iterations << "  ";
			for (int j=0; j<nFunc; j++)
				f_output << timeFuncTimer[j] / (double)iterations << "  ";

			f_output.close();
		}
	} else {
		if (bEvent) {
			
			timeFuncEvent = new float[_nFunc];
			for (int j=0; j<_nFunc; j++) 
				timeFuncEvent[j] = 0;

			if (warmUpIt)
				for (int i = -1; i < iterations; i++) {
					if (i == 0) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&timer);
						sdkStartTimer(&timer);
						for (int j=0; j<_nFunc; j++)
							timeFuncEvent[j] = 0;
						} 
					for (int j=0; j<_nFunc; j++) {
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						timeFuncEvent[j] += elapsed_time;
					}
				}
			else {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				for (int i = 0; i < iterations; i++)
					for (int j=0; j<_nFunc; j++) {
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
						timeFuncEvent[j] += elapsed_time;
					}
			}

			checkCudaErrors(cudaDeviceSynchronize());
			sdkStopTimer(&timer);
			timeTotTimer = sdkGetTimerValue(&timer);
			printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
			for (int j=0; j<nFunc; j++) 
				timeTotSumFuncEvent += timeFuncEvent[j];
			printf("\t timeTotSumFuncEvent = %.5f ms \n", timeTotSumFuncEvent / (double)iterations);
			for (int j=0; j<nFunc; j++) {
				std::cout << "FUNCTION[" << j << "]: " << functionName[j] << std::endl;
				printf("\t\t timeFuncEvent[%d] = %.5f ms \n\n", j, timeFuncEvent[j] / (double)iterations);
			}

			std::ofstream f_output;
			f_output.open(filename);
			
			f_output << timeTotTimer / (double)iterations << "  ";
			f_output << timeTotSumFuncEvent / (double)iterations << "  ";
			for (int j=0; j<nFunc; j++) 
				f_output << timeFuncEvent[j] / (double)iterations << "  ";
		
			f_output.close();
		
			delete[] timeFuncEvent;
		}
		else {

			if (warmUpIt)
				for (int i = -1; i < iterations; i++) {
					if (i == 0) {
						checkCudaErrors(cudaDeviceSynchronize());
						sdkResetTimer(&timer);
						sdkStartTimer(&timer);
						} 
					for (int j=0; j<_nFunc; j++)
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
				}
			else {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				for (int i = 0; i < iterations; i++)
					for (int j=0; j<_nFunc; j++)
						(this->*funcToCall[j])(f, &elapsed_time, bEvent);
			}

			checkCudaErrors(cudaDeviceSynchronize());
			sdkStopTimer(&timer);
			timeTotTimer = sdkGetTimerValue(&timer);
			printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);

			std::ofstream f_output;
			f_output.open(filename);
			
			f_output << timeTotTimer / (double)iterations << "  ";
		
			f_output.close();
		}
	}
}

void Population::call(std::string filename, std::string functionName, void (Population::*funcToCall)(cv::Mat &,float *,bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, cv::Mat &data) {

	float timeFuncEvent = 0, elapsed_time = 0;   // timing variable
	double timeTotTimer = 0;

	if (warmUpIt) {
		for (int i = -1; i < iterations; i++) {
			if (i == 0) {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				timeFuncEvent = 0;
			} 
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
	} else {
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		for (int i = 0; i < iterations; i++) {
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
    }

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	timeTotTimer = sdkGetTimerValue(&timer);
	printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
	if (bEvent) {
		std::cout << "FUNCTION: " << functionName << std::endl;
		printf("\t\t timeFuncEvent = %.5f ms \n\n", timeFuncEvent / (double)iterations);
	}
	std::ofstream f_output;
    f_output.open(filename);
			
	f_output << timeTotTimer / (double)iterations << "  ";
	if (bEvent) 
		f_output << timeFuncEvent / (double)iterations << "  ";
	
	f_output.close();
}
void Population::call(std::string filename, std::string functionName, void (Population::*funcToCall)(cv::Mat *,float *,bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, cv::Mat *data) {

	float timeFuncEvent = 0, elapsed_time = 0;   // timing variable
	double timeTotTimer = 0;

	if (warmUpIt) {
		for (int i = -1; i < iterations; i++) {
			if (i == 0) {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				timeFuncEvent = 0;
			} 
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
	} else {
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		for (int i = 0; i < iterations; i++) {
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
    }

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	timeTotTimer = sdkGetTimerValue(&timer);
	printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
	if (bEvent) {
		std::cout << "FUNCTION: " << functionName << std::endl;
		printf("\t\t timeFuncEvent = %.5f ms \n\n", timeFuncEvent / (double)iterations);
	}
	std::ofstream f_output;
    f_output.open(filename);
			
	f_output << timeTotTimer / (double)iterations << "  ";
	if (bEvent) 
		f_output << timeFuncEvent / (double)iterations << "  ";
	
	f_output.close();
}
void Population::call(std::string filename, std::string functionName, void (Population::*funcToCall)(float **,int,int,float *,bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, float **data,int imageW,int imageH) {

	float timeFuncEvent = 0, elapsed_time = 0;   // timing variable
	double timeTotTimer = 0;

	if (warmUpIt) {
		for (int i = -1; i < iterations; i++) {
			if (i == 0) {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				timeFuncEvent = 0;
			} 
			(this->*funcToCall)(data, imageW, imageH, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
	} else {
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		for (int i = 0; i < iterations; i++) {
			(this->*funcToCall)(data, imageW, imageH, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
    }

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	timeTotTimer = sdkGetTimerValue(&timer);
	printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
	if (bEvent) {
		std::cout << "FUNCTION: " << functionName << std::endl;
		printf("\t\t timeFuncEvent = %.5f ms \n\n", timeFuncEvent / (double)iterations);
	}
	std::ofstream f_output;
    f_output.open(filename);
			
	f_output << timeTotTimer / (double)iterations << "  ";
	if (bEvent) 
		f_output << timeFuncEvent / (double)iterations << "  ";
	
	f_output.close();
}
void Population::call(std::string filename, std::string functionName, void (Population::*funcToCall)(float *,float *,bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, float *data) {

	float timeFuncEvent = 0, elapsed_time = 0;   // timing variable
	double timeTotTimer = 0;

	if (warmUpIt) {
		for (int i = -1; i < iterations; i++) {
			if (i == 0) {
				checkCudaErrors(cudaDeviceSynchronize());
				sdkResetTimer(&timer);
				sdkStartTimer(&timer);
				timeFuncEvent = 0;
			} 
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
	} else {
		checkCudaErrors(cudaDeviceSynchronize());
		sdkResetTimer(&timer);
		sdkStartTimer(&timer);
		for (int i = 0; i < iterations; i++) {
			(this->*funcToCall)(data, &elapsed_time, bEvent);
			if (bEvent) 
				timeFuncEvent += elapsed_time;
		}
    }

	checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
	timeTotTimer = sdkGetTimerValue(&timer);
	printf("\t timeTotTimer = %.5f ms \n", timeTotTimer / (double)iterations);
	if (bEvent) {
		std::cout << "FUNCTION: " << functionName << std::endl;
		printf("\t\t timeFuncEvent = %.5f ms \n\n", timeFuncEvent / (double)iterations);
	}
	std::ofstream f_output;
    f_output.open(filename);
			
	f_output << timeTotTimer / (double)iterations << "  ";
	if (bEvent) 
		f_output << timeFuncEvent / (double)iterations << "  ";
	
	f_output.close();
}

// Image pair managing
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::mallocCPUImagePairCUDA(int _imageH, int _imageW) {
	imageW = _imageW;
	imageH = _imageH;

	src_hostCU = (float **)malloc(nEye * sizeof(float*));
	for (int eye=0; eye<nEye; eye++)
		src_hostCU[eye] = (float *)malloc(imageW * imageH * sizeof(float));
}
void Population::freeCPUImagePairCUDA() {
	for (int eye=0; eye<nEye; eye++)
		free(src_hostCU[eye]);
	free(src_hostCU);
}
void Population::mallocCPUImagePairOPENCV(int _imageH, int _imageW) { 
	imageW = _imageW;
	imageH = _imageH;
	src_hostCV = new cv::Mat[nEye];
	for (int eye=0; eye<nEye; eye++)
		src_hostCV[eye].create(imageH, imageW, CV_32FC1);
}
void Population::freeCPUImagePairOPENCV() {
	for (int eye=0; eye<nEye; eye++)
		src_hostCV[eye].release();
	delete [] src_hostCV;
}
void Population::mallocCPUResultCUDA(int _imageH, int _imageW) {
	
	dst_hostCU = (float *)malloc(nResults * imageW * imageH * sizeof(float));
}
void Population::freeCPUResultCUDA() {

	free(dst_hostCU);
}
void Population::mallocCPUResultOPENCV(int _imageH, int _imageW) { 
	
	dst_hostCV.create(imageH, nResults*imageW, CV_32FC1);
}
void Population::freeCPUResultOPENCV() {

	dst_hostCV.release();
}

void Population::loadImagePairCUDA(char** src, char* argv0) {
	
	std::string* sFilename = new std::string[nEye];
	for (int eye=0; eye<nEye; eye++) {
		char *fname = sdkFindFilePath(src[eye], argv0);
		if (fname)
			sFilename[eye] = fname;
		else 
			std::cout << "Unable to find: <" << fname << ">" << std::endl;

		std::ifstream infile(sFilename[eye].data(), std::ios_base::in);

		if (infile.good()) {
			std::cout << "Opened: <" << sFilename[eye].data() << "> successfully!" << std::endl;
			infile.close();
		} else {
			std::cout << "Unable to open: <" << sFilename[eye].data() << ">" << std::endl;
			infile.close();
		}
	}
    
    npp::ImageCPU_8u_C1 *NPPsrc_hostCU_8u = new npp::ImageCPU_8u_C1[nEye];
	for (int eye=0; eye<nEye; eye++) 
		npp::loadImage(sFilename[eye], NPPsrc_hostCU_8u[eye]);

	imageW = NPPsrc_hostCU_8u[0].width();
	imageH = NPPsrc_hostCU_8u[0].height();

	for (int eye=0; eye<nEye; eye++) 
		for (int i = 0; i <imageH; i++) 
			for(int j = 0; j < imageW; j++) 
				src_hostCU[eye][i*imageH+j] = (float)(*NPPsrc_hostCU_8u[eye].data(i,j));

	delete [] sFilename;
	delete [] NPPsrc_hostCU_8u;
}
void Population::loadImagePairOPENCV(std::string *src) {

	for (int eye=0; eye<nEye; eye++)
		src_hostCV[eye] = cv::imread(src[eye], CV_LOAD_IMAGE_GRAYSCALE);

	imageW = src_hostCV[0].cols;
    imageH = src_hostCV[0].rows;

	for (int eye=0; eye<nEye; eye++)
		if (src_hostCV[eye].data)
			std::cout << "Opened: <" << src[eye]<< "> successfully!" << std::endl;
		else 
			std::cout << "Unable to open: <" << src[eye] << ">" << std::endl;
	std::cout << std::endl;
}
void Population::copyCPUImagePairToCUDA(cv::Mat *_src_hostCV) {
    
    for (int eye=0; eye<nEye; eye++)
		for (int i = 0; i <imageH; i++) 
			for(int j = 0; j < imageW; j++) 
				src_hostCU[eye][i*imageH+j] = (float)*(_src_hostCV[eye].data + _src_hostCV[eye].step*i + j*_src_hostCV[eye].elemSize());
}

void Population::mallocGPUImagePairCUDA(int _imageH, int _imageW) {
	imageW = _imageW;
	imageH = _imageH;
	src_devCU = (float**) malloc(nEye * sizeof(float*));
	for (int eye=0; eye<nEye; eye++) 
        checkCudaErrors( cudaMalloc((void **)&src_devCU[eye],   imageW * imageH * sizeof(float)) );
	
}
void Population::freeGPUImagePairCUDA() { 
   for (int eye=0; eye<nEye; eye++) 
	   checkCudaErrors( cudaFree(src_devCU[eye]) );
   free(src_devCU);
}
void Population::mallocGPUImagePairOPENCV(int _imageH, int _imageW) {
	imageW = _imageW;
	imageH = _imageH;
	src_devCV = new cv::gpu::GpuMat[nEye];
	for (int eye=0; eye<nEye; eye++) 
		src_devCV[eye].create(imageH,imageW,CV_32FC1);

}
void Population::freeGPUImagePairOPENCV() {
	for (int eye=0; eye<nEye; eye++) 
		src_devCV[eye].release();
	delete [] src_devCV;
}

void Population::uploadImagePairCUDA(float **_src_hostCU, int _imageH, int _imageW, float* elapsed_time, bool bEvent) {

	if (bEvent)
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	imageW = _imageW;
	imageH = _imageH;

	for (int eye=0; eye<nEye; eye++)
		checkCudaErrors( cudaMemcpy(src_devCU[eye], _src_hostCU[eye], imageW * imageH * sizeof(float), cudaMemcpyHostToDevice) );

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::uploadImagePairOPENCV(cv::Mat *_src_hostCV, float* elapsed_time, bool bEvent) {
	
	if (bEvent)
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	imageW = _src_hostCV[0].cols;
    imageH = _src_hostCV[0].rows;

	for (int eye=0; eye<nEye; eye++) {
		src_devCV[eye].upload(_src_hostCV[eye]);
		src_devCV[eye].convertTo(src_devCV[eye],CV_32FC1);
		}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Cells' answers managing
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::mallocGPUCellOPENCV() {

	cell_devCV = (cv::gpu::GpuMat****)malloc(2 * sizeof(cv::gpu::GpuMat***)); // (L = 0, R = 1)
	for (int eye = 0; eye<nEye; eye++) { 
		cell_devCV[eye] = (cv::gpu::GpuMat***)malloc(2 * sizeof(cv::gpu::GpuMat**)); // (Re = 0, Im = 1)
		for (int i = 0; i<2; i++) {
			cell_devCV[eye][i] = (cv::gpu::GpuMat**)malloc(nOrient * sizeof(cv::gpu::GpuMat*)); 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) { // L: Re, Im are nOrient x 1
					cell_devCV[eye][i][theta] = new cv::gpu::GpuMat[1];
					cell_devCV[eye][i][theta][0].create(imageH,imageW,CV_32FC1);
				} else { // R: Re, Im are nOrient x nPhase
					cell_devCV[eye][i][theta] = new cv::gpu::GpuMat[nPhase];
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCV[eye][i][theta][phase].create(imageH,imageW,CV_32FC1);
				}
			} 
		}
	}		
}
void Population::freeGPUCellOPENCV() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) {
					cell_devCV[eye][i][theta][0].release();
				} else {
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCV[eye][i][theta][phase].release();
				}
			}
	free(cell_devCV);
}
void Population::mallocGPUCellCUDA() {

	cell_devCU = (float *****)malloc(2 * sizeof(float****)); // (L = 0, R = 1)
	for (int eye = 0; eye<nEye; eye++) { 
		cell_devCU[eye] = (float ****)malloc(2 * sizeof(float***));  // (Re = 0, Im = 1)
		for (int i = 0; i<2; i++) { 
			cell_devCU[eye][i] = (float ***)malloc(nOrient * sizeof(float**)); 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) { // L: Re, Im are nOrient x 1
					cell_devCU[eye][i][theta] = (float **)malloc(1 * sizeof(float*)); 
					checkCudaErrors( cudaMalloc((void **)&cell_devCU[eye][i][theta][0], imageW * imageH * sizeof(float)) );
				} else { // R: Re, Im are nOrient x nPhase
					cell_devCU[eye][i][theta] = (float **)malloc(nPhase * sizeof(float*)); 
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaMalloc((void **)&cell_devCU[eye][i][theta][phase], imageW * imageH * sizeof(float)) );
				}
			} 
		}
	}	
}
void Population::freeGPUCellCUDA() {

	for (int eye = 0; eye<nEye; eye++) 
		for (int i = 0; i<2; i++) 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) 
					checkCudaErrors( cudaFree(cell_devCU[eye][i][theta][0]) );
				else 
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaFree(cell_devCU[eye][i][theta][phase]) );
				}
	free(cell_devCU);
}

void Population::mallocGPUCellOPENCVoneOri() {

	cell_devCVoneOri = (cv::gpu::GpuMat***) malloc(nEye * sizeof(cv::gpu::GpuMat**));
	for (int eye=0; eye<nEye; eye++) {
		cell_devCVoneOri[eye] = (cv::gpu::GpuMat**) malloc(2 * sizeof(cv::gpu::GpuMat*));
		for (int i=0; i<2; i++) {
			if (eye == 0) {
					cell_devCVoneOri[eye][i] = new cv::gpu::GpuMat[1];
					cell_devCVoneOri[eye][i][0].create(imageH,imageW*nOrient,CV_32FC1);
				} else { 
					cell_devCVoneOri[eye][i] = new cv::gpu::GpuMat[nPhase];
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCVoneOri[eye][i][phase].create(imageH,imageW*nOrient,CV_32FC1);
				}
		}
	}
}
void Population::mallocGPUCellCUDAoneOri() {

	cell_devCUoneOri = (float ****)malloc(nEye * sizeof(float***));
	for (int eye = 0; eye < nEye; eye++) {
		cell_devCUoneOri[eye] = (float ***)malloc(2 * sizeof(float**));
		for (int i=0; i<2; i++) {
			if (eye == 0) { 
					cell_devCUoneOri[eye][i] = (float **)malloc(1 * sizeof(float*)); 
					checkCudaErrors( cudaMalloc((void **)&cell_devCUoneOri[eye][i][0], nOrient * imageW * imageH * sizeof(float)) );
				} else { 
					cell_devCUoneOri[eye][i] = (float **)malloc(nPhase * sizeof(float*)); 
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaMalloc((void **)&cell_devCUoneOri[eye][i][phase], nOrient * imageW * imageH * sizeof(float)) );
				}
		}
	}
}
void Population::freeGPUCellOPENCVoneOri() {

	for (int eye = 0; eye<nEye; eye++)
		for (int i=0; i<2; i++)  {
			if (eye == 0) {
					cell_devCVoneOri[eye][i][0].release();
				} else {
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCVoneOri[eye][i][phase].release();
				}
		}
	free(cell_devCVoneOri);
}
void Population::freeGPUCellCUDAoneOri() {

	for (int eye = 0; eye < nEye; eye++)
		for (int i=0; i<2; i++) {
			if (eye == 0) 
				checkCudaErrors( cudaFree(cell_devCUoneOri[eye][i][0]) );
			else 
				for (int phase = 0; phase<nPhase; phase++)
					checkCudaErrors( cudaFree(cell_devCUoneOri[eye][i][phase]) );
				}
	free(cell_devCUoneOri);
}

void Population::mallocGPUCellOPENCVrepOriRepPhase() {

	cell_devCVrepOriRepPhase = new cv::gpu::GpuMat[2];
	for (int eye=0; eye<nEye; eye++) 
		cell_devCVrepOriRepPhase[eye].create(2*imageH,imageW*nPhase*nOrient,CV_32FC1);
}
void Population::mallocGPUCellCUDArepOriRepPhase() {

	cell_devCUrepOriRepPhase = (float**) malloc(2 * sizeof(float*));
	for (int eye=0;eye<nEye; eye++)
		checkCudaErrors( cudaMalloc((void **)&cell_devCUrepOriRepPhase[eye], 2 * imageW * imageH * nPhase * nOrient * sizeof(float)) );
}
void Population::freeGPUCellOPENCVrepOriRepPhase() {

	for (int eye=0; eye<nEye; eye++)
		cell_devCVrepOriRepPhase[eye].release();
	delete [] cell_devCVrepOriRepPhase;
}
void Population::freeGPUCellCUDArepOriRepPhase() {

	for (int eye=0;eye<nEye; eye++)
		cudaFree(cell_devCUrepOriRepPhase[eye]);
	free(cell_devCUrepOriRepPhase);
}

void Population::mallocCPUCellOPENCV() {

	cell_hostCV = (cv::Mat****)malloc(2 * sizeof(cv::Mat***)); // (L = 0, R = 1)
	for (int eye = 0; eye<nEye; eye++) { 
		cell_hostCV[eye] = (cv::Mat***)malloc(2 * sizeof(cv::Mat**)); // (Re = 0, Im = 1)
		for (int i = 0; i<2; i++) {
			cell_hostCV[eye][i] = (cv::Mat**)malloc(nOrient * sizeof(cv::Mat*)); 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) { // L: Re, Im are nOrient x 1
					cell_hostCV[eye][i][theta] = new cv::Mat[1];
					cell_hostCV[eye][i][theta][0].create(imageH,imageW,CV_32FC1);
				} else { // R: Re, Im are nOrient x nPhase
					cell_hostCV[eye][i][theta] = new cv::Mat[nPhase];
					for (int phase = 0; phase<nPhase; phase++)
						cell_hostCV[eye][i][theta][phase].create(imageH,imageW,CV_32FC1);
				}
			} 
		}
	}
}
void Population::mallocCPUCellCUDA() {

	cell_hostCU = (float *****)malloc(2 * sizeof(float****)); // (L = 0, R = 1)
	for (int eye = 0; eye<nEye; eye++) { 
		cell_hostCU[eye] = (float ****)malloc(2 * sizeof(float***));  // (Re = 0, Im = 1)
		for (int i = 0; i<2; i++) { 
			cell_hostCU[eye][i] = (float ***)malloc(nOrient * sizeof(float**)); 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) { // L: Re, Im are nOrient x 1
					cell_hostCU[eye][i][theta] = (float **)malloc(1 * sizeof(float*)); 
					cell_hostCU[eye][i][theta][0] = (float *)malloc(imageW * imageH * sizeof(float));
				} else { // R: Re, Im are nOrient x nPhase
					cell_hostCU[eye][i][theta] = (float **)malloc(nPhase * sizeof(float*)); 
					for (int phase = 0; phase<nPhase; phase++)
						cell_hostCU[eye][i][theta][phase] = (float *)malloc(imageW * imageH * sizeof(float));
				}
			} 
		}
	}		
}
void Population::freeCPUCellOPENCV() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0) {
					cell_hostCV[eye][i][theta][0].release();
				} else {
					for (int phase = 0; phase<nPhase; phase++) 
						cell_hostCV[eye][i][theta][phase].release();
				}
			}
	free(cell_hostCV);
}
void Population::freeCPUCellCUDA() {

	for (int eye = 0; eye<nEye; eye++) 
		for (int i = 0; i<2; i++) 
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					free(cell_hostCU[eye][i][theta][0]);
				else
					for (int phase = 0; phase<nPhase; phase++)
						free(cell_hostCU[eye][i][theta][phase]);
			} 	
}

void Population::mallocGPUTempMatricesCUDA(){

	tempCU = (float**) malloc(nTempMatricesCU * sizeof(float*));
	for (int i=0; i<nTempMatricesCU; i++) {
		checkCudaErrors( cudaMalloc((void **)&tempCU[i],   imageW * imageH * sizeof(float)) );
		//cout<<"done CUDA temp allocation: "<<i<<endl;
	}
}
void Population::freeGPUTempMatricesCUDA() {

	for (int i=0; i<nTempMatricesCU; i++) 
		checkCudaErrors( cudaFree(tempCU[i]) );
	free(tempCU);
}
void Population::mallocGPUTempMatricesOPENCV(){

	tempCV = new cv::gpu::GpuMat[nTempMatricesCV];
	for (int i=0; i<nTempMatricesCV; i++) {
		tempCV[i].create(imageH,imageW,CV_32FC1);
		//cout<<"done OPENCV temp allocation: "<<i<<endl;
	}
    mask0 = cv::gpu::GpuMat(imageH,imageW,CV_32FC1,0);
}
void Population::freeGPUTempMatricesOPENCV() {

    for (int i=0; i<nEye; i++) 
		tempCV[i].release();
    delete [] tempCV;

    mask0.release();
}
void Population::set0GPUTempMatricesCUDA(int *idx, int N) {

	for (int i=0; i<N; i++)
		nppsSet_32f(0, tempCU[idx[i]], imageW*imageH);
}
void Population::set0GPUTempMatricesOPENCV(int *idx, int N) {

	for (int i=0; i<N; i++)
		tempCV[idx[i]].setTo(0);
}

void Population::mallocGPUTempMatricesCUDAoneOri() {

	tempCUoneOri = (float**) malloc(nTempMatricesCUoneOri * sizeof(float*));
	for (int i=0; i<nTempMatricesCUoneOri; i++) {
		checkCudaErrors( cudaMalloc((void **)&tempCUoneOri[i], nOrient * imageW * imageH * sizeof(float)) );
		//cout<<"done CUDA temp allocation: "<<i<<endl;
	}
}
void Population::mallocGPUTempMatricesOPENCVoneOri() {

	tempCVoneOri = new cv::gpu::GpuMat[nTempMatricesCVoneOri];
	for (int i=0; i<nTempMatricesCVoneOri; i++) {
		tempCVoneOri[i].create(imageH,nOrient*imageW,CV_32FC1);
		//cout<<"done OPENCV temp allocation: "<<i<<endl;
	}
    mask0 = cv::gpu::GpuMat(imageH,nOrient*imageW,CV_32FC1,0);
}
void Population::freeGPUTempMatricesCUDAoneOri() {

	for (int i=0; i<nTempMatricesCUoneOri; i++) 
		checkCudaErrors( cudaFree(tempCUoneOri[i]) );
	free(tempCUoneOri);
}
void Population::freeGPUTempMatricesOPENCVoneOri() {

	for (int i=0; i<nTempMatricesCVoneOri; i++) 
		tempCVoneOri[i].release();
    delete [] tempCVoneOri;

    mask0.release();
}
void Population::set0GPUTempMatricesCUDAoneOri(int *idx, int N) {

	for (int i=0; i<N; i++)
		nppsSet_32f(0, tempCUoneOri[idx[i]], nOrient*imageW*imageH);
}
void Population::set0GPUTempMatricesOPENCVoneOri(int *idx, int N) {

	for (int i=0; i<N; i++)
		tempCVoneOri[idx[i]].setTo(0);
}

void Population::mallocGPUTempMatricesCUDArepOriRepPhase() {

	tempCUrepOriRepPhase = (float**) malloc(nTempMatricesCUrepOriRepPhase * sizeof(float*));
	for (int i=0; i<nTempMatricesCUrepOriRepPhase; i++) {
		checkCudaErrors( cudaMalloc((void **)&tempCUrepOriRepPhase[i], nOrient * nPhase * imageW * imageH * sizeof(float)) );
		//cout<<"done CUDA temp allocation: "<<i<<endl;
	}
}
void Population::mallocGPUTempMatricesOPENCVrepOriRepPhase()  {

	tempCVrepOriRepPhase = new cv::gpu::GpuMat[nTempMatricesCVrepOriRepPhase];
	for (int i=0; i<nTempMatricesCVrepOriRepPhase; i++) {
		tempCVrepOriRepPhase[i].create(imageH,nOrient*nPhase*imageW,CV_32FC1);
		//cout<<"done OPENCV temp allocation: "<<i<<endl;
	}
}
void Population::freeGPUTempMatricesCUDArepOriRepPhase()  {

	for (int i=0; i<nTempMatricesCUrepOriRepPhase; i++) 
		checkCudaErrors( cudaFree(tempCUrepOriRepPhase[i]) );
	free(tempCUrepOriRepPhase);
}
void Population::freeGPUTempMatricesOPENCVrepOriRepPhase()  {

	for (int i=0; i<nTempMatricesCVrepOriRepPhase; i++) 
		tempCVrepOriRepPhase[i].release();
    delete [] tempCVrepOriRepPhase;
}
void Population::set0GPUTempMatricesCUDArepOriRepPhase(int *idx, int N) {
	
	for (int i=0; i<N; i++)
		nppsSet_32f(0, tempCUrepOriRepPhase[idx[i]], imageW*imageH*nOrient*nPhase);
}
void Population::set0GPUTempMatricesOPENCVrepOriRepPhase(int *idx, int N) {

	for (int i=0; i<N; i++)
		tempCVrepOriRepPhase[idx[i]].setTo(0);
}

void Population::mallocGPUStreamsCUDA(){

	strCU = (cudaStream_t*) malloc(nStrCU * sizeof(cudaStream_t));
	for (int i=0; i<nStrCU; i++)
			checkCudaErrors( cudaStreamCreate(&(strCU[i])) );
	streamCU = (cudaStream_t*) malloc(nStreamCU * sizeof(cudaStream_t));
    for (int i=0; i<nStreamCU; i++)
        checkCudaErrors( cudaStreamCreate(&(streamCU[i])) );   

	// the events are used for synchronization only and hence do not need to record timings
    // this also makes events not introduce global sync points when recorded which is critical to get overlap 
    kernelEventStrCU = (cudaEvent_t*) malloc(nStrCU * sizeof(cudaEvent_t));
    for(int i = 0; i < nStrCU; i++)
        checkCudaErrors( cudaEventCreateWithFlags(&(kernelEventStrCU[i]), cudaEventDisableTiming) );
    kernelEventStreamCU = (cudaEvent_t*) malloc(nStreamCU * sizeof(cudaEvent_t));
    for(int i = 0; i < nStreamCU; i++)
        checkCudaErrors( cudaEventCreateWithFlags(&(kernelEventStreamCU[i]), cudaEventDisableTiming) );
}
void Population::freeGPUStreamsCUDA() {

	for (int i=0; i<nStrCU; i++)
			checkCudaErrors( cudaStreamDestroy(strCU[i]) );
	for (int i=0; i<nStreamCU; i++)
			checkCudaErrors( cudaStreamDestroy(streamCU[i]) );
	free(strCU);
	free(streamCU);

    for(int i = 0; i < nStreamCU; i++) 
        checkCudaErrors( cudaEventDestroy(kernelEventStreamCU[i]) );
  	for(int i = 0; i < nStrCU; i++) 
       checkCudaErrors( cudaEventDestroy(kernelEventStrCU[i]) );
    free(kernelEventStrCU);
	free(kernelEventStreamCU);
}
void Population::mallocGPUStreamsOPENCV(){

	strCV = new cv::gpu::Stream[nStrCV];
	streamCV = new cv::gpu::Stream[nStreamCV];
}
void Population::freeGPUStreamsOPENCV() {

	for (int i=0; i<nStrCV; i++)
		strCV[i].~Stream();
	for (int i=0; i<nStreamCV; i++)
		streamCV[i].~Stream();
	// se si mettono entrambi questi distruttori (for e delete []) da un errore: quindi o uno o l'altro
	//delete [] strCV;
	//delete [] streamCV;

}

void Population::createStartStopEvents() {

	checkCudaErrors( cudaEventCreate(&start_event) );
    checkCudaErrors( cudaEventCreate(&stop_event) );
}
void Population::destroyStartStopEvents(){

	cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}
void Population::createFuncTimer() {

	sdkCreateTimer(&funcTimer);
}
void Population::destroyFuncTimer() {

	sdkDeleteTimer(&funcTimer);
}

void Population::downloadSimpleAnswerOPENCV() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					cell_devCV[eye][i][theta][0].download(cell_hostCV[eye][i][theta][0]);
				else
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCV[eye][i][theta][phase].download(cell_hostCV[eye][i][theta][phase]);
			}
}
void Population::downloadSimpleAnswerCUDA() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][0], cell_devCU[eye][i][theta][0], imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
				else
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][phase], cell_devCU[eye][i][theta][phase], imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
		    }
}
void Population::downloadSimpleAnswerOPENCVoneOri() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					cell_devCVoneOri[eye][i][0](cv::Rect(theta*imageW,0,imageW,imageH)).download(cell_hostCV[eye][i][theta][0]);
				else
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCVoneOri[eye][i][phase](cv::Rect(theta*imageW,0,imageW,imageH)).download(cell_hostCV[eye][i][theta][phase]);
			}
}
void Population::downloadSimpleAnswerCUDAoneOri() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][0], cell_devCUoneOri[eye][i][0] + theta*imageW*imageH, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
				else
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][phase], cell_devCUoneOri[eye][i][phase] + theta*imageW*imageH, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
		    }
}
void Population::downloadSimpleAnswerOPENCVrepOriRepPhase() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					cell_devCVrepOriRepPhase[eye](cv::Rect(0*imageW*nOrient + theta*imageW,i*imageH,imageW,imageH)).download(cell_hostCV[eye][i][theta][0]);
				else
					for (int phase = 0; phase<nPhase; phase++)
						cell_devCVrepOriRepPhase[eye](cv::Rect(phase*imageW*nOrient + theta*imageW,i*imageH,imageW,imageH)).download(cell_hostCV[eye][i][theta][phase]);
			}
}
void Population::downloadSimpleAnswerCUDArepOriRepPhase() {

	for (int eye=0; eye<nEye; eye++) 
		for (int i=0; i<2; i++)
			for (int theta = 0; theta<nOrient; theta++) {
				if (eye == 0)
					checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][0], cell_devCUrepOriRepPhase[eye] + i*imageW*imageH*nOrient*nPhase + 0*imageW*imageH*nOrient + theta*imageW*imageH, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
				else
					for (int phase = 0; phase<nPhase; phase++)
						checkCudaErrors( cudaMemcpy(cell_hostCU[eye][i][theta][phase], cell_devCUrepOriRepPhase[eye] + i*imageW*imageH*nOrient*nPhase + phase*imageW*imageH*nOrient + theta*imageW*imageH, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
		    }
}

void Population::downloadResultOPENCV(cv::Mat &_dst_hostCV, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int i=0; i<nResults; i++)
		cell_devCV[0][0][i][0].download(_dst_hostCV(cv::Rect(i*imageW,0,imageW,imageH)));

	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::downloadResultCUDA(float *_dst_hostCU, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int i=0; i<nResults; i++)
		checkCudaErrors( cudaMemcpy(_dst_hostCU + i*imageW*imageH, cell_devCU[0][0][i][0], imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );

	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::downloadResultOPENCVoneOri(cv::Mat &_dst_hostCV, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cell_devCVoneOri[0][0][0](cv::Rect(0,0,nResults*imageW,imageH)).download(_dst_hostCV);

	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::downloadResultCUDAoneOri(float *_dst_hostCU, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	checkCudaErrors( cudaMemcpy(_dst_hostCU, cell_devCUoneOri[0][0][0], nResults * imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );

	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::downloadResultOPENCVrepOriRepPhase(cv::Mat &_dst_hostCV, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nResults*imageW,imageH)).download(_dst_hostCV);

	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::downloadResultCUDArepOriRepPhase(float *_dst_hostCU, float* elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	checkCudaErrors( cudaMemcpy(_dst_hostCU, cell_devCUrepOriRepPhase[0], nResults * imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost) );
	
	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Cells' simple answers calculation
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::calcSimpleAnswerOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0]);
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1]);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0],cell_devCV[eye][1][0][phaseIdx]);
		f->filters1Dcolumn_devCV[2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1],cell_devCV[eye][1][4][phaseIdx]);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0],cell_devCV[eye][0][0][phaseIdx]);
		f->filters1Dcolumn_devCV[1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1],cell_devCV[eye][0][4][phaseIdx]);
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0]);
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1]);
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2]);
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3]);
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4]);
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5]);
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye*centralPhaseIdx;

			cv::gpu::subtract(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCV[eye][0][resIdx1][phaseIdx],mask0,-1);
			cv::gpu::add(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCV[eye][0][resIdx2][phaseIdx],mask0,-1);
		
			cv::gpu::add(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCV[eye][1][resIdx1][phaseIdx],mask0,-1);
			cv::gpu::subtract(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCV[eye][1][resIdx2][phaseIdx],mask0,-1);	
			}

    // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0],streamCV[eye]);
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1],streamCV[eye+nEye]);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		
		f->filters1Drow_devCV[1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0],cell_devCV[eye][0][0][phaseIdx],streamCV[eye]);
		f->filters1Dcolumn_devCV[1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1],cell_devCV[eye][0][4][phaseIdx],streamCV[eye+nEye]);
		
		f->filters1Drow_devCV[2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0],cell_devCV[eye][1][0][phaseIdx],streamCV[eye+2*nEye]);
		f->filters1Dcolumn_devCV[2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1],cell_devCV[eye][1][4][phaseIdx],streamCV[eye+3*nEye]);
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0],strCV[eye*nCoupleOri+callIdx]);
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1],strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			strCV[eye*nCoupleOri+callIdx].waitForCompletion();
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],strCV[eye*nCoupleOri+callIdx]);
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye * centralPhaseIdx;

			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			cv::gpu::subtract(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCV[eye][0][resIdx1][phaseIdx],mask0,-1,strCV[eye*nCoupleOri+callIdx]);
			cv::gpu::add(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCV[eye][0][resIdx2][phaseIdx],mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3].waitForCompletion();
			cv::gpu::add(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCV[eye][1][resIdx1][phaseIdx],mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			cv::gpu::subtract(tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCV[eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCV[eye][1][resIdx2][phaseIdx],mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);		
			}

     // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
		
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {	
		convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], src_devCU[eye], imageW, imageH, taps, 0);
		convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], src_devCU[eye], imageW, imageH, taps, 0);		
	}

    for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCU[eye][1][0][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], imageW, imageH, taps, 2);
		convolutionColumnsGPU(cell_devCU[eye][1][4][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], imageW, imageH, taps, 2);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCU[eye][0][0][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], imageW, imageH, taps, 1);
		convolutionColumnsGPU(cell_devCU[eye][0][4][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], imageW, imageH, taps, 1);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;
			convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], src_devCU[eye], imageW, imageH, taps, columnIdx1);
		    convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], src_devCU[eye], imageW, imageH, taps, columnIdx2);
		}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], imageW, imageH, taps, rowIdx1);
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], imageW, imageH, taps, rowIdx1);
}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;

			nppsSub_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],cell_devCU[eye][0][resIdx1][phaseIdx],imageW*imageH);
			nppsAdd_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCU[eye][0][resIdx2][phaseIdx],imageW*imageH);
			nppsAdd_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCU[eye][1][resIdx1][phaseIdx],imageW*imageH);
			nppsSub_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],cell_devCU[eye][1][resIdx2][phaseIdx],imageW*imageH);
			}
	
	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
		
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
    
	for (int eye=0; eye<nEye; eye++) {
		convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye]);
		convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye+nEye]);		
	}

	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;

		convolutionRowsGPU(cell_devCU[eye][0][0][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], imageW, imageH, taps, 1, streamCU[eye]);
		convolutionColumnsGPU(cell_devCU[eye][0][4][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], imageW, imageH, taps, 1,streamCU[eye+nEye]);

		convolutionRowsGPU(cell_devCU[eye][1][0][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 0], imageW, imageH, taps, 2, streamCU[eye+2*nEye]);
		convolutionColumnsGPU(cell_devCU[eye][1][4][phaseIdx], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + 0*nTempAnswers + 1], imageW, imageH, taps, 2,streamCU[eye+3*nEye]);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], src_devCU[eye], imageW, imageH, taps, columnIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionColumnsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], src_devCU[eye], imageW, imageH, taps, columnIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		    checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]) );

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]) );

			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 0], imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionRowsGPU(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3], tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 1], imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
			
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			nppsAdd_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],cell_devCU[eye][1][resIdx1][phaseIdx],imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			nppsSub_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 4],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 5],cell_devCU[eye][1][resIdx2][phaseIdx],imageW*imageH);
			
            checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx]);
			nppsSub_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],cell_devCU[eye][0][resIdx1][phaseIdx],imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f(tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 2],tempCU[nOrient + eye*nCoupleOri*nTempAnswers + callIdx*nTempAnswers + 3],cell_devCU[eye][0][resIdx2][phaseIdx],imageW*imageH);
		}
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcSimpleAnswerOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)));
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)));
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[2]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(0*imageW,0,imageW,imageH)));
		f->filters1Dcolumn_devCV[2]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(4*imageW,0,imageW,imageH)));
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[1]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(0*imageW,0,imageW,imageH)));
		f->filters1Dcolumn_devCV[1]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(4*imageW,0,imageW,imageH)));
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)));
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)));
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)));
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)));
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)));
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)));
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye*centralPhaseIdx;

			cv::gpu::subtract(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(resIdx1*imageW,0,imageW,imageH)),mask0,-1);
			cv::gpu::add(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(resIdx2*imageW,0,imageW,imageH)),mask0,-1);
		
			cv::gpu::add(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(resIdx1*imageW,0,imageW,imageH)),mask0,-1);
			cv::gpu::subtract(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(resIdx2*imageW,0,imageW,imageH)),mask0,-1);	
			}

     // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)),streamCV[eye]);
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)),streamCV[eye+nEye]);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		
		f->filters1Drow_devCV[1]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(0*imageW,0,imageW,imageH)),streamCV[eye]);
		f->filters1Dcolumn_devCV[1]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(4*imageW,0,imageW,imageH)),streamCV[eye+nEye]);
		
		f->filters1Drow_devCV[2]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*0,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(0*imageW,0,imageW,imageH)),streamCV[eye+2*nEye]);
		f->filters1Dcolumn_devCV[2]->operator ()(tempCVoneOri[eye*nCoupleOri + 0](cv::Rect(imageW*1,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(4*imageW,0,imageW,imageH)),streamCV[eye+3*nEye]);
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx]);
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			strCV[eye*nCoupleOri+callIdx].waitForCompletion();
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx]);
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*0,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*1,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye * centralPhaseIdx;

			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			cv::gpu::subtract(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(resIdx1*imageW,0,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx]);
			cv::gpu::add(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*2,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*3,0,imageW,imageH)),cell_devCVoneOri[eye][0][phaseIdx](cv::Rect(resIdx2*imageW,0,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3].waitForCompletion();
			cv::gpu::add(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(resIdx1*imageW,0,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			cv::gpu::subtract(tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*5,0,imageW,imageH)),tempCVoneOri[eye*nCoupleOri + callIdx](cv::Rect(imageW*4,0,imageW,imageH)),cell_devCVoneOri[eye][1][phaseIdx](cv::Rect(resIdx2*imageW,0,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);		
			}

    // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {	
		convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0);
		convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0);		
	}

    for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCUoneOri[eye][1][phaseIdx] + 0*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, imageW, imageH, taps, 2);
		convolutionColumnsGPU(cell_devCUoneOri[eye][1][phaseIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, imageW, imageH, taps, 2);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCUoneOri[eye][0][phaseIdx] + 0*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, imageW, imageH, taps, 1);
		convolutionColumnsGPU(cell_devCUoneOri[eye][0][phaseIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, imageW, imageH, taps, 1);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;
			convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx1);
		    convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx2);
		}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, imageW, imageH, taps, rowIdx1);
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, imageW, imageH, taps, rowIdx1);
}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;

			nppsSub_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH,cell_devCUoneOri[eye][0][phaseIdx] + resIdx1*imageW*imageH,imageW*imageH);
			nppsAdd_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH,cell_devCUoneOri[eye][0][phaseIdx] + resIdx2*imageW*imageH,imageW*imageH);
			nppsAdd_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH,cell_devCUoneOri[eye][1][phaseIdx] + resIdx1*imageW*imageH,imageW*imageH);
			nppsSub_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH,cell_devCUoneOri[eye][1][phaseIdx] + resIdx2*imageW*imageH,imageW*imageH);
			}
	
	 // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
    
	for (int eye=0; eye<nEye; eye++) {
		convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye]);
		convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye+nEye]);		
	}

	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;

		convolutionRowsGPU(cell_devCUoneOri[eye][0][phaseIdx] + 0*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, imageW, imageH, taps, 1, streamCU[eye]);
		convolutionColumnsGPU(cell_devCUoneOri[eye][0][phaseIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, imageW, imageH, taps, 1,streamCU[eye+nEye]);

		convolutionRowsGPU(cell_devCUoneOri[eye][1][phaseIdx] + 0*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 0*imageW*imageH, imageW, imageH, taps, 2, streamCU[eye+2*nEye]);
		convolutionColumnsGPU(cell_devCUoneOri[eye][1][phaseIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + 0] + 1*imageW*imageH, imageW, imageH, taps, 2,streamCU[eye+3*nEye]);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionColumnsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		    checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]) );

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]) );

			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 0*imageW*imageH, imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionRowsGPU(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH, tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 1*imageW*imageH, imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
			
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;
	
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			nppsAdd_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH,cell_devCUoneOri[eye][1][phaseIdx] + resIdx1*imageW*imageH,imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			nppsSub_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 4*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 5*imageW*imageH,cell_devCUoneOri[eye][1][phaseIdx] + resIdx2*imageW*imageH,imageW*imageH);
		
            checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx]);
			nppsSub_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH,cell_devCUoneOri[eye][0][phaseIdx] + resIdx1*imageW*imageH,imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f(tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 2*imageW*imageH,tempCUoneOri[1 + eye*nCoupleOri + callIdx] + 3*imageW*imageH,cell_devCUoneOri[eye][0][phaseIdx] + resIdx2*imageW*imageH,imageW*imageH);
		}
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcSimpleAnswerOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)));
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)));
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 0*imageW,1*imageH,imageW,imageH)));
		f->filters1Dcolumn_devCV[2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 4*imageW,1*imageH,imageW,imageH)));
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		f->filters1Drow_devCV[1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 0*imageW,0*imageH,imageW,imageH)));
		f->filters1Dcolumn_devCV[1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 4*imageW,0*imageH,imageW,imageH)));
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)));
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)));
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)));
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)));
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)));
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)));
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye*centralPhaseIdx;

			cv::gpu::subtract(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx1*imageW,0*imageH,imageW,imageH)),mask0,-1);
			cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx2*imageW,0*imageH,imageW,imageH)),mask0,-1);
		
			cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx1*imageW,1*imageH,imageW,imageH)),mask0,-1);
			cv::gpu::subtract(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx2*imageW,1*imageH,imageW,imageH)),mask0,-1);	
			}

   // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {
		f->filters1Dcolumn_devCV[0]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)),streamCV[eye]);
		f->filters1Drow_devCV[0]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)),streamCV[eye+nEye]);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		
		f->filters1Drow_devCV[1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 0*imageW,0*imageH,imageW,imageH)),streamCV[eye]);
		f->filters1Dcolumn_devCV[1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 4*imageW,0*imageH,imageW,imageH)),streamCV[eye+nEye]);
		
		f->filters1Drow_devCV[2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 0*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 0*imageW,1*imageH,imageW,imageH)),streamCV[eye+2*nEye]);
		f->filters1Dcolumn_devCV[2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + 0*imageW*nOrient + 1*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + 4*imageW,1*imageH,imageW,imageH)),streamCV[eye+3*nEye]);
	}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			f->filters1Dcolumn_devCV[columnIdx1]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx]);
			f->filters1Dcolumn_devCV[columnIdx2]->operator ()(src_devCV[eye],tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
		    
			strCV[eye*nCoupleOri+callIdx].waitForCompletion();
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx]);
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			
			f->filters1Drow_devCV[rowIdx2]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 0*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			f->filters1Drow_devCV[rowIdx1]->operator ()(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 1*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)),strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		    resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			phaseIdx = eye * centralPhaseIdx;

			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
			cv::gpu::subtract(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx1*imageW,0*imageH,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx]);
			cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 2*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx2*imageW,0*imageH,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		
			strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3].waitForCompletion();
			cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx1*imageW,1*imageH,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			cv::gpu::subtract(tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 5*imageW,0,imageW,imageH)),tempCVrepOriRepPhase[0](cv::Rect(eye*nCoupleOri*imageW*nOrient + callIdx*imageW*nOrient + 4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*imageW*nOrient + resIdx2*imageW,1*imageH,imageW,imageH)),mask0,-1,strCV[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);		
			}

   // wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int eye=0; eye<nEye; eye++) {	
		convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0);
		convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0);		
	}

    for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 0*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, 2);
		convolutionColumnsGPU(cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, 2);
	}
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		convolutionRowsGPU(cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 0*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, 1);
		convolutionColumnsGPU(cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, 1);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;
			convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx1);
		    convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx2);
		}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, rowIdx1);
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, rowIdx2);
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, rowIdx1);
}

	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;

			nppsSub_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx1*imageW*imageH,imageW*imageH);
			nppsAdd_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx2*imageW*imageH,imageW*imageH);
			nppsAdd_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx1*imageW*imageH,imageW*imageH);
			nppsSub_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx2*imageW*imageH,imageW*imageH);
			}
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcSimpleAnswerCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int columnIdx1;
	int columnIdx2;
	int rowIdx1;
	int rowIdx2;
	int resIdx1;
	int resIdx2;
	int phaseIdx;

	int columns[] = {7, 3, 5};
	int rows[] = {5, 3, 7}; 
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
    
	for (int eye=0; eye<nEye; eye++) {
		convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye]);
		convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, 0, streamCU[eye+nEye]);		
	}

	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;

		convolutionRowsGPU(cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 0*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, 1, streamCU[eye]);
		convolutionColumnsGPU(cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, 1,streamCU[eye+nEye]);

		convolutionRowsGPU(cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 0*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, 2, streamCU[eye+2*nEye]);
		convolutionColumnsGPU(cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + 0*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, 2,streamCU[eye+3*nEye]);
	}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			columnIdx1 = columns[callIdx];
			columnIdx2 = columnIdx1 + 1;

			convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionColumnsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, src_devCU[eye], imageW, imageH, taps, columnIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
		    checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}

	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			rowIdx1 = rows[callIdx];
			rowIdx2 = rowIdx1 + 1;

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]) );

			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]) );

			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 0*imageW*imageH, imageW, imageH, taps, rowIdx1, strCU[eye*nCoupleOri+callIdx]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx], strCU[eye*nCoupleOri+callIdx]) );
			
			convolutionRowsGPU(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH, tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 1*imageW*imageH, imageW, imageH, taps, rowIdx2, strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]) );
			
		}
	
	for (int eye=0; eye<nEye; eye++) 
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			phaseIdx = eye * centralPhaseIdx;
	
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
			nppsAdd_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx1*imageW*imageH,imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*2],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
			nppsSub_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 4*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 5*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 1*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx2*imageW*imageH,imageW*imageH);
		
            checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx], kernelEventStrCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx]);
			nppsSub_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx1*imageW*imageH,imageW*imageH);
			
			checkCudaErrors( cudaStreamWaitEvent(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[eye*nCoupleOri+callIdx],0) );
			nppSetStream(strCU[eye*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f(tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 2*imageW*imageH,tempCUrepOriRepPhase[1] + eye*nCoupleOri*imageW*imageH*nOrient + callIdx*imageW*imageH*nOrient + 3*imageW*imageH,cell_devCUrepOriRepPhase[eye] + 0*imageW*imageH*nPhase*nOrient + phaseIdx*imageW*imageH*nOrient + resIdx2*imageW*imageH,imageW*imageH);
		}
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Cells' simple answers shifting R wrt L
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::shiftSimpleAnswerOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for(int theta = 0; theta < nOrient; theta++) 
		for (int phase = 0; phase < nPhase; phase++)  {
				cv::gpu::addWeighted(cell_devCV[1][0][theta][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][1][theta][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCV[1][0][theta][phase],-1);
				cv::gpu::addWeighted(cell_devCV[1][1][theta][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][0][theta][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCV[1][1][theta][phase],-1);
			}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for(int theta = 0; theta < nOrient; theta++) 
		for (int phase = 0; phase < nPhase; phase++)  {
				nppsMulC_32f(cell_devCU[1][0][theta][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][0][theta][phase],imageW*imageH);
				nppsAddProductC_32f(cell_devCU[1][1][theta][centralPhaseIdx],-f->sinPhShift[phase],cell_devCU[1][0][theta][phase],imageW*imageH);
				
				nppsMulC_32f(cell_devCU[1][1][theta][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][1][theta][phase],imageW*imageH);
				nppsAddProductC_32f(cell_devCU[1][0][theta][centralPhaseIdx],f->sinPhShift[phase],cell_devCU[1][1][theta][phase],imageW*imageH);
			}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){

	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {

			cv::gpu::addWeighted(cell_devCV[1][0][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][1][0][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCV[1][0][0][phase],-1,streamCV[1]);
			cv::gpu::addWeighted(cell_devCV[1][0][4][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][1][4][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCV[1][0][4][phase],-1,streamCV[1 + nEye]);

			cv::gpu::addWeighted(cell_devCV[1][1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][0][0][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCV[1][1][0][phase],-1,streamCV[1 + 2*nEye]);
			cv::gpu::addWeighted(cell_devCV[1][1][4][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][0][4][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCV[1][1][4][phase],-1,streamCV[1 + 3*nEye]);

			for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
				resIdx1 = callIdx+1;
			    resIdx2 = nOrient-1-callIdx;

				cv::gpu::addWeighted(cell_devCV[1][0][resIdx1][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][1][resIdx1][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCV[1][0][resIdx1][phase],-1,strCV[1*nCoupleOri+callIdx]);
				cv::gpu::addWeighted(cell_devCV[1][0][resIdx2][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][1][resIdx2][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCV[1][0][resIdx2][phase],-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);

				cv::gpu::addWeighted(cell_devCV[1][1][resIdx1][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][0][resIdx1][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCV[1][1][resIdx1][phase],-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
				cv::gpu::addWeighted(cell_devCV[1][1][resIdx2][centralPhaseIdx],f->cosPhShift[phase],cell_devCV[1][0][resIdx2][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCV[1][1][resIdx2][phase],-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri*3]);

			}
		}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){

	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {
			
			nppSetStream(streamCU[1]);
			nppsMulC_32f(cell_devCU[1][0][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][0][0][phase],imageW*imageH);
			nppSetStream(streamCU[1 + nEye]);
			nppsMulC_32f(cell_devCU[1][0][4][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][0][4][phase],imageW*imageH);

			nppSetStream(streamCU[1 + 2*nEye]);
			nppsMulC_32f(cell_devCU[1][1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][1][0][phase],imageW*imageH);
			nppSetStream(streamCU[1 + 3*nEye]);
			nppsMulC_32f(cell_devCU[1][1][4][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][1][4][phase],imageW*imageH);

			for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
				resIdx1 = callIdx+1;
			    resIdx2 = nOrient-1-callIdx;

				nppSetStream(strCU[1*nCoupleOri+callIdx]);
				nppsMulC_32f(cell_devCU[1][0][resIdx1][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][0][resIdx1][phase],imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
				nppsMulC_32f(cell_devCU[1][0][resIdx2][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][0][resIdx2][phase],imageW*imageH);

				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
				nppsMulC_32f(cell_devCU[1][1][resIdx1][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][1][resIdx1][phase],imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
				nppsMulC_32f(cell_devCU[1][1][resIdx2][centralPhaseIdx],f->cosPhShift[phase],cell_devCU[1][1][resIdx2][phase],imageW*imageH);
			}

			nppSetStream(streamCU[1]);
			nppsAddProductC_32f(cell_devCU[1][1][0][centralPhaseIdx],-f->sinPhShift[phase],cell_devCU[1][0][0][phase],imageW*imageH);
			nppSetStream(streamCU[1 + nEye]);
			nppsAddProductC_32f(cell_devCU[1][1][4][centralPhaseIdx],-f->sinPhShift[phase],cell_devCU[1][0][4][phase],imageW*imageH);

			nppSetStream(streamCU[1 + 2*nEye]);
			nppsAddProductC_32f(cell_devCU[1][0][0][centralPhaseIdx],f->sinPhShift[phase],cell_devCU[1][1][0][phase],imageW*imageH);
			nppSetStream(streamCU[1 + 3*nEye]);
			nppsAddProductC_32f(cell_devCU[1][0][4][centralPhaseIdx],f->sinPhShift[phase],cell_devCU[1][1][4][phase],imageW*imageH);

			for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
				resIdx1 = callIdx+1;
			    resIdx2 = nOrient-1-callIdx;

				nppSetStream(strCU[1*nCoupleOri+callIdx]);
				nppsAddProductC_32f(cell_devCU[1][1][resIdx1][centralPhaseIdx],-f->sinPhShift[phase],cell_devCU[1][0][resIdx1][phase],imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
				nppsAddProductC_32f(cell_devCU[1][1][resIdx2][centralPhaseIdx],-f->sinPhShift[phase],cell_devCU[1][0][resIdx2][phase],imageW*imageH);

				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri*2]);
				nppsAddProductC_32f(cell_devCU[1][0][resIdx1][centralPhaseIdx],f->sinPhShift[phase],cell_devCU[1][1][resIdx1][phase],imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri*3]);
				nppsAddProductC_32f(cell_devCU[1][0][resIdx2][centralPhaseIdx],f->sinPhShift[phase],cell_devCU[1][1][resIdx2][phase],imageW*imageH);
			}
	}
	

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::shiftSimpleAnswerOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent){

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++)  {
		cv::gpu::addWeighted(cell_devCVoneOri[1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCVoneOri[1][1][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCVoneOri[1][0][phase],-1);
		cv::gpu::addWeighted(cell_devCVoneOri[1][1][centralPhaseIdx],f->cosPhShift[phase],cell_devCVoneOri[1][0][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCVoneOri[1][1][phase],-1);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++)  {
		nppsMulC_32f(cell_devCUoneOri[1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCUoneOri[1][0][phase],imageW*imageH*nOrient);
		nppsAddProductC_32f(cell_devCUoneOri[1][1][centralPhaseIdx],-f->sinPhShift[phase],cell_devCUoneOri[1][0][phase],imageW*imageH*nOrient);
				
		nppsMulC_32f(cell_devCUoneOri[1][1][centralPhaseIdx],f->cosPhShift[phase],cell_devCUoneOri[1][1][phase],imageW*imageH*nOrient);
		nppsAddProductC_32f(cell_devCUoneOri[1][0][centralPhaseIdx],f->sinPhShift[phase],cell_devCUoneOri[1][1][phase],imageW*imageH*nOrient);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;;
}
void Population::shiftSimpleAnswerOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++)  {
		cv::gpu::addWeighted(cell_devCVoneOri[1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCVoneOri[1][1][centralPhaseIdx],-f->sinPhShift[phase],0,cell_devCVoneOri[1][0][phase],-1,strCV[phase]);
		cv::gpu::addWeighted(cell_devCVoneOri[1][1][centralPhaseIdx],f->cosPhShift[phase],cell_devCVoneOri[1][0][centralPhaseIdx],f->sinPhShift[phase],0,cell_devCVoneOri[1][1][phase],-1,strCV[phase+nPhase]);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++)  {
		nppSetStream(strCU[phase]);
		nppsMulC_32f(cell_devCUoneOri[1][0][centralPhaseIdx],f->cosPhShift[phase],cell_devCUoneOri[1][0][phase],imageW*imageH*nOrient);
		nppsAddProductC_32f(cell_devCUoneOri[1][1][centralPhaseIdx],-f->sinPhShift[phase],cell_devCUoneOri[1][0][phase],imageW*imageH*nOrient);
		
		nppSetStream(strCU[phase+nPhase]);
		nppsMulC_32f(cell_devCUoneOri[1][1][centralPhaseIdx],f->cosPhShift[phase],cell_devCUoneOri[1][1][phase],imageW*imageH*nOrient);
		nppsAddProductC_32f(cell_devCUoneOri[1][0][centralPhaseIdx],f->sinPhShift[phase],cell_devCUoneOri[1][1][phase],imageW*imageH*nOrient);
	}

	nppSetStream(0);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::shiftSimpleAnswerOPENCV_oneOriOnePhase_noStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	cv::gpu::GpuMat cell_devCVoneOriOnePhase;
	cv::gpu::GpuMat *cell_devCVoneOriOnePhaseResult;
	cv::Mat diagCosSin_hostCV, diagCosMinusSin_hostCV;
	cv::gpu::GpuMat diagCosSin, diagCosMinusSin;

	cell_devCVoneOriOnePhase = cv::gpu::GpuMat(imageH * nOrient, imageW * 2, CV_32FC1);
	for (int i=0; i<2; i++) 
		for (int theta=0; theta<nOrient; theta++)
			cell_devCV[1][i][theta][centralPhaseIdx].copyTo(cell_devCVoneOriOnePhase(cv::Rect(imageW*i, theta*imageH, imageW, imageH)));	

	cell_devCVoneOriOnePhaseResult = new cv::gpu::GpuMat[2];
	for (int i=0; i<2; i++)
		cell_devCVoneOriOnePhaseResult[i].create(imageH * nOrient, imageW * nPhase, CV_32FC1);
	
	diagCosSin_hostCV = cv::Mat(imageW * 2, imageW * nPhase, CV_32FC1,cv::Scalar(0));
	diagCosMinusSin_hostCV  = cv::Mat(imageW * 2, imageW * nPhase, CV_32FC1,cv::Scalar(0));
	for (int phase=0; phase<nPhase; phase++) 
		for (int p=0; p<imageW; p++) {
			diagCosSin_hostCV.at<float>(0 * imageW + p,phase * imageW + p) = f->sinPhShift[phase];
			diagCosSin_hostCV.at<float>(1 * imageW + p,phase * imageW + p) = f->cosPhShift[phase];
			//*(diagCosSin_hostCV.data + diagCosSin_hostCV.step*((0 * imageW) + p) + diagCosSin_hostCV.elemSize()*((phase * imageW) + p)) = f->cosPhShift(phase);
			//*(diagCosSin_hostCV.data + diagCosSin_hostCV.step*((1 * imageW) + p) + diagCosSin_hostCV.elemSize()*((phase * imageW) + p)) = f->sinPhShift(phase);

			diagCosMinusSin_hostCV.at<float>(0 * imageW + p,phase * imageW + p) = f->cosPhShift[phase];
			diagCosMinusSin_hostCV.at<float>(1 * imageW + p,phase * imageW + p) = -f->sinPhShift[phase];
			//*(diagCosMinusSin_hostCV.data + diagCosMinusSin_hostCV.step*((0 * imageW) + p) + diagCosMinusSin_hostCV.elemSize()*((phase * imageW) + p)) = f->cosPhShift(phase);
			//*(diagCosMinusSin_hostCV.data + diagCosMinusSin_hostCV.step*((1 * imageW) + p) + diagCosMinusSin_hostCV.elemSize()*((phase * imageW) + p)) = -f->sinPhShift(phase);
		}
	diagCosSin = cv::gpu::GpuMat(imageW * 2, imageW * nPhase, CV_32FC1);
	diagCosMinusSin  = cv::gpu::GpuMat(imageW * 2, imageW * nPhase, CV_32FC1);
	diagCosSin.upload(diagCosSin_hostCV);
	diagCosMinusSin.upload(diagCosMinusSin_hostCV);

	cv::gpu::GpuMat a;

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::gemm(cell_devCVoneOriOnePhase,diagCosMinusSin,1,a,0,cell_devCVoneOriOnePhaseResult[0]);
	cv::gpu::gemm(cell_devCVoneOriOnePhase,diagCosSin,1,a,0,cell_devCVoneOriOnePhaseResult[1]);

	for (int theta=0; theta<nOrient; theta++) 
		for (int phase = 0; phase < nPhase; phase++) {
			cell_devCV[1][0][theta][phase] = cell_devCVoneOriOnePhaseResult[0](cv::Rect(imageW*phase, imageH*theta, imageW, imageH)).clone();
			cell_devCV[1][1][theta][phase] = cell_devCVoneOriOnePhaseResult[1](cv::Rect(imageW*phase, imageH*theta, imageW, imageH)).clone();	
		}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;

	cell_devCVoneOriOnePhase.release();
	for (int i=0; i<2; i++)
		cell_devCVoneOriOnePhaseResult[i].release();
	delete [] cell_devCVoneOriOnePhaseResult;
	diagCosSin.release();
	diagCosMinusSin.release();
	diagCosSin_hostCV.release();
	diagCosMinusSin_hostCV.release();
}
void Population::shiftSimpleAnswerCUDA_oneOriOnePhase_noStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	cublasStatus_t status;
	cublasHandle_t handle;

	float *cell_devCUoneOriOnePhase;
	float **cell_devCUoneOriOnePhaseResult;
	float *diagCosSin_hostCU, *diagCosMinusSin_hostCU; 
	float *diagCosSin, *diagCosMinusSin;

	float alpha = 1.0f;
    float beta = 0.0f;

	status = cublasCreate(&handle);

	checkCudaErrors( cudaMalloc((void **)&cell_devCUoneOriOnePhase, imageH * nOrient * imageW * 2  * sizeof(float)) );
	for (int i=0; i<2; i++) 
		for (int column=0; column<imageW; column++)
			for (int theta=0; theta<nOrient; theta++)
				for (int row=0; row<imageH; row++) 
					checkCudaErrors( cudaMemcpy(cell_devCUoneOriOnePhase + i * (imageH * nOrient * imageW) + column * (imageH * nOrient) + theta * (imageH) + row, cell_devCU[1][i][theta][centralPhaseIdx] + row * imageW + column, sizeof(float), cudaMemcpyDeviceToDevice) );
					
	cell_devCUoneOriOnePhaseResult = (float**) malloc(2 * sizeof(float*));
	for (int i=0; i<2; i++)
		checkCudaErrors( cudaMalloc((void **)&cell_devCUoneOriOnePhaseResult[i], imageH * nOrient * imageW * nPhase * sizeof(float)) );

	diagCosSin_hostCU = (float*) malloc(imageW * 2 * imageW * nPhase * sizeof(float));
	diagCosMinusSin_hostCU = (float*) malloc(imageW * 2 * imageW * nPhase * sizeof(float));
	memset(diagCosSin_hostCU, 0, imageW * 2 * imageW * nPhase * sizeof(float));
	memset(diagCosMinusSin_hostCU, 0, imageW * 2 * imageW * nPhase * sizeof(float));
	for (int phase=0; phase<nPhase; phase++) 
		for (int p=0; p<imageW; p++) {
			diagCosSin_hostCU[phase * (imageW * 2 * imageW) + p * (imageW * 2) + 0 * (imageW) + p] = f->sinPhShift[phase];
			diagCosSin_hostCU[phase * (imageW * 2 * imageW) + p * (imageW * 2) + 1 * (imageW) + p] = f->cosPhShift[phase];
			
			diagCosMinusSin_hostCU[phase * (imageW * 2 * imageW) + p * (imageW * 2) + 0 * (imageW) + p] = f->cosPhShift[phase];
			diagCosMinusSin_hostCU[phase * (imageW * 2 * imageW) + p * (imageW * 2) + 1 * (imageW) + p] = -f->sinPhShift[phase];
		}
	checkCudaErrors( cudaMalloc((void **)&diagCosSin, imageW * 2 * imageW * nPhase * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void **)&diagCosMinusSin, imageW * 2 * imageW * nPhase * sizeof(float)) );
	cublasSetVector(imageW * 2 * imageW * nPhase, sizeof(float), diagCosSin_hostCU, 1, diagCosSin, 1);
	cublasSetVector(imageW * 2 * imageW * nPhase, sizeof(float), diagCosMinusSin_hostCU, 1, diagCosMinusSin, 1);

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nOrient * imageH, imageW * nPhase, 2 * imageW, &alpha, cell_devCUoneOriOnePhase, nOrient * imageH, diagCosSin, 2 * imageW, &beta, cell_devCUoneOriOnePhaseResult[1], nOrient * imageH);
	status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, nOrient * imageH, imageW * nPhase, 2 * imageW, &alpha, cell_devCUoneOriOnePhase, nOrient * imageH, diagCosMinusSin, 2 * imageW, &beta, cell_devCUoneOriOnePhaseResult[0], nOrient * imageH);

	for (int theta=0; theta<nOrient; theta++)
		for (int phase = 0; phase < nPhase; phase++)
			for (int row=0; row<imageH; row++) 
				for (int column=0; column<imageW; column++) {	
					checkCudaErrors( cudaMemcpy(cell_devCU[1][0][theta][phase] + row * imageW + column, cell_devCUoneOriOnePhaseResult[0] + phase * (imageH * nOrient * imageW) + column * (imageH * nOrient) + theta * (imageH) + row, sizeof(float), cudaMemcpyDeviceToDevice) );
					checkCudaErrors( cudaMemcpy(cell_devCU[1][1][theta][phase] + row * imageW + column, cell_devCUoneOriOnePhaseResult[1] + phase * (imageH * nOrient * imageW) + column * (imageH * nOrient) + theta * (imageH) + row, sizeof(float), cudaMemcpyDeviceToDevice) );
			}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;

	cudaFree(cell_devCUoneOriOnePhase);
	for (int i=0; i<2; i++)
		cudaFree(cell_devCUoneOriOnePhaseResult[i]);
	free(cell_devCUoneOriOnePhaseResult);
	cudaFree(diagCosSin);
	cudaFree(diagCosMinusSin);
	free(diagCosSin_hostCU);
	free(diagCosMinusSin_hostCU);

	status = cublasDestroy(handle);
}

void Population::shiftSimpleAnswerOPENCV_repOriRepPhaseBlendLinear_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
			
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	int phaseIdx; 
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye*centralPhaseIdx;
		for (int phase=0; phase<nPhase; phase++) 
			if (phase!=phaseIdx)
				cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*nOrient*imageW,0,imageW*nOrient,2*imageH)).copyTo(cell_devCVrepOriRepPhase[eye](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,2*imageH)));
	}

	cv::gpu::blendLinear(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,f->sinPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[1]);
	cv::gpu::blendLinear(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,f->minusSinPhShift_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)));
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->sumCosMinusSinPhShift_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),1,-1);
	cv::gpu::multiply(tempCVrepOriRepPhase[1],f->sumCosSinPhShift_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),1,-1);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	int phaseIdx; 
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye*centralPhaseIdx;
		for (int phase=0; phase<nPhase; phase++) 
			if (phase!=phaseIdx)
				cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*nOrient*imageW,0,imageW*nOrient,2*imageH)).copyTo(cell_devCVrepOriRepPhase[eye](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,2*imageH)));
	}

	// reCos --> tempCVrepOriRepPhase[0]
	// reSin --> tempCVrepOriRepPhase[1]
	// imCos --> tempCVrepOriRepPhase[2]
	// imMinusSin --> tempCVrepOriRepPhase[3]
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[2]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->sinPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[1]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[0]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),f->minusSinPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[3]);
	cv::gpu::add(tempCVrepOriRepPhase[2],tempCVrepOriRepPhase[1],cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)));
	cv::gpu::add(tempCVrepOriRepPhase[0],tempCVrepOriRepPhase[3],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)));

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	int phaseIdx; 
	for (int eye=0; eye<nEye; eye++) {
		phaseIdx = eye*centralPhaseIdx;
		for (int phase=0; phase<nPhase; phase++) 
			if (phase!=phaseIdx)
				cell_devCVrepOriRepPhase[eye](cv::Rect(phaseIdx*nOrient*imageW,0,imageW*nOrient,2*imageH)).copyTo(cell_devCVrepOriRepPhase[eye](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,2*imageH)));
	}

	// reCos --> tempCVrepOriRepPhase[0]
	// reSin --> tempCVrepOriRepPhase[1]
	// imCos --> tempCVrepOriRepPhase[2]
	// imMinusSin --> tempCVrepOriRepPhase[3]
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[2],1.0,-1,strCV[0]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->sinPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[1],1.0,-1,strCV[1]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),f->cosPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[0],1.0,-1,strCV[2]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),f->minusSinPhShift_devCVrepOriRepPhase,tempCVrepOriRepPhase[3],1.0,-1,strCV[3]);
	strCV[1].waitForCompletion();
	cv::gpu::add(tempCVrepOriRepPhase[2],tempCVrepOriRepPhase[1],cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),mask0,-1,strCV[0]);
	strCV[3].waitForCompletion();
	cv::gpu::add(tempCVrepOriRepPhase[0],tempCVrepOriRepPhase[3],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),mask0,-1,strCV[2]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
				
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	int phaseIdx; 
	for (int eye=0;eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		for (int i=0; i<2; i++) 
			for(int phase=0; phase<nPhase; phase++) 
				if (phase!=phaseIdx)
					checkCudaErrors( cudaMemcpy(cell_devCUrepOriRepPhase[eye] + i * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient), cell_devCUrepOriRepPhase[eye] + i * (imageW * imageH * nPhase * nOrient) + phaseIdx * (imageW * imageH * nOrient) , nOrient * imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice) );
	}

	// non sovrascrivere tempCUrepOriRepPhase[0] che deve restare tutta nulla per il calcolo del centro di massa
	// reCos --> tempCUrepOriRepPhase[1]
	// reSin --> tempCUrepOriRepPhase[2]
	// imCos --> tempCUrepOriRepPhase[3]
	// imMinusSin --> tempCUrepOriRepPhase[4]
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),f->cosPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[3],imageW * imageH * nPhase * nOrient);
	nppsMul_32f(cell_devCUrepOriRepPhase[1],f->sinPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[2],imageW * imageH * nPhase * nOrient);
	nppsMul_32f(cell_devCUrepOriRepPhase[1],f->cosPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[1],imageW * imageH * nPhase * nOrient);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),f->minusSinPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[4],imageW * imageH * nPhase * nOrient);
	nppsAdd_32f(tempCUrepOriRepPhase[3],tempCUrepOriRepPhase[2],cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),imageW * imageH * nPhase * nOrient);
	nppsAdd_32f(tempCUrepOriRepPhase[1],tempCUrepOriRepPhase[4],cell_devCUrepOriRepPhase[1],imageW * imageH * nPhase * nOrient);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::shiftSimpleAnswerCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	int phaseIdx; 
	for (int eye=0;eye<nEye; eye++) {
		phaseIdx = eye * centralPhaseIdx;
		for (int i=0; i<2; i++) 
			for(int phase=0; phase<nPhase; phase++) 
				if (phase!=phaseIdx)
					checkCudaErrors( cudaMemcpy(cell_devCUrepOriRepPhase[eye] + i * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient), cell_devCUrepOriRepPhase[eye] + i * (imageW * imageH * nPhase * nOrient) + phaseIdx * (imageW * imageH * nOrient) , nOrient * imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice) );
	}

	// non sovrascrivere tempCUrepOriRepPhase[0] che deve restare tutta nulla per il calcolo del centro di massa
	// reCos --> tempCUrepOriRepPhase[1]
	// reSin --> tempCUrepOriRepPhase[2]
	// imCos --> tempCUrepOriRepPhase[3]
	// imMinusSin --> tempCUrepOriRepPhase[4]
	
	nppSetStream(strCU[0]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),f->cosPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[3],imageW * imageH * nPhase * nOrient);

	nppSetStream(strCU[1]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1],f->sinPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[2],imageW * imageH * nPhase * nOrient);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1], strCU[1]) );

	nppSetStream(strCU[2]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1],f->cosPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[1],imageW * imageH * nPhase * nOrient);

	nppSetStream(strCU[3]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),f->minusSinPhShift_devCUrepOriRepPhase,tempCUrepOriRepPhase[4],imageW * imageH * nPhase * nOrient);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[3], strCU[3]) );

	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[1],0) );
	nppSetStream(strCU[0]);
	nppsAdd_32f(tempCUrepOriRepPhase[3],tempCUrepOriRepPhase[2],cell_devCUrepOriRepPhase[1] + (imageW * imageH * nPhase * nOrient),imageW * imageH * nPhase * nOrient);

	checkCudaErrors( cudaStreamWaitEvent(strCU[2], kernelEventStrCU[3],0) );
	nppSetStream(strCU[2]);
	nppsAdd_32f(tempCUrepOriRepPhase[1],tempCUrepOriRepPhase[4],cell_devCUrepOriRepPhase[1],imageW * imageH * nPhase * nOrient);
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Cells' energy calculation
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::calcEnergyOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for(int theta = 0; theta < nOrient; theta++) 
		for (int phase = 0; phase < nPhase; phase++) {
			for (int i=0; i<2; i++) 
				cv::gpu::add(cell_devCV[0][i][theta][0], cell_devCV[1][i][theta][phase], cell_devCV[1][i][theta][phase]);
			cv::gpu::magnitude(cell_devCV[1][0][theta][phase], cell_devCV[1][1][theta][phase], cell_devCV[1][1][theta][phase]);
			cv::gpu::threshold(cell_devCV[1][1][theta][phase],cell_devCV[1][0][theta][phase],thEnergy,1,cv::THRESH_TOZERO);
		}
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for(int theta = 0; theta < nOrient; theta++) 
		for (int phase = 0; phase < nPhase; phase++) {
			for (int i = 0; i<2; i++) {
				nppsAdd_32f_I(cell_devCU[0][i][theta][0], cell_devCU[1][i][theta][phase], imageW*imageH);
				nppsSqr_32f_I(cell_devCU[1][i][theta][phase], imageW*imageH);
			}
			nppsAdd_32f_I(cell_devCU[1][0][theta][phase], cell_devCU[1][1][theta][phase], imageW*imageH);
			nppsSqrt_32f_I(cell_devCU[1][1][theta][phase], imageW*imageH);
			nppsThreshold_LTVal_32f(cell_devCU[1][1][theta][phase], cell_devCU[1][0][theta][phase], imageW*imageH, thEnergy, 0);
		}	
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {

		for (int i=0; i<2; i++) {
				cv::gpu::add(cell_devCV[0][i][0][0], cell_devCV[1][i][0][phase], cell_devCV[1][i][0][phase], mask0, -1, streamCV[1 + 2*i*nEye]);
				cv::gpu::add(cell_devCV[0][i][4][0], cell_devCV[1][i][4][phase], cell_devCV[1][i][4][phase], mask0, -1, streamCV[1 + nEye + 2*i*nEye]);
		}
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;		
			for (int i=0; i<2; i++) {
				cv::gpu::add(cell_devCV[0][i][resIdx1][0], cell_devCV[1][i][resIdx1][phase], cell_devCV[1][i][resIdx1][phase], mask0, -1, strCV[1*nCoupleOri+callIdx+2*i*nEye*nCoupleOri]);
				cv::gpu::add(cell_devCV[0][i][resIdx2][0], cell_devCV[1][i][resIdx2][phase], cell_devCV[1][i][resIdx2][phase], mask0, -1, strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*i*nEye*nCoupleOri]);
			}
		}

		streamCV[1 + 2*nEye].waitForCompletion();
		cv::gpu::magnitude(cell_devCV[1][0][0][phase], cell_devCV[1][1][0][phase], cell_devCV[1][1][0][phase], streamCV[1]);
		streamCV[1 + 3*nEye].waitForCompletion();
		cv::gpu::magnitude(cell_devCV[1][0][4][phase], cell_devCV[1][1][4][phase], cell_devCV[1][1][4][phase], streamCV[1 + nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			strCV[1*nCoupleOri+callIdx+2*nEye*nCoupleOri].waitForCompletion();
			cv::gpu::magnitude(cell_devCV[1][0][resIdx1][phase], cell_devCV[1][1][resIdx1][phase], cell_devCV[1][1][resIdx1][phase], strCV[1*nCoupleOri+callIdx]);
			strCV[1*nCoupleOri+callIdx+3*nEye*nCoupleOri].waitForCompletion();
			cv::gpu::magnitude(cell_devCV[1][0][resIdx2][phase], cell_devCV[1][1][resIdx2][phase], cell_devCV[1][1][resIdx2][phase], strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}

		cv::gpu::threshold(cell_devCV[1][1][0][phase], cell_devCV[1][0][0][phase], thEnergy, 1, cv::THRESH_TOZERO, streamCV[1]);
		cv::gpu::threshold(cell_devCV[1][1][4][phase], cell_devCV[1][0][4][phase], thEnergy, 1, cv::THRESH_TOZERO, streamCV[1 + nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			cv::gpu::threshold(cell_devCV[1][1][resIdx1][phase], cell_devCV[1][0][resIdx1][phase], thEnergy, 1, cv::THRESH_TOZERO, strCV[1*nCoupleOri+callIdx]);
			cv::gpu::threshold(cell_devCV[1][1][resIdx2][phase], cell_devCV[1][0][resIdx2][phase], thEnergy, 1, cv::THRESH_TOZERO, strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent){
	
	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {

		for (int i=0; i<2; i++) {
			nppSetStream(streamCU[1 + 2*i*nEye]);
			nppsAdd_32f_I(cell_devCU[0][i][0][0], cell_devCU[1][i][0][phase], imageW*imageH);
			nppSetStream(streamCU[1 + 2*i*nEye + nEye]);
			nppsAdd_32f_I(cell_devCU[0][i][4][0], cell_devCU[1][i][4][phase], imageW*imageH);
			nppSetStream(streamCU[1 + 2*i*nEye]);
			nppsSqr_32f_I(cell_devCU[1][i][0][phase], imageW*imageH);
			checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1 + 2*i*nEye], streamCU[1 + 2*i*nEye]) );
			nppSetStream(streamCU[1 + 2*i*nEye + nEye]);
			nppsSqr_32f_I(cell_devCU[1][i][4][phase], imageW*imageH);
			checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1 + 2*i*nEye + nEye], streamCU[1 + 2*i*nEye + nEye]) );
		}
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			for (int i=0; i<2; i++) {
				nppSetStream(strCU[1*nCoupleOri+callIdx+2*i*nEye*nCoupleOri]);
				nppsAdd_32f_I(cell_devCU[0][i][resIdx1][0], cell_devCU[1][i][resIdx1][phase], imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*i*nEye*nCoupleOri]);
				nppsAdd_32f_I(cell_devCU[0][i][resIdx2][0], cell_devCU[1][i][resIdx2][phase], imageW*imageH);
				nppSetStream(strCU[1*nCoupleOri+callIdx+2*i*nEye*nCoupleOri]);
				nppsSqr_32f_I(cell_devCU[1][i][resIdx1][phase], imageW*imageH);
				checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx+2*i*nEye*nCoupleOri], strCU[1*nCoupleOri+callIdx+2*i*nEye*nCoupleOri]) );
				nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*i*nEye*nCoupleOri]);
				nppsSqr_32f_I(cell_devCU[1][i][resIdx2][phase], imageW*imageH);
				checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*i*nEye*nCoupleOri], strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*i*nEye*nCoupleOri]) );
			}
		}

		checkCudaErrors( cudaStreamWaitEvent(streamCU[1], kernelEventStreamCU[1 + 2*nEye],0) );
		nppSetStream(streamCU[1]);
		nppsAdd_32f_I(cell_devCU[1][0][0][phase], cell_devCU[1][1][0][phase], imageW*imageH);
		checkCudaErrors( cudaStreamWaitEvent(streamCU[1 + nEye], kernelEventStreamCU[1 + 3*nEye],0) );
		nppSetStream(streamCU[1 + nEye]);
		nppsAdd_32f_I(cell_devCU[1][0][4][phase], cell_devCU[1][1][4][phase], imageW*imageH);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx], kernelEventStrCU[1*nCoupleOri+callIdx+2*nEye*nCoupleOri],0) );
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsAdd_32f_I(cell_devCU[1][0][resIdx1][phase], cell_devCU[1][1][resIdx1][phase], imageW*imageH);

			checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[1*nCoupleOri+callIdx+3*nEye*nCoupleOri],0) );
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f_I(cell_devCU[1][0][resIdx2][phase], cell_devCU[1][1][resIdx2][phase], imageW*imageH);
		}

		nppSetStream(streamCU[1]);
		nppsSqrt_32f_I(cell_devCU[1][1][0][phase], imageW*imageH);
		nppSetStream(streamCU[1 + nEye]);
		nppsSqrt_32f_I(cell_devCU[1][1][4][phase], imageW*imageH);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsSqrt_32f_I(cell_devCU[1][1][resIdx1][phase], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsSqrt_32f_I(cell_devCU[1][1][resIdx2][phase], imageW*imageH);
		}
		
		nppSetStream(streamCU[1]);
		nppsThreshold_LTVal_32f(cell_devCU[1][1][0][phase], cell_devCU[1][0][0][phase], imageW*imageH, thEnergy, 0);
		nppSetStream(streamCU[1 + nEye]);
		nppsThreshold_LTVal_32f(cell_devCU[1][1][4][phase], cell_devCU[1][0][4][phase], imageW*imageH, thEnergy, 0);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsThreshold_LTVal_32f(cell_devCU[1][1][resIdx1][phase], cell_devCU[1][0][resIdx1][phase], imageW*imageH, thEnergy, 0);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsThreshold_LTVal_32f(cell_devCU[1][1][resIdx2][phase], cell_devCU[1][0][resIdx2][phase], imageW*imageH, thEnergy, 0);
		}
	}
		
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcEnergyOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {
		for (int i = 0; i<2; i++) 
			cv::gpu::add(cell_devCVoneOri[0][i][0], cell_devCVoneOri[1][i][phase], cell_devCVoneOri[1][i][phase]);
		cv::gpu::magnitude(cell_devCVoneOri[1][0][phase], cell_devCVoneOri[1][1][phase], cell_devCVoneOri[1][1][phase]);
		cv::gpu::threshold(cell_devCVoneOri[1][1][phase],cell_devCVoneOri[1][0][phase],thEnergy,1,cv::THRESH_TOZERO);
	}	
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) {
		for (int i = 0; i<2; i++) {
			nppsAdd_32f_I(cell_devCUoneOri[0][i][0], cell_devCUoneOri[1][i][phase], imageW*nOrient*imageH);
			nppsSqr_32f_I(cell_devCUoneOri[1][i][phase], imageW*nOrient*imageH);
		}
		nppsAdd_32f_I(cell_devCUoneOri[1][0][phase], cell_devCUoneOri[1][1][phase], imageW*nOrient*imageH);
		nppsSqrt_32f_I(cell_devCUoneOri[1][1][phase], imageW*nOrient*imageH);
		nppsThreshold_LTVal_32f(cell_devCUoneOri[1][1][phase], cell_devCUoneOri[1][0][phase], imageW*nOrient*imageH, thEnergy, 0);
	}	
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) 
		for (int i = 0; i<2; i++) 
			cv::gpu::add(cell_devCVoneOri[0][i][0], cell_devCVoneOri[1][i][phase], cell_devCVoneOri[1][i][phase],mask0,-1,strCV[phase + i*nPhase]);
	for (int phase = 0; phase < nPhase; phase++) {
		strCV[phase+nPhase].waitForCompletion();
		cv::gpu::magnitude(cell_devCVoneOri[1][0][phase], cell_devCVoneOri[1][1][phase], cell_devCVoneOri[1][1][phase],strCV[phase]);
	}
	for (int phase = 0; phase < nPhase; phase++) 
		cv::gpu::threshold(cell_devCVoneOri[1][1][phase],cell_devCVoneOri[1][0][phase],thEnergy,1,cv::THRESH_TOZERO, strCV[phase]);
		
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase = 0; phase < nPhase; phase++) 
		for (int i = 0; i<2; i++) {
			nppSetStream(strCU[phase + i*nPhase]);
			nppsAdd_32f_I(cell_devCUoneOri[0][i][0], cell_devCUoneOri[1][i][phase], imageW*nOrient*imageH);
		}
	for (int phase = 0; phase < nPhase; phase++) 
		for (int i = 0; i<2; i++) {
			nppSetStream(strCU[phase + i*nPhase]);
			nppsSqr_32f_I(cell_devCUoneOri[1][i][phase], imageW*nOrient*imageH);
			checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase + i*nPhase], strCU[phase + i*nPhase]) );
		}
	for (int phase = 0; phase < nPhase; phase++) {
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase], kernelEventStrCU[phase+nPhase],0) );
		nppSetStream(strCU[phase]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][phase], cell_devCUoneOri[1][1][phase], imageW*nOrient*imageH);
	}
	for (int phase = 0; phase < nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsSqrt_32f_I(cell_devCUoneOri[1][1][phase], imageW*nOrient*imageH);
	}
	for (int phase = 0; phase < nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsThreshold_LTVal_32f(cell_devCUoneOri[1][1][phase], cell_devCUoneOri[1][0][phase], imageW*nOrient*imageH, thEnergy, 0);	
	}
		
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcEnergyOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
			
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::add(cell_devCVrepOriRepPhase[0],cell_devCVrepOriRepPhase[1],cell_devCVrepOriRepPhase[1]);
	cv::gpu::magnitude(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)));
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),thEnergy,1,cv::THRESH_TOZERO);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcEnergyCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppsAdd_32f_I(cell_devCUrepOriRepPhase[0],cell_devCUrepOriRepPhase[1],2*imageW*imageH*nPhase*nOrient);
	nppsSqr_32f(cell_devCUrepOriRepPhase[1],cell_devCUrepOriRepPhase[0],2*imageW*imageH*nPhase*nOrient);
	nppsAdd_32f(cell_devCUrepOriRepPhase[0], cell_devCUrepOriRepPhase[0] + imageW*imageH*nPhase*nOrient, cell_devCUrepOriRepPhase[1], imageW*imageH*nPhase*nOrient);
	nppsSqrt_32f(cell_devCUrepOriRepPhase[1], cell_devCUrepOriRepPhase[0], imageW*imageH*nPhase*nOrient);
	nppsThreshold_LTVal_32f(cell_devCUrepOriRepPhase[0], cell_devCUrepOriRepPhase[1], imageW*imageH*nPhase*nOrient, thEnergy, 0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Calculating center of mass (among phases) for each orientation
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::calcCenterOfMassOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) {
			f->filtGaussColumn_devCV->operator ()(cell_devCV[1][0][theta][phase],tempCV[theta+nOrient]);
			f->filtGaussRow_devCV->operator ()(tempCV[theta+nOrient],cell_devCV[1][0][theta][phase]);	
		}

	for (int theta = 0; theta<nOrient; theta++) {
		cv::gpu::max(cell_devCV[1][0][theta][0],cell_devCV[1][0][theta][1],tempCV[theta]);
		for (int phase=2; phase<nPhase; phase++) 
			cv::gpu::max(cell_devCV[1][0][theta][phase],tempCV[theta],tempCV[theta]);
	}

	for(int theta = 0; theta<nOrient; theta++) 
		for(int phase = 0; phase<nPhase; phase++) 
			cv::gpu::divide(cell_devCV[1][0][theta][phase],tempCV[theta],cell_devCV[1][0][theta][phase]);
	for(int theta = 0; theta<nOrient; theta++) 
		for(int phase = 0; phase<nPhase; phase++) 
			cv::gpu::threshold(cell_devCV[1][0][theta][phase],cell_devCV[1][0][theta][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO);
	
	for(int theta = 0; theta<nOrient; theta++) {
		cv::gpu::multiply(cell_devCV[1][0][theta][0],f->disp[0],tempCV[theta]);
		cv::gpu::multiply(cell_devCV[1][0][theta][1],f->disp[1],tempCV[theta+2*nOrient]);
		cv::gpu::add(tempCV[theta],tempCV[theta+2*nOrient],tempCV[theta]);
		cv::gpu::add(cell_devCV[1][0][theta][0],cell_devCV[1][0][theta][1],tempCV[theta+nOrient]);
		for(int phase = 2; phase<nPhase; phase++) {
			cv::gpu::multiply(cell_devCV[1][0][theta][phase],f->disp[phase],tempCV[theta+2*nOrient]);
			cv::gpu::add(tempCV[theta],tempCV[theta+2*nOrient],tempCV[theta]);
			cv::gpu::add(tempCV[theta+nOrient],cell_devCV[1][0][theta][phase],tempCV[theta+nOrient]);
		}
		cv::gpu::divide(tempCV[theta],tempCV[theta+nOrient],cell_devCV[1][0][theta][1]);
	}
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for(int theta = 0; theta<nOrient; theta++)
		for(int phase = 0; phase<nPhase; phase++) {
			convolutionColumnsGPU(tempCU[theta+nOrient], cell_devCU[1][0][theta][phase], imageW, imageH, filtGaussLength, nFilters1D);
			convolutionRowsGPU(cell_devCU[1][0][theta][phase], tempCU[theta+nOrient], imageW, imageH, filtGaussLength, nFilters1D);
		}

	for(int theta = 0; theta<nOrient; theta++) 
		for(int phase = 0; phase<nPhase; phase++) 
			nppsMaxEvery_32f_I(cell_devCU[1][0][theta][phase],tempCU[theta], imageW*imageH);

	for(int theta = 0; theta<nOrient; theta++) 
		for(int phase = 0; phase<nPhase; phase++) 
			nppsDiv_32f_I(tempCU[theta], cell_devCU[1][0][theta][phase], imageW*imageH);
	for(int theta = 0; theta<nOrient; theta++) 
		for(int phase = 0; phase<nPhase; phase++) 
			nppsThreshold_LTVal_32f_I(cell_devCU[1][0][theta][phase],imageW*imageH,thcalcCenterOfMass,0);
	
	for(int theta = 0; theta<nOrient; theta++) {
		nppsMulC_32f(cell_devCU[1][0][theta][0],f->disp[0],tempCU[theta], imageW*imageH);
		nppsMulC_32f(cell_devCU[1][0][theta][1],f->disp[1],tempCU[theta+2*nOrient], imageW*imageH);
		nppsAdd_32f_I(tempCU[theta+2*nOrient],tempCU[theta], imageW*imageH);
		nppsAdd_32f(cell_devCU[1][0][theta][0],cell_devCU[1][0][theta][1],tempCU[theta+nOrient], imageW*imageH);
		for(int phase = 2; phase<nPhase; phase++) {
			nppsMulC_32f(cell_devCU[1][0][theta][phase],f->disp[phase],tempCU[theta+2*nOrient], imageW*imageH);
			nppsAdd_32f_I(tempCU[theta+2*nOrient],tempCU[theta], imageW*imageH);
			nppsAdd_32f_I(cell_devCU[1][0][theta][phase],tempCU[theta+nOrient], imageW*imageH);
		}
		nppsDiv_32f(tempCU[theta+nOrient],tempCU[theta],cell_devCU[1][0][theta][1], imageW*imageH);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int resIdx1;
	int resIdx2;

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int phase=0; phase<nPhase; phase++) {
		f->filtGaussColumn_devCV->operator ()(cell_devCV[1][0][0][phase],tempCV[0+nOrient],streamCV[1]);
		f->filtGaussColumn_devCV->operator ()(cell_devCV[1][0][4][phase],tempCV[4+nOrient],streamCV[1+nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			f->filtGaussColumn_devCV->operator ()(cell_devCV[1][0][resIdx1][phase],tempCV[resIdx1+nOrient],strCV[1*nCoupleOri+callIdx]);
			f->filtGaussColumn_devCV->operator ()(cell_devCV[1][0][resIdx2][phase],tempCV[resIdx2+nOrient],strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
		f->filtGaussRow_devCV->operator ()(tempCV[0+nOrient],cell_devCV[1][0][0][phase],streamCV[1]);
		f->filtGaussRow_devCV->operator ()(tempCV[4+nOrient],cell_devCV[1][0][4][phase],streamCV[1+nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			f->filtGaussRow_devCV->operator ()(tempCV[resIdx1+nOrient],cell_devCV[1][0][resIdx1][phase],strCV[1*nCoupleOri+callIdx]);
			f->filtGaussRow_devCV->operator ()(tempCV[resIdx2+nOrient],cell_devCV[1][0][resIdx2][phase],strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}
	
	cv::gpu::max(cell_devCV[1][0][0][0],cell_devCV[1][0][0][1],tempCV[0],streamCV[1]);
	cv::gpu::max(cell_devCV[1][0][4][0],cell_devCV[1][0][4][1],tempCV[4],streamCV[1+nEye]);
	for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;	
		cv::gpu::max(cell_devCV[1][0][resIdx1][0],cell_devCV[1][0][resIdx1][1],tempCV[resIdx1],strCV[1*nCoupleOri+callIdx]);
		cv::gpu::max(cell_devCV[1][0][resIdx2][0],cell_devCV[1][0][resIdx2][1],tempCV[resIdx2],strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	for (int phase=2; phase<nPhase; phase++) {
		cv::gpu::max(cell_devCV[1][0][0][phase],tempCV[0],tempCV[0],streamCV[1]);
		cv::gpu::max(cell_devCV[1][0][4][phase],tempCV[4],tempCV[4],streamCV[1+nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			cv::gpu::max(cell_devCV[1][0][resIdx1][phase],tempCV[resIdx1],tempCV[resIdx1],strCV[1*nCoupleOri+callIdx]);
			cv::gpu::max(cell_devCV[1][0][resIdx2][phase],tempCV[resIdx2],tempCV[resIdx2],strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}

	for (int phase = 0; phase<nPhase; phase++) {
		cv::gpu::divide(cell_devCV[1][0][0][phase],tempCV[0],cell_devCV[1][0][0][phase],1.0,-1,streamCV[1]);
		cv::gpu::divide(cell_devCV[1][0][4][phase],tempCV[4],cell_devCV[1][0][4][phase],1.0,-1,streamCV[1+nEye]);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			cv::gpu::divide(cell_devCV[1][0][resIdx1][phase],tempCV[resIdx1],cell_devCV[1][0][resIdx1][phase],1.0,-1,strCV[1*nCoupleOri+callIdx]);
			cv::gpu::divide(cell_devCV[1][0][resIdx2][phase],tempCV[resIdx2],cell_devCV[1][0][resIdx2][phase],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			}
	}

	for (int phase = 0; phase<nPhase; phase++) {
		cv::gpu::threshold(cell_devCV[1][0][0][phase],cell_devCV[1][0][0][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO,streamCV[1]);
		cv::gpu::threshold(cell_devCV[1][0][4][phase],cell_devCV[1][0][4][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO,streamCV[1+nEye]);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			cv::gpu::threshold(cell_devCV[1][0][resIdx1][phase],cell_devCV[1][0][resIdx1][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO,strCV[1*nCoupleOri+callIdx]);
			cv::gpu::threshold(cell_devCV[1][0][resIdx2][phase],cell_devCV[1][0][resIdx2][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}

	cv::gpu::multiply(cell_devCV[1][0][0][0],f->disp[0],tempCV[0],1.0,-1,streamCV[1]);
	cv::gpu::multiply(cell_devCV[1][0][0][1],f->disp[1],tempCV[0+2*nOrient],1.0,-1,streamCV[1]);
	cv::gpu::multiply(cell_devCV[1][0][4][0],f->disp[0],tempCV[4],1.0,-1,streamCV[1+nEye]);
	cv::gpu::multiply(cell_devCV[1][0][4][1],f->disp[1],tempCV[4+2*nOrient],1.0,-1,streamCV[1+nEye]);
	cv::gpu::add(tempCV[0],tempCV[0+2*nOrient],tempCV[0],mask0,-1,streamCV[1]);
	cv::gpu::add(tempCV[4],tempCV[4+2*nOrient],tempCV[4],mask0,-1,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][0],f->disp[0],tempCV[resIdx1],1.0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][1],f->disp[1],tempCV[resIdx1+2*nOrient],1.0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][0],f->disp[0],tempCV[resIdx2],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][1],f->disp[1],tempCV[resIdx2+2*nOrient],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		cv::gpu::add(tempCV[resIdx1],tempCV[resIdx1+2*nOrient],tempCV[resIdx1],mask0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::add(tempCV[resIdx2],tempCV[resIdx2+2*nOrient],tempCV[resIdx2],mask0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	for (int phase = 2; phase<nPhase; phase++) {
		cv::gpu::multiply(cell_devCV[1][0][0][phase],f->disp[phase],tempCV[0+2*nOrient],1.0,-1,streamCV[1]);
		cv::gpu::multiply(cell_devCV[1][0][4][phase],f->disp[phase],tempCV[4+2*nOrient],1.0,-1,streamCV[1+nEye]);
		cv::gpu::add(tempCV[0],tempCV[0+2*nOrient],tempCV[0],mask0,-1,streamCV[1]);
		cv::gpu::add(tempCV[4],tempCV[4+2*nOrient],tempCV[4],mask0,-1,streamCV[1+nEye]);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			cv::gpu::multiply(cell_devCV[1][0][resIdx1][phase],f->disp[phase],tempCV[resIdx1+2*nOrient],1.0,-1,strCV[1*nCoupleOri+callIdx]);
			cv::gpu::multiply(cell_devCV[1][0][resIdx2][phase],f->disp[phase],tempCV[resIdx2+2*nOrient],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			cv::gpu::add(tempCV[resIdx1],tempCV[resIdx1+2*nOrient],tempCV[resIdx1],mask0,-1,strCV[1*nCoupleOri+callIdx]);
			cv::gpu::add(tempCV[resIdx2],tempCV[resIdx2+2*nOrient],tempCV[resIdx2],mask0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}

	cv::gpu::add(cell_devCV[1][0][0][0],cell_devCV[1][0][0][1],tempCV[0+nOrient],mask0,-1,streamCV[1]);
	cv::gpu::add(cell_devCV[1][0][4][0],cell_devCV[1][0][4][1],tempCV[4+nOrient],mask0,-1,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::add(cell_devCV[1][0][resIdx1][0],cell_devCV[1][0][resIdx1][1],tempCV[resIdx1+nOrient],mask0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::add(cell_devCV[1][0][resIdx2][0],cell_devCV[1][0][resIdx2][1],tempCV[resIdx2+nOrient],mask0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	for (int phase = 2; phase<nPhase; phase++) {
		cv::gpu::add(tempCV[0+nOrient],cell_devCV[1][0][0][phase],tempCV[0+nOrient],mask0,-1,streamCV[1]);
		cv::gpu::add(tempCV[4+nOrient],cell_devCV[1][0][4][phase],tempCV[4+nOrient],mask0,-1,streamCV[1+nEye]);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			cv::gpu::add(tempCV[resIdx1+nOrient],cell_devCV[1][0][resIdx1][phase],tempCV[resIdx1+nOrient],mask0,-1,strCV[1*nCoupleOri+callIdx]);
			cv::gpu::add(tempCV[resIdx2+nOrient],cell_devCV[1][0][resIdx2][phase],tempCV[resIdx2+nOrient],mask0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}

	cv::gpu::divide(tempCV[0],tempCV[0+nOrient],cell_devCV[1][0][0][1],1.0,-1,streamCV[1]);
	cv::gpu::divide(tempCV[4],tempCV[4+nOrient],cell_devCV[1][0][4][1],1.0,-1,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::divide(tempCV[resIdx1],tempCV[resIdx1+nOrient],cell_devCV[1][0][resIdx1][1],1.0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::divide(tempCV[resIdx2],tempCV[resIdx2+nOrient],cell_devCV[1][0][resIdx2][1],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	int resIdx1;
	int resIdx2;

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase=0; phase<nPhase; phase++) {
		convolutionColumnsGPU(tempCU[0+nOrient], cell_devCU[1][0][0][phase], imageW, imageH, filtGaussLength, nFilters1D,streamCU[1]);
		convolutionColumnsGPU(tempCU[4+nOrient], cell_devCU[1][0][4][phase], imageW, imageH, filtGaussLength, nFilters1D,streamCU[1+nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			convolutionColumnsGPU(tempCU[resIdx1+nOrient], cell_devCU[1][0][resIdx1][phase], imageW, imageH, filtGaussLength, nFilters1D,strCU[1*nCoupleOri+callIdx]);
			convolutionColumnsGPU(tempCU[resIdx2+nOrient], cell_devCU[1][0][resIdx2][phase], imageW, imageH, filtGaussLength, nFilters1D,strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
		convolutionRowsGPU(cell_devCU[1][0][0][phase], tempCU[0+nOrient], imageW, imageH, filtGaussLength, nFilters1D,streamCU[1]);
		convolutionRowsGPU(cell_devCU[1][0][4][phase], tempCU[4+nOrient], imageW, imageH, filtGaussLength, nFilters1D,streamCU[1+nEye]);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			convolutionRowsGPU(cell_devCU[1][0][resIdx1][phase], tempCU[resIdx1+nOrient], imageW, imageH, filtGaussLength, nFilters1D,strCU[1*nCoupleOri+callIdx]);
			convolutionRowsGPU(cell_devCU[1][0][resIdx2][phase], tempCU[resIdx2+nOrient], imageW, imageH, filtGaussLength, nFilters1D,strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		}
	}
	
	for (int phase=0; phase<nPhase; phase++) {
		nppSetStream(streamCU[1]);
		nppsMaxEvery_32f_I(cell_devCU[1][0][0][phase],tempCU[0], imageW*imageH);
		nppSetStream(streamCU[1+nEye]);
		nppsMaxEvery_32f_I(cell_devCU[1][0][4][phase],tempCU[4], imageW*imageH);
		for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsMaxEvery_32f_I(cell_devCU[1][0][resIdx1][phase],tempCU[resIdx1], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsMaxEvery_32f_I(cell_devCU[1][0][resIdx2][phase],tempCU[resIdx2], imageW*imageH);
		}
	}

	for (int phase = 0; phase<nPhase; phase++) {
		nppSetStream(streamCU[1]);
		nppsDiv_32f_I(tempCU[0], cell_devCU[1][0][0][phase], imageW*imageH);
		nppSetStream(streamCU[1+nEye]);
		nppsDiv_32f_I(tempCU[4], cell_devCU[1][0][4][phase], imageW*imageH);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsDiv_32f_I(tempCU[resIdx1], cell_devCU[1][0][resIdx1][phase], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsDiv_32f_I(tempCU[resIdx2], cell_devCU[1][0][resIdx2][phase], imageW*imageH);
			}
	}

	for (int phase = 0; phase<nPhase; phase++) {
		nppSetStream(streamCU[1]);
		nppsThreshold_LTVal_32f_I(cell_devCU[1][0][0][phase],imageW*imageH,thcalcCenterOfMass,0);
		nppSetStream(streamCU[1+nEye]);
		nppsThreshold_LTVal_32f_I(cell_devCU[1][0][4][phase],imageW*imageH,thcalcCenterOfMass,0);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;	
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsThreshold_LTVal_32f_I(cell_devCU[1][0][resIdx1][phase],imageW*imageH,thcalcCenterOfMass,0);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsThreshold_LTVal_32f_I(cell_devCU[1][0][resIdx2][phase],imageW*imageH,thcalcCenterOfMass,0);
		}
	}

	nppSetStream(streamCU[1]);
	nppsMulC_32f(cell_devCU[1][0][0][0],f->disp[0],tempCU[0], imageW*imageH);
	nppSetStream(streamCU[1]);
	nppsMulC_32f(cell_devCU[1][0][0][1],f->disp[1],tempCU[0+2*nOrient], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsMulC_32f(cell_devCU[1][0][4][0],f->disp[0],tempCU[4], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsMulC_32f(cell_devCU[1][0][4][1],f->disp[1],tempCU[4+2*nOrient], imageW*imageH);
	nppSetStream(streamCU[1]);
	nppsAdd_32f_I(tempCU[0+2*nOrient],tempCU[0], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsAdd_32f_I(tempCU[4+2*nOrient],tempCU[4], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsMulC_32f(cell_devCU[1][0][resIdx1][0],f->disp[0],tempCU[resIdx1], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsMulC_32f(cell_devCU[1][0][resIdx1][1],f->disp[1],tempCU[resIdx1+2*nOrient], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsMulC_32f(cell_devCU[1][0][resIdx2][0],f->disp[0],tempCU[resIdx2], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsMulC_32f(cell_devCU[1][0][resIdx2][1],f->disp[1],tempCU[resIdx2+2*nOrient], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsAdd_32f_I(tempCU[resIdx1+2*nOrient],tempCU[resIdx1], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsAdd_32f_I(tempCU[resIdx2+2*nOrient],tempCU[resIdx2], imageW*imageH);
	}
	for (int phase = 2; phase<nPhase; phase++) {
		nppSetStream(streamCU[1]);
		nppsMulC_32f(cell_devCU[1][0][0][phase],f->disp[phase],tempCU[0+2*nOrient], imageW*imageH);
		nppSetStream(streamCU[1+nEye]);
		nppsMulC_32f(cell_devCU[1][0][4][phase],f->disp[phase],tempCU[4+2*nOrient], imageW*imageH);
		nppSetStream(streamCU[1]);
		nppsAdd_32f_I(tempCU[0+2*nOrient],tempCU[0], imageW*imageH);
		nppSetStream(streamCU[1+nEye]);
		nppsAdd_32f_I(tempCU[4+2*nOrient],tempCU[4], imageW*imageH);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsMulC_32f(cell_devCU[1][0][resIdx1][phase],f->disp[phase],tempCU[resIdx1+2*nOrient], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsMulC_32f(cell_devCU[1][0][resIdx2][phase],f->disp[phase],tempCU[resIdx2+2*nOrient], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsAdd_32f_I(tempCU[resIdx1+2*nOrient],tempCU[resIdx1], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f_I(tempCU[resIdx2+2*nOrient],tempCU[resIdx2], imageW*imageH);
		}
	}

	nppSetStream(streamCU[1]);
	nppsAdd_32f(cell_devCU[1][0][0][0],cell_devCU[1][0][0][1],tempCU[0+nOrient], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsAdd_32f(cell_devCU[1][0][4][0],cell_devCU[1][0][4][1],tempCU[4+nOrient], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsAdd_32f(cell_devCU[1][0][resIdx1][0],cell_devCU[1][0][resIdx1][1],tempCU[resIdx1+nOrient], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsAdd_32f(cell_devCU[1][0][resIdx2][0],cell_devCU[1][0][resIdx2][1],tempCU[resIdx2+nOrient], imageW*imageH);
	}
	for (int phase = 2; phase<nPhase; phase++) {
		nppSetStream(streamCU[1]);
		nppsAdd_32f_I(cell_devCU[1][0][0][phase],tempCU[0+nOrient], imageW*imageH);
		nppSetStream(streamCU[1+nEye]);
		nppsAdd_32f_I(cell_devCU[1][0][4][phase],tempCU[4+nOrient], imageW*imageH);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
			resIdx1 = callIdx+1;
			resIdx2 = nOrient-1-callIdx;
			nppSetStream(strCU[1*nCoupleOri+callIdx]);
			nppsAdd_32f_I(cell_devCU[1][0][resIdx1][phase],tempCU[resIdx1+nOrient], imageW*imageH);
			nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
			nppsAdd_32f_I(cell_devCU[1][0][resIdx2][phase],tempCU[resIdx2+nOrient], imageW*imageH);
		}
	}

	nppSetStream(streamCU[1]);
	nppsDiv_32f(tempCU[0+nOrient],tempCU[0],cell_devCU[1][0][0][1], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsDiv_32f(tempCU[4+nOrient],tempCU[4],cell_devCU[1][0][4][1], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsDiv_32f(tempCU[resIdx1+nOrient],tempCU[resIdx1],cell_devCU[1][0][resIdx1][1], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsDiv_32f(tempCU[resIdx2+nOrient],tempCU[resIdx2],cell_devCU[1][0][resIdx2][1], imageW*imageH);
	}
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcCenterOfMassOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );
	
	for (int phase=0; phase<nPhase; phase++) 
		f->filtGaussColumn_devCV->operator ()(cell_devCVoneOri[1][0][phase],tempCVoneOri[phase]);
	for (int phase=0; phase<nPhase; phase++) 
		f->filtGaussRow_devCV->operator ()(tempCVoneOri[phase],cell_devCVoneOri[1][0][phase]);	

	cv::gpu::max(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][1],tempCVoneOri[0]);
	for(int phase = 2; phase<nPhase; phase++) 
		cv::gpu::max(cell_devCVoneOri[1][0][phase],tempCVoneOri[0],tempCVoneOri[0]);

	for(int phase = 0; phase<nPhase; phase++)
		cv::gpu::divide(cell_devCVoneOri[1][0][phase],tempCVoneOri[0],cell_devCVoneOri[1][0][phase]);
	for(int phase = 0; phase<nPhase; phase++)
		cv::gpu::threshold(cell_devCVoneOri[1][0][phase],cell_devCVoneOri[1][0][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO);
	for(int phase = 0; phase<nPhase; phase++)
		cv::gpu::multiply(cell_devCVoneOri[1][0][phase],f->disp[phase],tempCVoneOri[phase]);
	
	cv::gpu::add(tempCVoneOri[0],tempCVoneOri[1],tempCVoneOri[nPhase]);
	cv::gpu::add(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][1],tempCVoneOri[nPhase+1]);
	for(int phase = 2; phase<nPhase; phase++) {
		cv::gpu::add(tempCVoneOri[nPhase],tempCVoneOri[phase],tempCVoneOri[nPhase]);
		cv::gpu::add(tempCVoneOri[nPhase+1],cell_devCVoneOri[1][0][phase],tempCVoneOri[nPhase+1]);
	}
	cv::gpu::divide(tempCVoneOri[nPhase],tempCVoneOri[nPhase+1],cell_devCVoneOri[1][0][1]);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionColumnsGPU(tempCUoneOri[phase+1]+imageW*imageH*theta, cell_devCUoneOri[1][0][phase]+imageW*imageH*theta, imageW, imageH, filtGaussLength, nFilters1D);
	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionRowsGPU(cell_devCUoneOri[1][0][phase]+imageW*imageH*theta, tempCUoneOri[phase+1]+imageW*imageH*theta, imageW, imageH, filtGaussLength, nFilters1D);
	
	for(int phase = 0; phase<nPhase; phase++) 
		nppsMaxEvery_32f_I(cell_devCUoneOri[1][0][phase],tempCUoneOri[0], nOrient*imageW*imageH);

	for(int phase = 0; phase<nPhase; phase++) 
		nppsDiv_32f_I(tempCUoneOri[0], cell_devCUoneOri[1][0][phase], nOrient*imageW*imageH);
	for(int phase = 0; phase<nPhase; phase++) 
		nppsThreshold_LTVal_32f_I(cell_devCUoneOri[1][0][phase],nOrient*imageW*imageH,thcalcCenterOfMass,0);
	for(int phase = 0; phase<nPhase; phase++) 
		nppsMulC_32f(cell_devCUoneOri[1][0][phase],f->disp[phase],tempCUoneOri[phase], nOrient*imageW*imageH);
	
	nppsAdd_32f(tempCUoneOri[0],tempCUoneOri[1],tempCUoneOri[nPhase], nOrient*imageW*imageH);
	nppsAdd_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],tempCUoneOri[nPhase+1], nOrient*imageW*imageH);
	for(int phase = 2; phase<nPhase; phase++) {
		nppsAdd_32f_I(tempCUoneOri[phase],tempCUoneOri[nPhase], nOrient*imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][phase],tempCUoneOri[nPhase+1], nOrient*imageW*imageH);
	}
	nppsDiv_32f(tempCUoneOri[nPhase+1],tempCUoneOri[nPhase],cell_devCUoneOri[1][0][1], nOrient*imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int phase=0; phase<nPhase; phase++)
		f->filtGaussColumn_devCV->operator ()(cell_devCVoneOri[1][0][phase],tempCVoneOri[phase],strCV[phase]);
	for (int phase=0; phase<nPhase; phase++) 
		f->filtGaussRow_devCV->operator ()(tempCVoneOri[phase],cell_devCVoneOri[1][0][phase],strCV[phase]);	

	strCV[0].waitForCompletion();
	cv::gpu::max(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][1],tempCVoneOri[0],strCV[1]);
	for(int phase = 2; phase<nPhase; phase++) {
		strCV[phase-1].waitForCompletion();
		cv::gpu::max(cell_devCVoneOri[1][0][phase],tempCVoneOri[0],tempCVoneOri[0],strCV[phase]);
	}

	for(int phase = 0; phase<nPhase; phase++) 
		cv::gpu::divide(cell_devCVoneOri[1][0][phase],tempCVoneOri[0],cell_devCVoneOri[1][0][phase],1.0,-1,strCV[phase]);
	for(int phase = 0; phase<nPhase; phase++) 
		cv::gpu::threshold(cell_devCVoneOri[1][0][phase],cell_devCVoneOri[1][0][phase],thcalcCenterOfMass,1,cv::THRESH_TOZERO,strCV[phase]);
	for(int phase = 0; phase<nPhase; phase++) 
		cv::gpu::multiply(cell_devCVoneOri[1][0][phase],f->disp[phase],tempCVoneOri[phase],1.0,-1,strCV[phase]);
	
	strCV[0].waitForCompletion();
	strCV[1].waitForCompletion();
	cv::gpu::add(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][1],tempCVoneOri[nPhase+1],mask0,-1,strCV[1+nPhase]);
	cv::gpu::add(tempCVoneOri[0],tempCVoneOri[1],tempCVoneOri[nPhase],mask0,-1,strCV[1]);
	for(int phase = 2; phase<nPhase; phase++) {
		
		strCV[phase+nPhase-1].waitForCompletion();
		cv::gpu::add(tempCVoneOri[nPhase+1],cell_devCVoneOri[1][0][phase],tempCVoneOri[nPhase+1],mask0,-1,strCV[phase+nPhase]);
		
		strCV[phase-1].waitForCompletion();
		cv::gpu::add(tempCVoneOri[nPhase],tempCVoneOri[phase],tempCVoneOri[nPhase],mask0,-1,strCV[phase]);
	}

	strCV[nPhase+nPhase-1].waitForCompletion();
	strCV[nPhase-1].waitForCompletion();
	cv::gpu::divide(tempCVoneOri[nPhase],tempCVoneOri[nPhase+1],cell_devCVoneOri[1][0][1],1.0,-1,strCV[0]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionColumnsGPU(tempCUoneOri[phase+1]+imageW*imageH*theta, cell_devCUoneOri[1][0][phase]+imageW*imageH*theta, imageW, imageH, filtGaussLength, nFilters1D,strCU[phase]);
	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionRowsGPU(cell_devCUoneOri[1][0][phase]+imageW*imageH*theta, tempCUoneOri[phase+1]+imageW*imageH*theta, imageW, imageH, filtGaussLength, nFilters1D,strCU[phase]);
		
	nppSetStream(strCU[0]);
	nppsMaxEvery_32f_I(cell_devCUoneOri[1][0][0],tempCUoneOri[0], nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[0], strCU[0]) );
	for(int phase = 1; phase<nPhase; phase++) {
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase], kernelEventStrCU[phase-1],0) );
		nppSetStream(strCU[phase]);
		nppsMaxEvery_32f_I(cell_devCUoneOri[1][0][phase],tempCUoneOri[0], nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}

	for(int phase = 0; phase<nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsDiv_32f_I(tempCUoneOri[0], cell_devCUoneOri[1][0][phase], nOrient*imageW*imageH);
	}
	for(int phase = 0; phase<nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsThreshold_LTVal_32f_I(cell_devCUoneOri[1][0][phase],nOrient*imageW*imageH,thcalcCenterOfMass,0);
	}
	for(int phase = 0; phase<nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsMulC_32f(cell_devCUoneOri[1][0][phase],f->disp[phase],tempCUoneOri[phase], nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}
		
	checkCudaErrors( cudaStreamWaitEvent(strCU[1+nPhase], kernelEventStrCU[0],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[1+nPhase], kernelEventStrCU[1],0) );
	nppSetStream(strCU[1+nPhase]);
	nppsAdd_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],tempCUoneOri[nPhase+1], nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1+nPhase], strCU[1+nPhase]) );
	nppSetStream(strCU[1]);
	nppsAdd_32f(tempCUoneOri[0],tempCUoneOri[1],tempCUoneOri[nPhase], nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1], strCU[1]) );
	for(int phase = 2; phase<nPhase; phase++) {
		
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase+nPhase], kernelEventStrCU[phase+nPhase-1],0) );
		nppSetStream(strCU[phase+nPhase]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][phase],tempCUoneOri[nPhase+1], nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase+nPhase], strCU[phase+nPhase]) );
		
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase], kernelEventStrCU[phase-1],0) );
		nppSetStream(strCU[phase]);
		nppsAdd_32f_I(tempCUoneOri[phase],tempCUoneOri[nPhase], nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[nPhase-1],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[nPhase+nPhase-1],0) );
	nppSetStream(strCU[0]);
	nppsDiv_32f(tempCUoneOri[nPhase+1],tempCUoneOri[nPhase],cell_devCUoneOri[1][0][1], nOrient*imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::calcCenterOfMassOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	f->filtGaussColumn_devCV->operator ()(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),tempCVrepOriRepPhase[0]);
	f->filtGaussRow_devCV->operator ()(tempCVrepOriRepPhase[0],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)));	

	cv::gpu::max(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)));
	for(int phase = 2; phase<nPhase; phase++) 
		cv::gpu::max(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)));
	for(int phase = 1; phase<nPhase; phase++) 
		tempCVrepOriRepPhase[0](cv::Rect(0, 0, imageW*nOrient, imageH)).copyTo(tempCVrepOriRepPhase[0](cv::Rect(phase*imageW*nOrient, 0, imageW*nOrient, imageH)));

	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),tempCVrepOriRepPhase[0],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)));
	cv::gpu::threshold(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),thcalcCenterOfMass,1,cv::THRESH_TOZERO);
	
	for(int phase = 0; phase<nPhase; phase++) 
		cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),f->disp[phase],tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)));

	cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)));
	cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)));
	for(int phase = 2; phase<nPhase; phase++) {
		cv::gpu::add(tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)));
		cv::gpu::add(tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)));
	}

	cv::gpu::divide(tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(imageW*nOrient,0,imageW*nOrient,imageH)));
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) {
			convolutionColumnsGPU(tempCUrepOriRepPhase[1] + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), imageW, imageH, filtGaussLength, nFilters1D);
			convolutionRowsGPU(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), tempCUrepOriRepPhase[1] + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), imageW, imageH, filtGaussLength, nFilters1D);
		}

	for(int phase = 0; phase<nPhase; phase++) 
		nppsMaxEvery_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	for(int phase = 1; phase<nPhase; phase++) 
		checkCudaErrors( cudaMemcpy(tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient), tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient), nOrient * imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice) );
		
	nppsDiv_32f_I(tempCUrepOriRepPhase[0], cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient), imageW * imageH * nOrient * nPhase);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient), nOrient * imageW * imageH * nPhase, thcalcCenterOfMass, 0);	
	
	for(int phase = 0; phase<nPhase; phase++) 
		nppsMulC_32f(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient), f->disp[phase], tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient), nOrient * imageW * imageH);
	
	nppsAdd_32f(tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[0] + 1 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	nppsAdd_32f(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 0 * (imageW * imageH * nOrient),cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 1 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	for(int phase = 2; phase<nPhase; phase++) {
		nppsAdd_32f_I(tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	}

	nppsDiv_32f(tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient),cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 1 * (imageW * imageH * nOrient), nOrient * imageW * imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	f->filtGaussColumn_devCV->operator ()(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),tempCVrepOriRepPhase[0],strCV[0]);
	f->filtGaussRow_devCV->operator ()(tempCVrepOriRepPhase[0],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),strCV[0]);	
	
	strCV[0].waitForCompletion();
	cv::gpu::max(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)),strCV[1]);
	for (int phase = 2; phase<nPhase; phase++) {
		strCV[phase-1].waitForCompletion();
		cv::gpu::max(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(0,0,imageW*nOrient,imageH)),strCV[phase]);
	}
	for (int phase = 1; phase<nPhase; phase++) 
		tempCVrepOriRepPhase[0](cv::Rect(0, 0, imageW*nOrient, imageH)).copyTo(tempCVrepOriRepPhase[0](cv::Rect(phase*imageW*nOrient, 0, imageW*nOrient, imageH)));

	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),tempCVrepOriRepPhase[0],cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),1.0,-1,strCV[0]);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nPhase*nOrient,imageH)),thcalcCenterOfMass,1,cv::THRESH_TOZERO,strCV[0]);
	
	strCV[0].waitForCompletion();
	for (int phase = 0; phase<nPhase; phase++) 
		cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),f->disp[phase],tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),1.0,-1,strCV[phase]);

	strCV[0].waitForCompletion();
	strCV[1].waitForCompletion();
	cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),mask0,-1,strCV[1+nPhase]);
	cv::gpu::add(tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*1,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),mask0,-1,strCV[1]);
	for (int phase = 2; phase<nPhase; phase++) {
		strCV[phase+nPhase-1].waitForCompletion();
		cv::gpu::add(tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),mask0,-1,strCV[phase+nPhase]);
		strCV[phase-1].waitForCompletion();
		cv::gpu::add(tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[0](cv::Rect(nOrient*imageW*phase,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),mask0,-1,strCV[phase]);
	}

	strCV[nPhase+nPhase-1].waitForCompletion();
	strCV[nPhase-1].waitForCompletion();
	cv::gpu::divide(tempCVrepOriRepPhase[1](cv::Rect(0,0,imageW*nOrient,imageH)),tempCVrepOriRepPhase[2](cv::Rect(0,0,imageW*nOrient,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(imageW*nOrient,0,imageW*nOrient,imageH)),1.0,-1,strCV[0]);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::calcCenterOfMassCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent) {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionColumnsGPU(tempCUrepOriRepPhase[1] + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), imageW, imageH, filtGaussLength, nFilters1D,strCU[phase]);
	for (int theta=0; theta<nOrient; theta++)
		for (int phase=0; phase<nPhase; phase++) 
			convolutionRowsGPU(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), tempCUrepOriRepPhase[1] + phase * (imageW * imageH * nOrient) + theta * (imageW * imageH), imageW, imageH, filtGaussLength, nFilters1D,strCU[phase]);
	
	nppSetStream(strCU[0]);
	nppsMaxEvery_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 0 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[0], strCU[0]) );
	for(int phase = 1; phase<nPhase; phase++) {
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase], kernelEventStrCU[phase-1],0) );
		nppSetStream(strCU[phase]);
		nppsMaxEvery_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}
	for(int phase = 1; phase<nPhase; phase++) 
		checkCudaErrors( cudaMemcpy(tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient), tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient), nOrient * imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice) );
	
	nppSetStream(strCU[0]);
	nppsDiv_32f_I(tempCUrepOriRepPhase[0], cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient), imageW * imageH * nOrient * nPhase);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient), nOrient * imageW * imageH * nPhase, thcalcCenterOfMass, 0);	
	
	for(int phase = 0; phase<nPhase; phase++) {
		nppSetStream(strCU[phase]);
		nppsMulC_32f(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient), f->disp[phase], tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient), nOrient * imageW * imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}
		
	checkCudaErrors( cudaStreamWaitEvent(strCU[1+nPhase], kernelEventStrCU[0],0) );
	nppSetStream(strCU[1+nPhase]);
	nppsAdd_32f(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 0 * (imageW * imageH * nOrient),cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 1 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1+nPhase], strCU[1+nPhase]) );
	nppSetStream(strCU[1]);
	nppsAdd_32f(tempCUrepOriRepPhase[0] + 0 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[0] + 1 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1], strCU[1]) );
	for(int phase = 2; phase<nPhase; phase++) {
	
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase+nPhase], kernelEventStrCU[phase+nPhase-1],0) );
		nppSetStream(strCU[phase+nPhase]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase+nPhase], strCU[phase+nPhase]) );
		
		checkCudaErrors( cudaStreamWaitEvent(strCU[phase], kernelEventStrCU[phase-1],0) );
		nppSetStream(strCU[phase]);
		nppsAdd_32f_I(tempCUrepOriRepPhase[0] + phase * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient), nOrient*imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[phase], strCU[phase]) );
	}
	
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[nPhase-1],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[nPhase+nPhase-1],0) );
	nppSetStream(strCU[0]);
	nppsDiv_32f(tempCUrepOriRepPhase[2] + 0 * (imageW * imageH * nOrient),tempCUrepOriRepPhase[1] + 0 * (imageW * imageH * nOrient),cell_devCUrepOriRepPhase[1] + 0 * (imageW * imageH * nPhase * nOrient) + 1 * (imageW * imageH * nOrient), nOrient * imageW * imageH);
	
	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Project center of mass to horizontal/vertical axes
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::projectXYCenterOfMassOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0; theta<nOrient; theta++) {
		cv::gpu::multiply(cell_devCV[1][0][theta][1],f->cosOriTuning[theta],cell_devCV[1][0][theta][0]);
		cv::gpu::multiply(cell_devCV[1][0][theta][1],f->sinOriTuning[theta],cell_devCV[1][1][theta][0]);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	for (int theta=0;theta<nOrient; theta++) {
		nppsMulC_32f(cell_devCU[1][0][theta][1],f->cosOriTuning[theta],cell_devCU[1][0][theta][0], imageW*imageH);
		nppsMulC_32f(cell_devCU[1][0][theta][1],f->sinOriTuning[theta],cell_devCU[1][1][theta][0], imageW*imageH);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::multiply(cell_devCV[1][0][0][1],f->cosOriTuning[0],cell_devCV[1][0][0][0],1.0,-1,streamCV[1]);
	cv::gpu::multiply(cell_devCV[1][0][4][1],f->cosOriTuning[4],cell_devCV[1][0][4][0],1.0,-1,streamCV[1+nEye]);
	cv::gpu::multiply(cell_devCV[1][0][0][1],f->sinOriTuning[0],cell_devCV[1][1][0][0],1.0,-1,streamCV[1+2*nEye]);
	cv::gpu::multiply(cell_devCV[1][0][4][1],f->sinOriTuning[4],cell_devCV[1][1][4][0],1.0,-1,streamCV[1+nEye+2*nEye]);
	for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][1],f->cosOriTuning[resIdx1],cell_devCV[1][0][resIdx1][0],1.0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][1],f->cosOriTuning[resIdx2],cell_devCV[1][0][resIdx2][0],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][1],f->sinOriTuning[resIdx1],cell_devCV[1][1][resIdx1][0],1.0,-1,strCV[1*nCoupleOri+callIdx+2*nEye*nCoupleOri]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][1],f->sinOriTuning[resIdx2],cell_devCV[1][1][resIdx2][0],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*nEye*nCoupleOri]);
	}

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppSetStream(streamCU[1]);
	nppsMulC_32f(cell_devCU[1][0][0][1],f->cosOriTuning[0],cell_devCU[1][0][0][0], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	nppsMulC_32f(cell_devCU[1][0][4][1],f->cosOriTuning[4],cell_devCU[1][0][4][0], imageW*imageH);
	nppSetStream(streamCU[1+2*nEye]);
	nppsMulC_32f(cell_devCU[1][0][0][1],f->sinOriTuning[0],cell_devCU[1][1][0][0], imageW*imageH);
	nppSetStream(streamCU[1+nEye+2*nEye]);
	nppsMulC_32f(cell_devCU[1][0][4][1],f->sinOriTuning[4],cell_devCU[1][1][4][0], imageW*imageH);
	for (int callIdx=0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsMulC_32f(cell_devCU[1][0][resIdx1][1],f->cosOriTuning[resIdx1],cell_devCU[1][0][resIdx1][0], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsMulC_32f(cell_devCU[1][0][resIdx2][1],f->cosOriTuning[resIdx2],cell_devCU[1][0][resIdx2][0], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+2*nEye*nCoupleOri]);
		nppsMulC_32f(cell_devCU[1][0][resIdx1][1],f->sinOriTuning[resIdx1],cell_devCU[1][1][resIdx1][0], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri+2*nEye*nCoupleOri]);
		nppsMulC_32f(cell_devCU[1][0][resIdx2][1],f->sinOriTuning[resIdx2],cell_devCU[1][1][resIdx2][0], imageW*imageH);
	}

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
	
void Population::projectXYCenterOfMassOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::multiply(cell_devCVoneOri[1][0][1],f->cosOriTuning_devCVoneOri,cell_devCVoneOri[1][0][0]);
	cv::gpu::multiply(cell_devCVoneOri[1][0][1],f->sinOriTuning_devCVoneOri,cell_devCVoneOri[1][1][0]);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppsMul_32f(cell_devCUoneOri[1][0][1],f->cosOriTuning_devCUoneOri,cell_devCUoneOri[1][0][0], nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][1],f->sinOriTuning_devCUoneOri,cell_devCUoneOri[1][1][0], nOrient*imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::multiply(cell_devCVoneOri[1][0][1],f->cosOriTuning_devCVoneOri,cell_devCVoneOri[1][0][0],strCV[0]);
	cv::gpu::multiply(cell_devCVoneOri[1][0][1],f->sinOriTuning_devCVoneOri,cell_devCVoneOri[1][1][0],strCV[1]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppSetStream(strCU[0]);
	nppsMul_32f(cell_devCUoneOri[1][0][1],f->cosOriTuning_devCUoneOri,cell_devCUoneOri[1][0][0], nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	nppsMul_32f(cell_devCUoneOri[1][0][1],f->sinOriTuning_devCUoneOri,cell_devCUoneOri[1][1][0], nOrient*imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),f->cosOriTuning_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)));
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),f->sinOriTuning_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)));
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 1*nOrient*imageW*imageH,f->cosOriTuning_devCUrepOriRepPhase,cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 0*nOrient*imageW*imageH, nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 1*nOrient*imageW*imageH,f->sinOriTuning_devCUrepOriRepPhase,cell_devCUrepOriRepPhase[1] + 1*nPhase*nOrient*imageW*imageH + 0*nOrient*imageW*imageH, nOrient*imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),f->cosOriTuning_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),1.0,-1,strCV[0]);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),f->sinOriTuning_devCVrepOriRepPhase,cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),1.0,-1,strCV[1]);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::projectXYCenterOfMassCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	nppSetStream(strCU[0]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 1*nOrient*imageW*imageH,f->cosOriTuning_devCUrepOriRepPhase,cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 0*nOrient*imageW*imageH, nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*nPhase*nOrient*imageW*imageH + 1*nOrient*imageW*imageH,f->sinOriTuning_devCUrepOriRepPhase,cell_devCUrepOriRepPhase[1] + 1*nPhase*nOrient*imageW*imageH + 0*nOrient*imageW*imageH, nOrient*imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Intersection of constraints
/////////////////////////////////////////////////////////////////////////////////////////////

void Population::solveApertureOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent) {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCV[0][0][theta][0]
	b1ZeroDisp -> cell_devCV[1][1][theta][3]
	b1NonZeroDisp -> cell_devCV[1][0][theta][3]
	countNonZeroDisp -> cell_devCV[1][0][0][3]
	
	dx -> cell_devCV[1][0][theta][0]
	sumX -> cell_devCV[1][0][0][0]
	dy -> cell_devCV[1][1][theta][0]
	sumY -> cell_devCV[1][1][0][0]
	dxdy -> cell_devCV[1][0][theta][2]
	sumXY -> cell_devCV[1][0][0][2]
	dxdx-> cell_devCV[1][0][theta][1]
	sumXX -> cell_devCV[1][0][0][1]
	dydy -> cell_devCV[1][1][theta][1]
	sumYY -> cell_devCV[1][1][0][1]

	sumXsumYY -> cell_devCV[0][0][0][0]
	sumYsumXX -> cell_devCV[0][0][1][0]
	sumXXsumYY -> cell_devCV[0][0][2][0]
	sumYsumXY -> cell_devCV[0][0][3][0]
	sumXsumXY -> cell_devCV[0][0][4][0]
	sumXYsumXY -> cell_devCV[0][0][5][0]
	
	dispX -> cell_devCV[0][0][0][0]
	dispY -> cell_devCV[0][0][1][0]
	den -> cell_devCV[0][0][2][0]

	b1ZeroDen -> cell_devCV[0][0][3][0]
	b1NonZeroDen -> cell_devCV[0][0][4][0]
	*/

	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::magnitudeSqr(cell_devCV[1][0][theta][0],cell_devCV[1][1][theta][0],cell_devCV[0][0][theta][0]);

	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::threshold(cell_devCV[0][0][theta][0],cell_devCV[1][0][theta][3],EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[1][0][theta][0],cell_devCV[1][0][theta][3], cell_devCV[1][0][theta][0]);
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[1][1][theta][0],cell_devCV[1][0][theta][3], cell_devCV[1][1][theta][0]);
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[0][0][theta][0],cell_devCV[1][0][theta][3], cell_devCV[0][0][theta][0]);

	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::threshold(cell_devCV[0][0][theta][0],cell_devCV[1][1][theta][3],0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::add(cell_devCV[0][0][theta][0],cell_devCV[1][1][theta][3], cell_devCV[0][0][theta][0]);
	// cv::gpu::multiply(dx,dx,dxdx);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[1][0][theta][0],cell_devCV[1][0][theta][0], cell_devCV[1][0][theta][1]);
	// cv::gpu::multiply(dy,dy,dydy);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[1][1][theta][0],cell_devCV[1][1][theta][0], cell_devCV[1][1][theta][1]);
	// cv::gpu::multiply(dx,dy,dxdy);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::multiply(cell_devCV[1][0][theta][0],cell_devCV[1][1][theta][0], cell_devCV[1][0][theta][2]);

	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::divide(cell_devCV[1][0][theta][1],cell_devCV[0][0][theta][0],cell_devCV[1][0][theta][1]);
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::divide(cell_devCV[1][1][theta][1],cell_devCV[0][0][theta][0],cell_devCV[1][1][theta][1]);
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	for (int theta=0; theta<nOrient; theta++)
		cv::gpu::divide(cell_devCV[1][0][theta][2],cell_devCV[0][0][theta][0],cell_devCV[1][0][theta][2]);    

	for (int theta=1; theta<nOrient; theta++) {
		// cv::gpu::add(sumX,dx(theta),sumX);
		cv::gpu::add(cell_devCV[1][0][0][0],cell_devCV[1][0][theta][0],cell_devCV[1][0][0][0]);
		// cv::gpu::add(sumY,dy(theta),sumY);
		cv::gpu::add(cell_devCV[1][1][0][0],cell_devCV[1][1][theta][0],cell_devCV[1][1][0][0]);
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		cv::gpu::add(cell_devCV[1][0][0][1],cell_devCV[1][0][theta][1],cell_devCV[1][0][0][1]);
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		cv::gpu::add(cell_devCV[1][0][0][2],cell_devCV[1][0][theta][2],cell_devCV[1][0][0][2]);
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		cv::gpu::add(cell_devCV[1][1][0][1],cell_devCV[1][1][theta][1],cell_devCV[1][1][0][1]);
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		cv::gpu::add(cell_devCV[1][0][0][3],cell_devCV[1][0][theta][3],cell_devCV[1][0][0][3]);
	}

	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][0][0][2],cell_devCV[0][0][4][0]);
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][0][0][2],cell_devCV[0][0][3][0]);
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][0][0][1],cell_devCV[0][0][1][0]);
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][1][0][1],cell_devCV[0][0][0][0]);
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCV[1][0][0][2],cell_devCV[1][0][0][2],cell_devCV[0][0][5][0]);
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCV[1][0][0][1], cell_devCV[1][1][0][1],cell_devCV[0][0][2][0]);
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCV[1][0][0][3],cell_devCV[1][0][0][3],nMinOri,1,cv::THRESH_BINARY);

	/*cv::Mat temporanea(imageH,imageW,CV_32FC1);
	cell_devCV[0][0][2][0].download(temporanea);
	std::cout << "sumXXsumYY -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea.at<float>(i,j);*/

	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	cv::gpu::subtract(cell_devCV[0][0][0][0],cell_devCV[0][0][3][0],cell_devCV[0][0][0][0]);
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	cv::gpu::subtract(cell_devCV[0][0][1][0],cell_devCV[0][0][4][0],cell_devCV[0][0][1][0]);
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCV[0][0][2][0],cell_devCV[0][0][5][0],cell_devCV[0][0][2][0]);
	
	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCV[0][0][2][0],cell_devCV[0][0][4][0],EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	cv::gpu::multiply(cell_devCV[0][0][0][0],cell_devCV[0][0][4][0],cell_devCV[0][0][0][0]);
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCV[0][0][1][0],cell_devCV[0][0][4][0],cell_devCV[0][0][1][0]);
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCV[0][0][2][0],cell_devCV[0][0][4][0],cell_devCV[0][0][2][0]);

	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	cv::gpu::multiply(cell_devCV[0][0][0][0],cell_devCV[1][0][0][3],cell_devCV[0][0][0][0]); 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCV[0][0][1][0],cell_devCV[1][0][0][3],cell_devCV[0][0][1][0]);
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCV[0][0][2][0],cell_devCV[0][0][3][0],0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCV[0][0][2][0],cell_devCV[0][0][3][0],cell_devCV[0][0][2][0]);

	// cv::gpu::divide(dispX,den,dispX);
	cv::gpu::divide(cell_devCV[0][0][0][0],cell_devCV[0][0][2][0],cell_devCV[0][0][0][0]);
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCV[0][0][1][0],cell_devCV[0][0][2][0],cell_devCV[0][0][1][0]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCU[0][0][theta][0]
	b1NonZeroDisp -> cell_devCU[1][0][theta][3]
	countNonZeroDisp -> cell_devCU[1][0][0][3]
	
	dx -> cell_devCU[1][0][theta][0]
	sumX -> cell_devCU[1][0][0][0]
	dy -> cell_devCU[1][1][theta][0]
	sumY -> cell_devCU[1][1][0][0]
	dxdy -> cell_devCU[1][0][theta][2]
	sumXY -> cell_devCU[1][0][0][2]
	dxdx-> cell_devCU[1][0][theta][1]
	sumXX -> cell_devCU[1][0][0][1]
	dydy -> cell_devCU[1][1][theta][1]
	sumYY -> cell_devCU[1][1][0][1]

	sumXsumYY -> cell_devCU[0][0][0][0]
	sumYsumXX -> cell_devCU[0][0][1][0]
	sumXXsumYY -> cell_devCU[0][0][2][0]
	sumYsumXY -> cell_devCU[0][0][3][0]
	sumXsumXY -> cell_devCU[0][0][4][0]
	sumXYsumXY -> cell_devCU[0][0][5][0]
	
	dispX -> cell_devCU[0][0][0][0]
	dispY -> cell_devCU[0][0][1][0]
	den -> cell_devCU[0][0][2][0]

	b1NonZeroDen -> cell_devCU[0][0][4][0]
	*/

	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsSqr_32f(cell_devCU[1][0][theta][0],cell_devCU[1][0][theta][1], imageW*imageH);
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsSqr_32f(cell_devCU[1][1][theta][0],cell_devCU[1][1][theta][1], imageW*imageH);
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsAdd_32f(cell_devCU[1][0][theta][1],cell_devCU[1][1][theta][1],cell_devCU[0][0][theta][0], imageW*imageH);

	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	for (int theta=0; theta<nOrient; theta++)
		nppsThreshold_LTVal_32f(cell_devCU[0][0][theta][0],cell_devCU[1][0][theta][3], imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	for (int theta=0; theta<nOrient; theta++)
		nppsThreshold_GTVal_32f_I(cell_devCU[1][0][theta][3], imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsMul_32f_I(cell_devCU[1][0][theta][3],cell_devCU[1][0][theta][0],imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsMul_32f_I(cell_devCU[1][0][theta][3],cell_devCU[1][1][theta][0], imageW*imageH);
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsMul_32f(cell_devCU[1][0][theta][0],cell_devCU[1][0][theta][0],cell_devCU[1][0][theta][1], imageW*imageH);
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsMul_32f(cell_devCU[1][1][theta][0],cell_devCU[1][1][theta][0],cell_devCU[1][1][theta][1], imageW*imageH);
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsMul_32f(cell_devCU[1][0][theta][0],cell_devCU[1][1][theta][0],cell_devCU[1][0][theta][2], imageW*imageH);
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	for (int theta=0; theta<nOrient; theta++)
		nppsThreshold_LTVal_32f_I(cell_devCU[0][0][theta][0], imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsDiv_32f_I(cell_devCU[0][0][theta][0],cell_devCU[1][0][theta][1], imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsDiv_32f_I(cell_devCU[0][0][theta][0],cell_devCU[1][1][theta][1], imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	for (int theta=0; theta<nOrient; theta++)
		nppsDiv_32f_I(cell_devCU[0][0][theta][0],cell_devCU[1][0][theta][2], imageW*imageH);

	for (int theta=1; theta<nOrient; theta++) {
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][0][theta][0],cell_devCU[1][0][0][0], imageW*imageH);
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][1][theta][0],cell_devCU[1][1][0][0], imageW*imageH);
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][0][theta][1],cell_devCU[1][0][0][1], imageW*imageH);
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][0][theta][2],cell_devCU[1][0][0][2], imageW*imageH);
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][1][theta][1],cell_devCU[1][1][0][1], imageW*imageH);
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppsAdd_32f_I(cell_devCU[1][0][theta][3],cell_devCU[1][0][0][3], imageW*imageH);
	}
	
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppsMul_32f(cell_devCU[1][0][0][0], cell_devCU[1][0][0][2], cell_devCU[0][0][4][0], imageW*imageH);
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCU[1][1][0][0], cell_devCU[1][0][0][2], cell_devCU[0][0][3][0], imageW*imageH);
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppsMul_32f(cell_devCU[1][1][0][0], cell_devCU[1][0][0][1], cell_devCU[0][0][1][0], imageW*imageH);
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCU[1][0][0][0], cell_devCU[1][1][0][1], cell_devCU[0][0][0][0], imageW*imageH);
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCU[1][0][0][2], cell_devCU[1][0][0][2], cell_devCU[0][0][5][0], imageW*imageH);
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCU[1][0][0][1], cell_devCU[1][1][0][1], cell_devCU[0][0][2][0], imageW*imageH);
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppsThreshold_LTVal_32f_I(cell_devCU[1][0][0][3], imageW*imageH,nMinOri,0);
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCU[1][0][0][3], imageW*imageH,0,1);
	
	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppsSub_32f_I(cell_devCU[0][0][3][0], cell_devCU[0][0][0][0], imageW*imageH);
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppsSub_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][1][0], imageW*imageH);
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppsSub_32f_I(cell_devCU[0][0][5][0], cell_devCU[0][0][2][0], imageW*imageH);
	
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCU[0][0][2][0], cell_devCU[0][0][4][0], imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCU[0][0][4][0], imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][0][0], imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][1][0], imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCU[1][0][0][3], cell_devCU[0][0][0][0], imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCU[1][0][0][3], cell_devCU[0][0][1][0], imageW*imageH);
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppsThreshold_LTVal_32f_I(cell_devCU[0][0][2][0], imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppsDiv_32f_I(cell_devCU[0][0][2][0], cell_devCU[0][0][0][0], imageW*imageH);
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppsDiv_32f_I(cell_devCU[0][0][2][0], cell_devCU[0][0][1][0], imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	int resIdx1;
	int resIdx2;

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCV[0][0][theta][0]
	b1ZeroDisp -> cell_devCV[1][1][theta][3]
	b1NonZeroDisp -> cell_devCV[1][0][theta][3]
	countNonZeroDisp -> cell_devCV[1][0][0][3]
	
	dx -> cell_devCV[1][0][theta][0]
	sumX -> cell_devCV[1][0][0][0]
	dy -> cell_devCV[1][1][theta][0]
	sumY -> cell_devCV[1][1][0][0]
	dxdy -> cell_devCV[1][0][theta][2]
	sumXY -> cell_devCV[1][0][0][2]
	dxdx-> cell_devCV[1][0][theta][1]
	sumXX -> cell_devCV[1][0][0][1]
	dydy -> cell_devCV[1][1][theta][1]
	sumYY -> cell_devCV[1][1][0][1]

	sumXsumYY -> cell_devCV[0][0][0][0]
	sumYsumXX -> cell_devCV[0][0][1][0]
	sumXXsumYY -> cell_devCV[0][0][2][0]
	sumYsumXY -> cell_devCV[0][0][3][0]
	sumXsumXY -> cell_devCV[0][0][4][0]
	sumXYsumXY -> cell_devCV[0][0][5][0]
	
	dispX -> cell_devCV[0][0][0][0]
	dispY -> cell_devCV[0][0][1][0]
	den -> cell_devCV[0][0][2][0]

	b1ZeroDen -> cell_devCV[0][0][3][0]
	b1NonZeroDen -> cell_devCV[0][0][4][0]
	*/

	// s1, p0
	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	cv::gpu::magnitudeSqr(cell_devCV[1][0][0][0],cell_devCV[1][1][0][0],cell_devCV[0][0][0][0],streamCV[1]);
	cv::gpu::magnitudeSqr(cell_devCV[1][0][4][0],cell_devCV[1][1][4][0],cell_devCV[0][0][4][0],streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::magnitudeSqr(cell_devCV[1][0][resIdx1][0],cell_devCV[1][1][resIdx1][0],cell_devCV[0][0][resIdx1][0],strCV[1*nCoupleOri+callIdx]);
		cv::gpu::magnitudeSqr(cell_devCV[1][0][resIdx2][0],cell_devCV[1][1][resIdx2][0],cell_devCV[0][0][resIdx2][0],strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}

	// s2, p0
	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCV[0][0][0][0],cell_devCV[1][0][0][3],EPS_float,1,cv::THRESH_BINARY,streamCV[1]);
	cv::gpu::threshold(cell_devCV[0][0][4][0],cell_devCV[1][0][4][3],EPS_float,1,cv::THRESH_BINARY,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::threshold(cell_devCV[0][0][resIdx1][0],cell_devCV[1][0][resIdx1][3],EPS_float,1,cv::THRESH_BINARY,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::threshold(cell_devCV[0][0][resIdx2][0],cell_devCV[1][0][resIdx2][3],EPS_float,1,cv::THRESH_BINARY,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}

	// s3, p1 (p0.wait)
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	strCV[1].waitForCompletion();
	strCV[1+nEye].waitForCompletion();
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		strCV[1*nCoupleOri+callIdx].waitForCompletion();
		strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
	}
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][0][0][3], cell_devCV[1][0][0][0],1.0,-1,streamCV[4]);
	cv::gpu::multiply(cell_devCV[1][0][4][0],cell_devCV[1][0][4][3], cell_devCV[1][0][4][0],1.0,-1,streamCV[4+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][0],cell_devCV[1][0][resIdx1][3], cell_devCV[1][0][resIdx1][0],1.0,-1,strCV[4*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][0],cell_devCV[1][0][resIdx2][3], cell_devCV[1][0][resIdx2][0],1.0,-1,strCV[4*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s3, p2
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][0][0][3], cell_devCV[1][1][0][0],1.0,-1,streamCV[5]);
	cv::gpu::multiply(cell_devCV[1][1][4][0],cell_devCV[1][0][4][3], cell_devCV[1][1][4][0],1.0,-1,streamCV[5+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][1][resIdx1][0],cell_devCV[1][0][resIdx1][3], cell_devCV[1][1][resIdx1][0],1.0,-1,strCV[5*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][1][resIdx2][0],cell_devCV[1][0][resIdx2][3], cell_devCV[1][1][resIdx2][0],1.0,-1,strCV[5*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s3, p0 
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	cv::gpu::multiply(cell_devCV[0][0][0][0],cell_devCV[1][0][0][3], cell_devCV[0][0][0][0],1.0,-1,streamCV[1]);
	cv::gpu::multiply(cell_devCV[0][0][4][0],cell_devCV[1][0][4][3], cell_devCV[0][0][4][0],1.0,-1,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[0][0][resIdx1][0],cell_devCV[1][0][resIdx1][3], cell_devCV[0][0][resIdx1][0],1.0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[0][0][resIdx2][0],cell_devCV[1][0][resIdx2][3], cell_devCV[0][0][resIdx2][0],1.0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}

	// s4, p3 (p1.wait)
	// cv::gpu::multiply(dx,dx,dxdx);
	streamCV[4].waitForCompletion();
	streamCV[4].waitForCompletion();
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		strCV[4*nCoupleOri+callIdx].waitForCompletion();
		strCV[4*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
	}
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][0][0][0], cell_devCV[1][0][0][1],1.0,-1,streamCV[8]);
	cv::gpu::multiply(cell_devCV[1][0][4][0],cell_devCV[1][0][4][0], cell_devCV[1][0][4][1],1.0,-1,streamCV[8+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][0],cell_devCV[1][0][resIdx1][0], cell_devCV[1][0][resIdx1][1],1.0,-1,strCV[8*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][0],cell_devCV[1][0][resIdx2][0], cell_devCV[1][0][resIdx2][1],1.0,-1,strCV[8*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s4, p4 (p2.wait)
	// cv::gpu::multiply(dy,dy,dydy);
	streamCV[5].waitForCompletion();
	streamCV[5].waitForCompletion();
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		strCV[5*nCoupleOri+callIdx].waitForCompletion();
		strCV[5*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
	}
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][1][0][0], cell_devCV[1][1][0][1],1.0,-1,streamCV[9]);
	cv::gpu::multiply(cell_devCV[1][1][4][0],cell_devCV[1][1][4][0], cell_devCV[1][1][4][1],1.0,-1,streamCV[9+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][1][resIdx1][0],cell_devCV[1][1][resIdx1][0], cell_devCV[1][1][resIdx1][1],1.0,-1,strCV[9*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][1][resIdx2][0],cell_devCV[1][1][resIdx2][0], cell_devCV[1][1][resIdx2][1],1.0,-1,strCV[9*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s4, p5 
	// cv::gpu::multiply(dx,dy,dxdy);
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][1][0][0], cell_devCV[1][0][0][2],1.0,-1,streamCV[0]);
	cv::gpu::multiply(cell_devCV[1][0][4][0],cell_devCV[1][1][4][0], cell_devCV[1][0][4][2],1.0,-1,streamCV[0+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::multiply(cell_devCV[1][0][resIdx1][0],cell_devCV[1][1][resIdx1][0], cell_devCV[1][0][resIdx1][2],1.0,-1,strCV[0*nCoupleOri+callIdx]);
		cv::gpu::multiply(cell_devCV[1][0][resIdx2][0],cell_devCV[1][1][resIdx2][0], cell_devCV[1][0][resIdx2][2],1.0,-1,strCV[0*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s4, p0
	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCV[0][0][0][0],cell_devCV[1][1][0][3],0,1,cv::THRESH_BINARY_INV,streamCV[1]);
	cv::gpu::threshold(cell_devCV[0][0][4][0],cell_devCV[1][1][4][3],0,1,cv::THRESH_BINARY_INV,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::threshold(cell_devCV[0][0][resIdx1][0],cell_devCV[1][1][resIdx1][3],0,1,cv::THRESH_BINARY_INV,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::threshold(cell_devCV[0][0][resIdx2][0],cell_devCV[1][1][resIdx2][3],0,1,cv::THRESH_BINARY_INV,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s4, p0
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	cv::gpu::add(cell_devCV[0][0][0][0],cell_devCV[1][1][0][3], cell_devCV[0][0][0][0],mask0,-1,streamCV[1]);
	cv::gpu::add(cell_devCV[0][0][4][0],cell_devCV[1][1][4][3], cell_devCV[0][0][4][0],mask0,-1,streamCV[1+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::add(cell_devCV[0][0][resIdx1][0],cell_devCV[1][1][resIdx1][3], cell_devCV[0][0][resIdx1][0],mask0,-1,strCV[1*nCoupleOri+callIdx]);
		cv::gpu::add(cell_devCV[0][0][resIdx2][0],cell_devCV[1][1][resIdx2][3], cell_devCV[0][0][resIdx2][0],mask0,-1,strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}

	// s5, p3 (p0.wait)
	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	strCV[1].waitForCompletion();
	strCV[1+nEye].waitForCompletion();
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		strCV[1*nCoupleOri+callIdx].waitForCompletion();
		strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
	}
	cv::gpu::divide(cell_devCV[1][0][0][1],cell_devCV[0][0][0][0],cell_devCV[1][0][0][1],1.0,-1,streamCV[8]);
	cv::gpu::divide(cell_devCV[1][0][4][1],cell_devCV[0][0][4][0],cell_devCV[1][0][4][1],1.0,-1,streamCV[8+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::divide(cell_devCV[1][0][resIdx1][1],cell_devCV[0][0][resIdx1][0],cell_devCV[1][0][resIdx1][1],1.0,-1,strCV[8*nCoupleOri+callIdx]);
		cv::gpu::divide(cell_devCV[1][0][resIdx2][1],cell_devCV[0][0][resIdx2][0],cell_devCV[1][0][resIdx2][1],1.0,-1,strCV[8*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s5, p4 
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	cv::gpu::divide(cell_devCV[1][1][0][1],cell_devCV[0][0][0][0],cell_devCV[1][1][0][1],1.0,-1,streamCV[9]);
	cv::gpu::divide(cell_devCV[1][1][4][1],cell_devCV[0][0][4][0],cell_devCV[1][1][4][1],1.0,-1,streamCV[9+nEye]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::divide(cell_devCV[1][1][resIdx1][1],cell_devCV[0][0][resIdx1][0],cell_devCV[1][1][resIdx1][1],1.0,-1,strCV[9*nCoupleOri+callIdx]);
		cv::gpu::divide(cell_devCV[1][1][resIdx2][1],cell_devCV[0][0][resIdx2][0],cell_devCV[1][1][resIdx2][1],1.0,-1,strCV[9*nCoupleOri+callIdx+nEye*nCoupleOri]);
	}
	// s5, p5 
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	cv::gpu::divide(cell_devCV[1][0][0][2],cell_devCV[0][0][0][0],cell_devCV[1][0][0][2],1.0,-1,streamCV[0]);  
	cv::gpu::divide(cell_devCV[1][0][4][2],cell_devCV[0][0][4][0],cell_devCV[1][0][4][2],1.0,-1,streamCV[0+nEye]); 
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		cv::gpu::divide(cell_devCV[1][0][resIdx1][2],cell_devCV[0][0][resIdx1][0],cell_devCV[1][0][resIdx1][2],1.0,-1,strCV[0*nCoupleOri+callIdx]);  
		cv::gpu::divide(cell_devCV[1][0][resIdx2][2],cell_devCV[0][0][resIdx2][0],cell_devCV[1][0][resIdx2][2],1.0,-1,strCV[0*nCoupleOri+callIdx+nEye*nCoupleOri]); 
	}

	streamCV[4].waitForCompletion();
	streamCV[4+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][0][0][0],cell_devCV[1][0][4][0],cell_devCV[1][0][0][0],mask0,-1,strCV[4*nCoupleOri]);
	streamCV[5].waitForCompletion();
	streamCV[5+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][1][0][0],cell_devCV[1][1][4][0],cell_devCV[1][1][0][0],mask0,-1,strCV[5*nCoupleOri]);
	streamCV[8].waitForCompletion();
	streamCV[8+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][0][0][1],cell_devCV[1][0][4][1],cell_devCV[1][0][0][1],mask0,-1,strCV[8*nCoupleOri]);
	streamCV[9].waitForCompletion();
	streamCV[9+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][1][0][1],cell_devCV[1][1][4][1],cell_devCV[1][1][0][1],mask0,-1,strCV[9*nCoupleOri]);
	streamCV[0].waitForCompletion();
	streamCV[0+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][0][0][2],cell_devCV[1][0][4][2],cell_devCV[1][0][0][2],mask0,-1,strCV[0*nCoupleOri]);
	streamCV[1].waitForCompletion();
	streamCV[1+nEye].waitForCompletion();
	cv::gpu::add(cell_devCV[1][0][0][3],cell_devCV[1][0][4][3],cell_devCV[1][0][0][3],mask0,-1,strCV[1*nCoupleOri]);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		// s6, p1
		// cv::gpu::add(sumX,dx(theta),sumX);
		strCV[4*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][0],cell_devCV[1][0][resIdx1][0],cell_devCV[1][0][0][0],mask0,-1,strCV[4*nCoupleOri]);
		// s6, p2
		// cv::gpu::add(sumY,dy(theta),sumY);
		strCV[5*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][1][0][0],cell_devCV[1][1][resIdx1][0],cell_devCV[1][1][0][0],mask0,-1,strCV[5*nCoupleOri]);
		// s6, p3
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		strCV[8*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][1],cell_devCV[1][0][resIdx1][1],cell_devCV[1][0][0][1],mask0,-1,strCV[8*nCoupleOri]);
		// s6, p4
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		strCV[9*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][1][0][1],cell_devCV[1][1][resIdx1][1],cell_devCV[1][1][0][1],mask0,-1,strCV[9*nCoupleOri]);
		// s6, p5
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		strCV[0*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][2],cell_devCV[1][0][resIdx1][2],cell_devCV[1][0][0][2],mask0,-1,strCV[0*nCoupleOri]);
		// s6, p0
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		strCV[1*nCoupleOri+callIdx].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][3],cell_devCV[1][0][resIdx1][3],cell_devCV[1][0][0][3],mask0,-1,strCV[1*nCoupleOri]);

		strCV[4*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][0],cell_devCV[1][0][resIdx2][0],cell_devCV[1][0][0][0],mask0,-1,strCV[4*nCoupleOri]);
		strCV[5*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][1][0][0],cell_devCV[1][1][resIdx2][0],cell_devCV[1][1][0][0],mask0,-1,strCV[5*nCoupleOri]);
		strCV[8*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][1],cell_devCV[1][0][resIdx2][1],cell_devCV[1][0][0][1],mask0,-1,strCV[8*nCoupleOri]);
		strCV[9*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][1][0][1],cell_devCV[1][1][resIdx2][1],cell_devCV[1][1][0][1],mask0,-1,strCV[9*nCoupleOri]);
		strCV[0*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][2],cell_devCV[1][0][resIdx2][2],cell_devCV[1][0][0][2],mask0,-1,strCV[0*nCoupleOri]);
		strCV[1*nCoupleOri+callIdx+nEye*nCoupleOri].waitForCompletion();
		cv::gpu::add(cell_devCV[1][0][0][3],cell_devCV[1][0][resIdx2][3],cell_devCV[1][0][0][3],mask0,-1,strCV[1*nCoupleOri]);
	}

	/*cv::Mat temporanea(imageH,imageW,CV_32FC1);
	cell_devCV[1][0][0][1].download(temporanea);
	std::cout << "sumXX -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea.at<float>(i,j);*/

	// s7, p1 (p5.wait)
	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	strCV[0*nCoupleOri].waitForCompletion();
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][0][0][2],cell_devCV[0][0][4][0],1.0,-1,strCV[4*nCoupleOri]);
	// s7, p2
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][0][0][2],cell_devCV[0][0][3][0],1.0,-1,strCV[5*nCoupleOri]);
	// s7, p3 
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCV[1][1][0][0],cell_devCV[1][0][0][1],cell_devCV[0][0][1][0],1.0,-1,strCV[8*nCoupleOri]);
	// s7, p4 
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCV[1][0][0][0],cell_devCV[1][1][0][1],cell_devCV[0][0][0][0],1.0,-1,strCV[9*nCoupleOri]);
	// s7, p5
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCV[1][0][0][2],cell_devCV[1][0][0][2],cell_devCV[0][0][5][0],1.0,-1,strCV[0*nCoupleOri]);
	// s7, p5 
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCV[1][0][0][1], cell_devCV[1][1][0][1],cell_devCV[0][0][2][0],1.0,-1,strCV[0*nCoupleOri]);
	// s7, p0
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCV[1][0][0][3],cell_devCV[1][0][0][3],nMinOri,1,cv::THRESH_BINARY,strCV[1*nCoupleOri]);

	// s8, p5
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCV[0][0][2][0],cell_devCV[0][0][5][0],cell_devCV[0][0][2][0],mask0,-1,strCV[0*nCoupleOri]);
	// s8, p4 (p2.wait)
	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	strCV[5*nCoupleOri].waitForCompletion();
	cv::gpu::subtract(cell_devCV[0][0][0][0],cell_devCV[0][0][3][0],cell_devCV[0][0][0][0],mask0,-1,strCV[9*nCoupleOri]);
	// s8, p3 (p1.wait)
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	strCV[4*nCoupleOri].waitForCompletion();
	cv::gpu::subtract(cell_devCV[0][0][1][0],cell_devCV[0][0][4][0],cell_devCV[0][0][1][0],mask0,-1,strCV[8*nCoupleOri]);
	
	// s9, p5
	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCV[0][0][2][0],cell_devCV[0][0][4][0],EPS_float,1,cv::THRESH_BINARY,strCV[0*nCoupleOri]);
	
	// s10, p4 (p5.wait)
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	strCV[0*nCoupleOri].waitForCompletion();
	cv::gpu::multiply(cell_devCV[0][0][0][0],cell_devCV[0][0][4][0],cell_devCV[0][0][0][0],1.0,-1,strCV[9*nCoupleOri]);
	// s10, p3
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCV[0][0][1][0],cell_devCV[0][0][4][0],cell_devCV[0][0][1][0],1.0,-1,strCV[8*nCoupleOri]);
	// s10, p5
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCV[0][0][2][0],cell_devCV[0][0][4][0],cell_devCV[0][0][2][0],1.0,-1,strCV[0*nCoupleOri]);

	// s11, p4 (p0.wait)
	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	strCV[1*nCoupleOri].waitForCompletion();
	cv::gpu::multiply(cell_devCV[0][0][0][0],cell_devCV[1][0][0][3],cell_devCV[0][0][0][0],1.0,-1,strCV[9*nCoupleOri]); 
	// s11, p3 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCV[0][0][1][0],cell_devCV[1][0][0][3],cell_devCV[0][0][1][0],1.0,-1,strCV[8*nCoupleOri]);
	// s11, p5
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCV[0][0][2][0],cell_devCV[0][0][3][0],0,1,cv::THRESH_BINARY_INV,strCV[0*nCoupleOri]);
	// s11, p5 
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCV[0][0][2][0],cell_devCV[0][0][3][0],cell_devCV[0][0][2][0],mask0,-1,strCV[0*nCoupleOri]);

	// s12, p4 (p5.wait)
	// cv::gpu::divide(dispX,den,dispX);
	strCV[0*nCoupleOri].waitForCompletion();
	cv::gpu::divide(cell_devCV[0][0][0][0],cell_devCV[0][0][2][0],cell_devCV[0][0][0][0],1.0,-1,strCV[9*nCoupleOri]);
	// s12, p3
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCV[0][0][1][0],cell_devCV[0][0][2][0],cell_devCV[0][0][1][0],1.0,-1,strCV[8*nCoupleOri]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	int resIdx1;
	int resIdx2;
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCU[0][0][theta][0]
	b1NonZeroDisp -> cell_devCU[1][0][theta][3]
	countNonZeroDisp -> cell_devCU[1][0][0][3]
	
	dx -> cell_devCU[1][0][theta][0]
	sumX -> cell_devCU[1][0][0][0]
	dy -> cell_devCU[1][1][theta][0]
	sumY -> cell_devCU[1][1][0][0]
	dxdy -> cell_devCU[1][0][theta][2]
	sumXY -> cell_devCU[1][0][0][2]
	dxdx-> cell_devCU[1][0][theta][1]
	sumXX -> cell_devCU[1][0][0][1]
	dydy -> cell_devCU[1][1][theta][1]
	sumYY -> cell_devCU[1][1][0][1]

	sumXsumYY -> cell_devCU[0][0][0][0]
	sumYsumXX -> cell_devCU[0][0][1][0]
	sumXXsumYY -> cell_devCU[0][0][2][0]
	sumYsumXY -> cell_devCU[0][0][3][0]
	sumXsumXY -> cell_devCU[0][0][4][0]
	sumXYsumXY -> cell_devCU[0][0][5][0]
	
	dispX -> cell_devCU[0][0][0][0]
	dispY -> cell_devCU[0][0][1][0]
	den -> cell_devCU[0][0][2][0]

	b1NonZeroDen -> cell_devCU[0][0][4][0]
	*/

	// s1, p1
	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(streamCU[4]);
	nppsSqr_32f(cell_devCU[1][0][0][0],cell_devCU[1][0][0][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[4], streamCU[4]) );
	nppSetStream(streamCU[4+nEye]);
	nppsSqr_32f(cell_devCU[1][0][4][0],cell_devCU[1][0][4][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[4+nEye], streamCU[4+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[4*nCoupleOri+callIdx]);
		nppsSqr_32f(cell_devCU[1][0][resIdx1][0],cell_devCU[1][0][resIdx1][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[4*nCoupleOri+callIdx], strCU[4*nCoupleOri+callIdx]) );
		nppSetStream(strCU[4*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsSqr_32f(cell_devCU[1][0][resIdx2][0],cell_devCU[1][0][resIdx2][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[4*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[4*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s1, p2
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	nppSetStream(streamCU[5]);
	nppsSqr_32f(cell_devCU[1][1][0][0],cell_devCU[1][1][0][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	nppSetStream(streamCU[5+nEye]);
	nppsSqr_32f(cell_devCU[1][1][4][0],cell_devCU[1][1][4][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5+nEye], streamCU[5+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[5*nCoupleOri+callIdx]);
		nppsSqr_32f(cell_devCU[1][1][resIdx1][0],cell_devCU[1][1][resIdx1][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[5*nCoupleOri+callIdx], strCU[5*nCoupleOri+callIdx]) );
		nppSetStream(strCU[5*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsSqr_32f(cell_devCU[1][1][resIdx2][0],cell_devCU[1][1][resIdx2][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[5*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[5*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s1, p0 (p1.wait, p2.wait)
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	nppSetStream(streamCU[1]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[1], kernelEventStreamCU[4],0) );
	checkCudaErrors( cudaStreamWaitEvent(streamCU[1], kernelEventStreamCU[5],0) );
	nppsAdd_32f(cell_devCU[1][0][0][1],cell_devCU[1][1][0][1],cell_devCU[0][0][0][0], imageW*imageH);
	nppSetStream(streamCU[1+nEye]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[1+nEye], kernelEventStreamCU[4+nEye],0) );
	checkCudaErrors( cudaStreamWaitEvent(streamCU[1+nEye], kernelEventStreamCU[5+nEye],0) );
	nppsAdd_32f(cell_devCU[1][0][4][1],cell_devCU[1][1][4][1],cell_devCU[0][0][4][0], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx], kernelEventStrCU[4*nCoupleOri+callIdx],0) );
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx], kernelEventStrCU[5*nCoupleOri+callIdx],0) );
		nppsAdd_32f(cell_devCU[1][0][resIdx1][1],cell_devCU[1][1][resIdx1][1],cell_devCU[0][0][resIdx1][0], imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[4*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[5*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f(cell_devCU[1][0][resIdx2][1],cell_devCU[1][1][resIdx2][1],cell_devCU[0][0][resIdx2][0], imageW*imageH);
		}

	// s2, p0
	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	nppSetStream(streamCU[1]);
	nppsThreshold_LTVal_32f(cell_devCU[0][0][0][0],cell_devCU[1][0][0][3], imageW*imageH,EPS_float,0);
	nppSetStream(streamCU[1+nEye]);
	nppsThreshold_LTVal_32f(cell_devCU[0][0][4][0],cell_devCU[1][0][4][3], imageW*imageH,EPS_float,0);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsThreshold_LTVal_32f(cell_devCU[0][0][resIdx1][0],cell_devCU[1][0][resIdx1][3], imageW*imageH,EPS_float,0);
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsThreshold_LTVal_32f(cell_devCU[0][0][resIdx2][0],cell_devCU[1][0][resIdx2][3], imageW*imageH,EPS_float,0);
		}
	// s2, p0
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	nppSetStream(streamCU[1]);
	nppsThreshold_GTVal_32f_I(cell_devCU[1][0][0][3], imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	nppSetStream(streamCU[1+nEye]);
	nppsThreshold_GTVal_32f_I(cell_devCU[1][0][4][3], imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1+nEye], streamCU[1+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsThreshold_GTVal_32f_I(cell_devCU[1][0][resIdx1][3], imageW*imageH,0,1);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx], strCU[1*nCoupleOri+callIdx]) );
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsThreshold_GTVal_32f_I(cell_devCU[1][0][resIdx2][3], imageW*imageH,0,1);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]) );
	}

	// s3, p1 (p0.wait)
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	nppSetStream(streamCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[4], kernelEventStreamCU[1],0) );
	nppsMul_32f_I(cell_devCU[1][0][0][3],cell_devCU[1][0][0][0],imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[4], streamCU[4]) );
	nppSetStream(streamCU[4+nEye]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[4+nEye], kernelEventStreamCU[1+nEye],0) );
	nppsMul_32f_I(cell_devCU[1][0][4][3],cell_devCU[1][0][4][0],imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[4+nEye], streamCU[4+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[4*nCoupleOri+callIdx]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri+callIdx], kernelEventStrCU[1*nCoupleOri+callIdx],0) );
		nppsMul_32f_I(cell_devCU[1][0][resIdx1][3],cell_devCU[1][0][resIdx1][0],imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[4*nCoupleOri+callIdx], strCU[4*nCoupleOri+callIdx]) );
		nppSetStream(strCU[4*nCoupleOri+callIdx+nEye*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsMul_32f_I(cell_devCU[1][0][resIdx2][3],cell_devCU[1][0][resIdx2][0],imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[4*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[4*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s3, p2
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	nppSetStream(streamCU[5]);
	nppsMul_32f_I(cell_devCU[1][0][0][3],cell_devCU[1][1][0][0], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	nppSetStream(streamCU[5+nEye]);
	nppsMul_32f_I(cell_devCU[1][0][4][3],cell_devCU[1][1][4][0], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5+nEye], streamCU[5+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[5*nCoupleOri+callIdx]);
		nppsMul_32f_I(cell_devCU[1][0][resIdx1][3],cell_devCU[1][1][resIdx1][0], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[5*nCoupleOri+callIdx], strCU[5*nCoupleOri+callIdx]) );
		nppSetStream(strCU[5*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsMul_32f_I(cell_devCU[1][0][resIdx2][3],cell_devCU[1][1][resIdx2][0], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[5*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[5*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s3, p0
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	nppSetStream(streamCU[1]);
	nppsThreshold_LTVal_32f_I(cell_devCU[0][0][0][0], imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	nppSetStream(streamCU[1+nEye]);
	nppsThreshold_LTVal_32f_I(cell_devCU[0][0][4][0], imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1+nEye], streamCU[1+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[1*nCoupleOri+callIdx]);
		nppsThreshold_LTVal_32f_I(cell_devCU[0][0][resIdx1][0], imageW*imageH,EPS_float,1);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx], strCU[1*nCoupleOri+callIdx]) );
		nppSetStream(strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsThreshold_LTVal_32f_I(cell_devCU[0][0][resIdx2][0], imageW*imageH,EPS_float,1);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[1*nCoupleOri+callIdx+nEye*nCoupleOri]) );
	}
	
	// s4, p3 (p1.wait)
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(streamCU[8]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[8], kernelEventStreamCU[4],0) );
	nppsMul_32f(cell_devCU[1][0][0][0],cell_devCU[1][0][0][0],cell_devCU[1][0][0][1], imageW*imageH);
	nppSetStream(streamCU[8+nEye]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[8+nEye], kernelEventStreamCU[4+nEye],0) );
	nppsMul_32f(cell_devCU[1][0][4][0],cell_devCU[1][0][4][0],cell_devCU[1][0][4][1], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[8*nCoupleOri+callIdx]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri+callIdx], kernelEventStrCU[4*nCoupleOri+callIdx],0) );
		nppsMul_32f(cell_devCU[1][0][resIdx1][0],cell_devCU[1][0][resIdx1][0],cell_devCU[1][0][resIdx1][1], imageW*imageH);
		nppSetStream(strCU[8*nCoupleOri+callIdx+nEye*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[4*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsMul_32f(cell_devCU[1][0][resIdx2][0],cell_devCU[1][0][resIdx2][0],cell_devCU[1][0][resIdx2][1], imageW*imageH);
		}
	// s4, p4 (p2.wait)
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	nppSetStream(streamCU[9]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[9], kernelEventStreamCU[5],0) );
	nppsMul_32f(cell_devCU[1][1][0][0],cell_devCU[1][1][0][0],cell_devCU[1][1][0][1], imageW*imageH);
	nppSetStream(streamCU[9+nEye]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[9+nEye], kernelEventStreamCU[5+nEye],0) );
	nppsMul_32f(cell_devCU[1][1][4][0],cell_devCU[1][1][4][0],cell_devCU[1][1][4][1], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[9*nCoupleOri+callIdx]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri+callIdx], kernelEventStrCU[5*nCoupleOri+callIdx],0) );
		nppsMul_32f(cell_devCU[1][1][resIdx1][0],cell_devCU[1][1][resIdx1][0],cell_devCU[1][1][resIdx1][1], imageW*imageH);
		nppSetStream(strCU[9*nCoupleOri+callIdx+nEye*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[5*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsMul_32f(cell_devCU[1][1][resIdx2][0],cell_devCU[1][1][resIdx2][0],cell_devCU[1][1][resIdx2][1], imageW*imageH);
		}
	// s4, p5
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	nppSetStream(streamCU[0]);
	nppsMul_32f(cell_devCU[1][0][0][0],cell_devCU[1][1][0][0],cell_devCU[1][0][0][2], imageW*imageH);
	nppSetStream(streamCU[0+nEye]);
	nppsMul_32f(cell_devCU[1][0][4][0],cell_devCU[1][1][4][0],cell_devCU[1][0][4][2], imageW*imageH);
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[0*nCoupleOri+callIdx]);
		nppsMul_32f(cell_devCU[1][0][resIdx1][0],cell_devCU[1][1][resIdx1][0],cell_devCU[1][0][resIdx1][2], imageW*imageH);
		nppSetStream(strCU[0*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsMul_32f(cell_devCU[1][0][resIdx2][0],cell_devCU[1][1][resIdx2][0],cell_devCU[1][0][resIdx2][2], imageW*imageH);
		}

	// s5, p3 (p0.wait)
	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	nppSetStream(streamCU[8]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[8], kernelEventStreamCU[1],0) );
	nppsDiv_32f_I(cell_devCU[0][0][0][0],cell_devCU[1][0][0][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[8], streamCU[8]) );
	nppSetStream(streamCU[8+nEye]);
	checkCudaErrors( cudaStreamWaitEvent(streamCU[8+nEye], kernelEventStreamCU[1+nEye],0) );
	nppsDiv_32f_I(cell_devCU[0][0][4][0],cell_devCU[1][0][4][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[8+nEye], streamCU[8+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[8*nCoupleOri+callIdx]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri+callIdx], kernelEventStrCU[1*nCoupleOri+callIdx],0) );
		nppsDiv_32f_I(cell_devCU[0][0][resIdx1][0],cell_devCU[1][0][resIdx1][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[8*nCoupleOri+callIdx], strCU[8*nCoupleOri+callIdx]) );
		nppSetStream(strCU[8*nCoupleOri+callIdx+nEye*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri+callIdx+nEye*nCoupleOri], kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsDiv_32f_I(cell_devCU[0][0][resIdx2][0],cell_devCU[1][0][resIdx2][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[8*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[8*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s5, p4
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	nppSetStream(streamCU[9]);
	nppsDiv_32f_I(cell_devCU[0][0][0][0],cell_devCU[1][1][0][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[9], streamCU[9]) );
	nppSetStream(streamCU[9+nEye]);
	nppsDiv_32f_I(cell_devCU[0][0][4][0],cell_devCU[1][1][4][1], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[9+nEye], streamCU[9+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[9*nCoupleOri+callIdx]);
		nppsDiv_32f_I(cell_devCU[0][0][resIdx1][0],cell_devCU[1][1][resIdx1][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[9*nCoupleOri+callIdx], strCU[9*nCoupleOri+callIdx]) );
		nppSetStream(strCU[9*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsDiv_32f_I(cell_devCU[0][0][resIdx2][0],cell_devCU[1][1][resIdx2][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[9*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[9*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}
	// s5, p5
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	nppSetStream(streamCU[0]);
	nppsDiv_32f_I(cell_devCU[0][0][0][0],cell_devCU[1][0][0][2], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );
	nppSetStream(streamCU[0+nEye]);
	nppsDiv_32f_I(cell_devCU[0][0][4][0],cell_devCU[1][0][4][2], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0+nEye], streamCU[0+nEye]) );
	for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		nppSetStream(strCU[0*nCoupleOri+callIdx]);
		nppsDiv_32f_I(cell_devCU[0][0][resIdx1][0],cell_devCU[1][0][resIdx1][2], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[0*nCoupleOri+callIdx], strCU[0*nCoupleOri+callIdx]) );
		nppSetStream(strCU[0*nCoupleOri+callIdx+nEye*nCoupleOri]);
		nppsDiv_32f_I(cell_devCU[0][0][resIdx2][0],cell_devCU[1][0][resIdx2][2], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[0*nCoupleOri+callIdx+nEye*nCoupleOri], strCU[0*nCoupleOri+callIdx+nEye*nCoupleOri]) );
		}

	nppSetStream(strCU[4*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri], kernelEventStreamCU[4],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri], kernelEventStreamCU[4+nEye],0) );
	nppsAdd_32f(cell_devCU[1][0][0][0],cell_devCU[1][0][4][0],cell_devCU[1][0][0][0], imageW*imageH);
	nppSetStream(strCU[5*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[5*nCoupleOri], kernelEventStreamCU[5],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[5*nCoupleOri], kernelEventStreamCU[5+nEye],0) );
	nppsAdd_32f(cell_devCU[1][1][0][0],cell_devCU[1][1][4][0],cell_devCU[1][1][0][0], imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri], kernelEventStreamCU[8],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri], kernelEventStreamCU[8+nEye],0) );
	nppsAdd_32f(cell_devCU[1][0][0][1],cell_devCU[1][0][4][1],cell_devCU[1][0][0][1], imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStreamCU[9],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStreamCU[9+nEye],0) );
	nppsAdd_32f(cell_devCU[1][1][0][1],cell_devCU[1][1][4][1],cell_devCU[1][1][0][1], imageW*imageH);
	nppSetStream(strCU[0*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[0*nCoupleOri], kernelEventStreamCU[0],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[0*nCoupleOri], kernelEventStreamCU[0+nEye],0) );
	nppsAdd_32f(cell_devCU[1][0][0][2],cell_devCU[1][0][4][2],cell_devCU[1][0][0][2], imageW*imageH);
	nppSetStream(strCU[1*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri], kernelEventStreamCU[1],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri], kernelEventStreamCU[1+nEye],0) );
	nppsAdd_32f(cell_devCU[1][0][0][3],cell_devCU[1][0][4][3],cell_devCU[1][0][0][3], imageW*imageH);
		for (int callIdx = 0; callIdx<nCoupleOri; callIdx++) {
		resIdx1 = callIdx+1;
		resIdx2 = nOrient-1-callIdx;
		// s6, p1
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppSetStream(strCU[4*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri], kernelEventStrCU[4*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx1][0],cell_devCU[1][0][0][0], imageW*imageH);
		// s6, p2
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppSetStream(strCU[5*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[5*nCoupleOri], kernelEventStrCU[5*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][1][resIdx1][0],cell_devCU[1][1][0][0], imageW*imageH);
		// s6, p3
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppSetStream(strCU[8*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri], kernelEventStrCU[8*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx1][1],cell_devCU[1][0][0][1], imageW*imageH);
		// s6, p4
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppSetStream(strCU[9*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[9*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][1][resIdx1][1],cell_devCU[1][1][0][1], imageW*imageH);
		// s6, p5
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppSetStream(strCU[0*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[0*nCoupleOri], kernelEventStrCU[0*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx1][2],cell_devCU[1][0][0][2], imageW*imageH);
		// s6, p0
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppSetStream(strCU[1*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri], kernelEventStrCU[1*nCoupleOri+callIdx],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx1][3],cell_devCU[1][0][0][3], imageW*imageH);

		nppSetStream(strCU[4*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri], kernelEventStrCU[4*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx2][0],cell_devCU[1][0][0][0], imageW*imageH);
		nppSetStream(strCU[5*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[5*nCoupleOri], kernelEventStrCU[5*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][1][resIdx2][0],cell_devCU[1][1][0][0], imageW*imageH);
		nppSetStream(strCU[8*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri], kernelEventStrCU[8*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx2][1],cell_devCU[1][0][0][1], imageW*imageH);
		nppSetStream(strCU[9*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[9*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx2][2],cell_devCU[1][0][0][2], imageW*imageH);
		nppSetStream(strCU[0*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[0*nCoupleOri], kernelEventStrCU[0*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][1][resIdx2][1],cell_devCU[1][1][0][1], imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStrCU[0*nCoupleOri], strCU[0*nCoupleOri]) );
		nppSetStream(strCU[1*nCoupleOri]);
		checkCudaErrors( cudaStreamWaitEvent(strCU[1*nCoupleOri], kernelEventStrCU[1*nCoupleOri+callIdx+nEye*nCoupleOri],0) );
		nppsAdd_32f_I(cell_devCU[1][0][resIdx2][3],cell_devCU[1][0][0][3], imageW*imageH);
	}
	
	// s7, p0
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppSetStream(strCU[1*nCoupleOri]);
	nppsThreshold_LTVal_32f_I(cell_devCU[1][0][0][3], imageW*imageH,nMinOri,0);
	// s7, p1 (p5.wait)
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppSetStream(strCU[4*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4*nCoupleOri], kernelEventStrCU[0*nCoupleOri],0) );
	nppsMul_32f(cell_devCU[1][0][0][0], cell_devCU[1][0][0][2], cell_devCU[0][0][4][0], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[4*nCoupleOri], strCU[4*nCoupleOri]) );
	// s7, p2
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppSetStream(strCU[5*nCoupleOri]);
	nppsMul_32f(cell_devCU[1][1][0][0], cell_devCU[1][0][0][2], cell_devCU[0][0][3][0], imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[5*nCoupleOri], strCU[5*nCoupleOri]) );
	// s7, p3
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	nppsMul_32f(cell_devCU[1][1][0][0], cell_devCU[1][0][0][1], cell_devCU[0][0][1][0], imageW*imageH);
	// s7, p4
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	nppsMul_32f(cell_devCU[1][0][0][0], cell_devCU[1][1][0][1], cell_devCU[0][0][0][0], imageW*imageH);
	// s7, p5
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsMul_32f(cell_devCU[1][0][0][2], cell_devCU[1][0][0][2], cell_devCU[0][0][5][0], imageW*imageH);
	// s7, p5
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsMul_32f(cell_devCU[1][0][0][1], cell_devCU[1][1][0][1], cell_devCU[0][0][2][0], imageW*imageH);
	// s7, p0
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppSetStream(strCU[1*nCoupleOri]);
	nppsThreshold_GTVal_32f_I(cell_devCU[1][0][0][3], imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[1*nCoupleOri], strCU[1*nCoupleOri]) );

	// s8, p5
	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsSub_32f_I(cell_devCU[0][0][3][0], cell_devCU[0][0][0][0], imageW*imageH);
	// s8, p4 (p2.wait)
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[5*nCoupleOri],0) );
	nppsSub_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][1][0], imageW*imageH);
	// s8, p3 (p1.wait)
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[8*nCoupleOri], kernelEventStrCU[4*nCoupleOri],0) );
	nppsSub_32f_I(cell_devCU[0][0][5][0], cell_devCU[0][0][2][0], imageW*imageH);

	// s9, p5
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsThreshold_LTVal_32f(cell_devCU[0][0][2][0], cell_devCU[0][0][4][0], imageW*imageH,EPS_float,0);
	// s9, p5
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsThreshold_GTVal_32f_I(cell_devCU[0][0][4][0], imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[0*nCoupleOri], strCU[0*nCoupleOri]) );

	// s10, p4 (p5.wait)
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[0*nCoupleOri],0) );
	nppsMul_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][0][0], imageW*imageH);
	// s10, p3
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	nppsMul_32f_I(cell_devCU[0][0][4][0], cell_devCU[0][0][1][0], imageW*imageH);
	// s10, p5
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppSetStream(strCU[0*nCoupleOri]);
	nppsThreshold_LTVal_32f_I(cell_devCU[0][0][2][0], imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStrCU[0*nCoupleOri], strCU[0*nCoupleOri]) );

	// s11, p4 (p0.wait)
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[1*nCoupleOri],0) );
	nppsMul_32f_I(cell_devCU[1][0][0][3], cell_devCU[0][0][0][0], imageW*imageH);
	// s11, p3
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	nppsMul_32f_I(cell_devCU[1][0][0][3], cell_devCU[0][0][1][0], imageW*imageH);
	
	// s12, p4 (p5.wait)
	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppSetStream(strCU[9*nCoupleOri]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[9*nCoupleOri], kernelEventStrCU[0*nCoupleOri],0) );
	nppsDiv_32f_I(cell_devCU[0][0][2][0], cell_devCU[0][0][0][0], imageW*imageH);
	// s12, p3
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppSetStream(strCU[8*nCoupleOri]);
	nppsDiv_32f_I(cell_devCU[0][0][2][0], cell_devCU[0][0][1][0], imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
	
void Population::solveApertureOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCVoneOri[0][0][0]
	b1ZeroDisp -> cell_devCVoneOri[1][1][3]
	b1NonZeroDisp -> cell_devCVoneOri[1][0][3]
	countNonZeroDisp -> cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH))
	
	dx -> cell_devCVoneOri[1][0][0]
	sumX -> cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH))
	dy -> cell_devCVoneOri[1][1][0]
	sumY -> cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH))
	dxdy -> cell_devCVoneOri[1][0][2]
	sumXY -> cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH))
	dxdx-> cell_devCVoneOri[1][0][1]
	sumXX -> cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH))
	dydy -> cell_devCVoneOri[1][1][1]
	sumYY -> cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH))

	sumXsumYY -> cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH))
	sumYsumXX -> cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH))
	sumXXsumYY -> cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH))
	sumYsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH))
	sumXsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH))
	sumXYsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH))
	
	dispX -> cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH))
	dispY -> cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH))
	den -> cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH))

	b1ZeroDen -> cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH))
	b1NonZeroDen -> cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH))
	*/

	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	cv::gpu::magnitudeSqr(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][1][0],cell_devCVoneOri[0][0][0]);

	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0], cell_devCVoneOri[1][0][3],EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[1][0][0]);
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[1][1][0]);
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[0][0][0]);

	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][1][3],0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	cv::gpu::add(cell_devCVoneOri[0][0][0], cell_devCVoneOri[1][1][3], cell_devCVoneOri[0][0][0]);
	// cv::gpu::multiply(dx,dx,dxdx);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][0], cell_devCVoneOri[1][0][1]);
	// cv::gpu::multiply(dy,dy,dydy);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0],cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][1][1]);
	// cv::gpu::multiply(dx,dy,dxdy);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][0][2]);

	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	cv::gpu::divide(cell_devCVoneOri[1][0][1],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][0][1]);
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	cv::gpu::divide(cell_devCVoneOri[1][1][1],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][1][1]);
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	cv::gpu::divide(cell_devCVoneOri[1][0][2],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][0][2]);    

	for (int theta=1; theta<nOrient; theta++) {
		// cv::gpu::add(sumX,dx(theta),sumX);
		cv::gpu::add(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][0](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(sumY,dy(theta),sumY);
		cv::gpu::add(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][0](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		cv::gpu::add(cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		cv::gpu::add(cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		cv::gpu::add(cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		cv::gpu::add(cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)));
	}

	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)), cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)));
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),nMinOri,1,cv::THRESH_BINARY);

	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)));
	
	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)));

	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH))); 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)));

	// cv::gpu::divide(dispX,den,dispX);
	cv::gpu::divide(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)));

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCUoneOri[0][0][0]
	b1NonZeroDisp -> cell_devCUoneOri[1][0][3] 
	countNonZeroDisp -> cell_devCUoneOri[1][0][3] + 0*(imageW*imageH)
	
	dx -> cell_devCUoneOri[1][0][0]
	sumX -> cell_devCUoneOri[1][0][0] + 0*(imageW*imageH)
	dy -> cell_devCUoneOri[1][1][0]
	sumY -> cell_devCUoneOri[1][1][0] + 0*(imageW*imageH)
	dxdy -> cell_devCUoneOri[1][0][2]
	sumXY -> cell_devCUoneOri[1][0][2] + 0*(imageW*imageH)
	dxdx-> cell_devCUoneOri[1][0][1]
	sumXX -> cell_devCUoneOri[1][0][1] + 0*(imageW*imageH)
	dydy -> cell_devCUoneOri[1][1][1]
	sumYY -> cell_devCUoneOri[1][1][1] + 0*(imageW*imageH)

	sumXsumYY -> cell_devCUoneOri[0][0][0] + 0*(imageW*imageH) 
	sumYsumXX -> cell_devCUoneOri[0][0][0] + 1*(imageW*imageH)
	sumXXsumYY -> cell_devCUoneOri[0][0][0] + 2*(imageW*imageH) 
	sumYsumXY -> cell_devCUoneOri[0][0][0] + 3*(imageW*imageH)
	sumXsumXY -> cell_devCUoneOri[0][0][0] + 4*(imageW*imageH)
	sumXYsumXY -> cell_devCUoneOri[0][0][0] + 5*(imageW*imageH)
	
	dispX -> cell_devCUoneOri[0][0][0] + 0*(imageW*imageH) 
	dispY -> cell_devCUoneOri[0][0][0] + 1*(imageW*imageH)
	den -> cell_devCUoneOri[0][0][0] + 2*(imageW*imageH)

	b1NonZeroDen -> cell_devCUoneOri[0][0][0] + 4*(imageW*imageH)
	*/

	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	nppsSqr_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	nppsSqr_32f(cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	nppsAdd_32f(cell_devCUoneOri[1][0][1],cell_devCUoneOri[1][1][1],cell_devCUoneOri[0][0][0],nOrient*imageW*imageH);

	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][3],nOrient*imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[1][0][3],nOrient*imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3],cell_devCUoneOri[1][0][0],nOrient*imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3],cell_devCUoneOri[1][1][0],nOrient*imageW*imageH);
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][0][2],nOrient*imageW*imageH);
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[0][0][0],nOrient*imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][2],nOrient*imageW*imageH);

	for (int theta=1; theta<nOrient; theta++) {
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][0] + theta*(imageW*imageH),cell_devCUoneOri[1][0][0] + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][1][0] + theta*(imageW*imageH),cell_devCUoneOri[1][1][0] + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][1] + theta*(imageW*imageH),cell_devCUoneOri[1][0][1] + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][2] + theta*(imageW*imageH),cell_devCUoneOri[1][0][2] + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][1][1] + theta*(imageW*imageH),cell_devCUoneOri[1][1][1] + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][3] + theta*(imageW*imageH),cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH);
	}
	
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 4*(imageW*imageH),imageW*imageH);
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][1][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 3*(imageW*imageH), imageW*imageH);
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][1][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][0] + 0*(imageW*imageH), cell_devCUoneOri[1][1][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 5*(imageW*imageH), imageW*imageH);
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCUoneOri[1][0][1] + 0*(imageW*imageH), cell_devCUoneOri[1][1][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), imageW*imageH);
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH,nMinOri,0);
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH,0,1);
	
	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 3*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 5*(imageW*imageH), cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), imageW*imageH);
	
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH),imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH),imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCVoneOri[0][0][0]
	b1ZeroDisp -> cell_devCVoneOri[1][1][3]
	b1NonZeroDisp -> cell_devCVoneOri[1][0][3]
	countNonZeroDisp -> cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH))
	
	dx -> cell_devCVoneOri[1][0][0]
	sumX -> cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH))
	dy -> cell_devCVoneOri[1][1][0]
	sumY -> cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH))
	dxdy -> cell_devCVoneOri[1][0][2]
	sumXY -> cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH))
	dxdx-> cell_devCVoneOri[1][0][1]
	sumXX -> cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH))
	dydy -> cell_devCVoneOri[1][1][1]
	sumYY -> cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH))

	sumXsumYY -> cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH))
	sumYsumXX -> cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH))
	sumXXsumYY -> cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH))
	sumYsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH))
	sumXsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH))
	sumXYsumXY -> cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH))
	
	dispX -> cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH))
	dispY -> cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH))
	den -> cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH))

	b1ZeroDen -> cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH))
	b1NonZeroDen -> cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH))
	*/

	// s1, p0
	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	cv::gpu::magnitudeSqr(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][1][0],cell_devCVoneOri[0][0][0],strCV[0]);

	// s2, p0
	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0], cell_devCVoneOri[1][0][3],EPS_float,1,cv::THRESH_BINARY,strCV[0]);
	
	// s3, p1 (p0.wait)
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	strCV[0].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[1][0][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[1][0][0],1.0,-1,strCV[1]);
	// s3, p2
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[1][1][0],1.0,-1,strCV[2]);
	// s3, p0 
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0], cell_devCVoneOri[1][0][3], cell_devCVoneOri[0][0][0],1.0,-1,strCV[0]);

	// s4, p3 (p1.wait)
	// cv::gpu::multiply(dx,dx,dxdx);
	strCV[1].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][0][0], cell_devCVoneOri[1][0][1],1.0,-1,strCV[3]);
	// s4, p4 (p2.wait)
	// cv::gpu::multiply(dy,dy,dydy);
	strCV[2].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[1][1][0],cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][1][1],1.0,-1,strCV[4]);
	// s4, p5 
	// cv::gpu::multiply(dx,dy,dxdy);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0],cell_devCVoneOri[1][1][0], cell_devCVoneOri[1][0][2],1.0,-1,strCV[5]);
	// s4, p0
	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][1][3],0,1,cv::THRESH_BINARY_INV,strCV[0]);
	// s4, p0
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	cv::gpu::add(cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][1][3], cell_devCVoneOri[0][0][0],mask0,-1,strCV[0]);

	// s5, p3 (p0.wait)
	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	strCV[0].waitForCompletion();
	cv::gpu::divide(cell_devCVoneOri[1][0][1],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][0][1],1.0,-1,strCV[3]);
	// s5, p4 
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	cv::gpu::divide(cell_devCVoneOri[1][1][1],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][1][1],1.0,-1,strCV[4]);
	// s5, p5 
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	cv::gpu::divide(cell_devCVoneOri[1][0][2],cell_devCVoneOri[0][0][0],cell_devCVoneOri[1][0][2],1.0,-1,strCV[5]);    

	for (int theta=1; theta<nOrient; theta++) {
		// s6, p1
		// cv::gpu::add(sumX,dx(theta),sumX);
		cv::gpu::add(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][0](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[1]);
		// s6, p2
		// cv::gpu::add(sumY,dy(theta),sumY);
		cv::gpu::add(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][0](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[2]);
		// s6, p3
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		cv::gpu::add(cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[3]);
		// s6, p4
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		cv::gpu::add(cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[4]);
		// s6, p5
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		cv::gpu::add(cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[5]);
		// s6, p0
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		cv::gpu::add(cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[0]);
	}

	// s7, p1 (p5.wait)
	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	strCV[5].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),1.0,-1,strCV[1]);
	// s7, p2
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),1.0,-1,strCV[2]);
	// s7, p3 
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCVoneOri[1][1][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s7, p4 
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s7, p5
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][2](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);
	// s7, p5 
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCVoneOri[1][0][1](cv::Rect(0,0,imageW,imageH)), cell_devCVoneOri[1][1][1](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);
	// s7, p0
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),nMinOri,1,cv::THRESH_BINARY,strCV[0]);

	// s8, p5
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(5*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),mask0,-1,strCV[5]);
	// s8, p4 (p2.wait)
	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	strCV[2].waitForCompletion();
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[4]);
	// s8, p3 (p1.wait)
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	strCV[1].waitForCompletion();
	cv::gpu::subtract(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),mask0,-1,strCV[3]);
	
	/*cv::Mat temporanea(imageH,imageW,CV_32FC1);
	cell_devCVoneOri[0](cv::Rect(2*imageW,0,imageW,imageH)).download(temporanea);
	std::cout << "den -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea.at<float>(i,j);*/

	// s9, p5
	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),EPS_float,1,cv::THRESH_BINARY,strCV[5]);
	
	// s10, p4 (p5.wait)
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	strCV[5].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s10, p3
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s10, p5
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);

	// s11, p4 (p0.wait)
	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	strCV[0].waitForCompletion();
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]); 
	// s11, p3 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[1][0][3](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s11, p5
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),0,1,cv::THRESH_BINARY_INV,strCV[5]);
	// s11, p5 
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),mask0,-1,strCV[5]);

	// s12, p4 (p5.wait)
	// cv::gpu::divide(dispX,den,dispX);
	strCV[5].waitForCompletion();
	cv::gpu::divide(cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s12, p3
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVoneOri[0][0][0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCUoneOri[0][0][0]
	b1NonZeroDisp -> cell_devCUoneOri[1][0][3] 
	countNonZeroDisp -> cell_devCUoneOri[1][0][3] + 0*(imageW*imageH)
	
	dx -> cell_devCUoneOri[1][0][0]
	sumX -> cell_devCUoneOri[1][0][0] + 0*(imageW*imageH)
	dy -> cell_devCUoneOri[1][1][0]
	sumY -> cell_devCUoneOri[1][1][0] + 0*(imageW*imageH)
	dxdy -> cell_devCUoneOri[1][0][2]
	sumXY -> cell_devCUoneOri[1][0][2] + 0*(imageW*imageH)
	dxdx-> cell_devCUoneOri[1][0][1]
	sumXX -> cell_devCUoneOri[1][0][1] + 0*(imageW*imageH)
	dydy -> cell_devCUoneOri[1][1][1]
	sumYY -> cell_devCUoneOri[1][1][1] + 0*(imageW*imageH)

	sumXsumYY -> cell_devCUoneOri[0][0][0] + 0*(imageW*imageH) 
	sumYsumXX -> cell_devCUoneOri[0][0][0] + 1*(imageW*imageH)
	sumXXsumYY -> cell_devCUoneOri[0][0][0] + 2*(imageW*imageH) 
	sumYsumXY -> cell_devCUoneOri[0][0][0] + 3*(imageW*imageH)
	sumXsumXY -> cell_devCUoneOri[0][0][0] + 4*(imageW*imageH)
	sumXYsumXY -> cell_devCUoneOri[0][0][0] + 5*(imageW*imageH)
	
	dispX -> cell_devCUoneOri[0][0][0] + 0*(imageW*imageH) 
	dispY -> cell_devCUoneOri[0][0][0] + 1*(imageW*imageH)
	den -> cell_devCUoneOri[0][0][0] + 2*(imageW*imageH)

	b1NonZeroDen -> cell_devCUoneOri[0][0][0] + 4*(imageW*imageH)
	*/

	// s1, p1
	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	nppsSqr_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s1, p2
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[2]);
	nppsSqr_32f(cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s1, p0 (p1.wait, p2.wait)
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	nppSetStream(strCU[0]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[1],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[2],0) );
	nppsAdd_32f(cell_devCUoneOri[1][0][1],cell_devCUoneOri[1][1][1],cell_devCUoneOri[0][0][0],nOrient*imageW*imageH);

	// s2, p0
	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][3],nOrient*imageW*imageH,EPS_float,0);
	// s2, p0
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[1][0][3],nOrient*imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );
	
	// s3, p1 (p0.wait)
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[1], kernelEventStrCU[0],0) );
	nppsMul_32f_I(cell_devCUoneOri[1][0][3],cell_devCUoneOri[1][0][0],nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s3, p2
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	nppSetStream(strCU[2]);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3],cell_devCUoneOri[1][1][0],nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s3, p0
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	nppSetStream(strCU[0]);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[0][0][0],nOrient*imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );

	// s4, p3 (p1.wait)
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[1],0) );
	nppsMul_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	// s4, p4 (p2.wait)
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[2],0) );
	nppsMul_32f(cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	// s4, p5
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUoneOri[1][0][0],cell_devCUoneOri[1][1][0],cell_devCUoneOri[1][0][2],nOrient*imageW*imageH);

	// s5, p3 (p0.wait)
	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[0],0) );
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][1],nOrient*imageW*imageH);
	// s5, p4
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[4]);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][1][1],nOrient*imageW*imageH);
	// s5, p5
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	nppSetStream(strCU[5]);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0],cell_devCUoneOri[1][0][2],nOrient*imageW*imageH);

	for (int theta=1; theta<nOrient; theta++) {
		// s6, p1
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppSetStream(strCU[1]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][0] + theta*(imageW*imageH),cell_devCUoneOri[1][0][0] + 0*(imageW*imageH),imageW*imageH);
		// s6, p2
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppSetStream(strCU[2]);
		nppsAdd_32f_I(cell_devCUoneOri[1][1][0] + theta*(imageW*imageH),cell_devCUoneOri[1][1][0] + 0*(imageW*imageH),imageW*imageH);
		// s6, p3
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppSetStream(strCU[3]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][1] + theta*(imageW*imageH),cell_devCUoneOri[1][0][1] + 0*(imageW*imageH),imageW*imageH);
		// s6, p4
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppSetStream(strCU[4]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][2] + theta*(imageW*imageH),cell_devCUoneOri[1][0][2] + 0*(imageW*imageH),imageW*imageH);
		// s6, p5
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppSetStream(strCU[5]);
		nppsAdd_32f_I(cell_devCUoneOri[1][1][1] + theta*(imageW*imageH),cell_devCUoneOri[1][1][1] + 0*(imageW*imageH),imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
		// s6, p0
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppSetStream(strCU[0]);
		nppsAdd_32f_I(cell_devCUoneOri[1][0][3]  + theta*(imageW*imageH),cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH);
	}
	
	// s7, p0
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppSetStream(strCU[0]);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH,nMinOri,0);
	// s7, p1 (p5.wait)
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppSetStream(strCU[1]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[1], kernelEventStrCU[5],0) );
	nppsMul_32f(cell_devCUoneOri[1][0][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 4*(imageW*imageH),imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s7, p2
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppSetStream(strCU[2]);
	nppsMul_32f(cell_devCUoneOri[1][1][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 3*(imageW*imageH), imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s7, p3
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f(cell_devCUoneOri[1][1][0] + 0*(imageW*imageH), cell_devCUoneOri[1][0][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	// s7, p4
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppSetStream(strCU[4]);
	nppsMul_32f(cell_devCUoneOri[1][0][0] + 0*(imageW*imageH), cell_devCUoneOri[1][1][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	// s7, p5
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[1][0][2] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 5*(imageW*imageH), imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	// s7, p5
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUoneOri[1][0][1] + 0*(imageW*imageH), cell_devCUoneOri[1][1][1] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), imageW*imageH);
	// s7, p0
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppSetStream(strCU[0]);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH),imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );

	// s8, p4 (p2.wait)
	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[2],0) );
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 3*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	// s8, p3 (p1.wait)
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[1],0) );
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	// s8, p5
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsSub_32f_I(cell_devCUoneOri[0][0][0] + 5*(imageW*imageH), cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), imageW*imageH);

	// s9, p5
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppSetStream(strCU[5]);
	nppsThreshold_LTVal_32f(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), imageW*imageH,EPS_float,0);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	// s9, p5
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), imageW*imageH,0,1);
	
	// s10, p4 (p5.wait)
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[5],0) );
	nppsMul_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH),imageW*imageH);
	// s10, p3
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f_I(cell_devCUoneOri[0][0][0] + 4*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	// s10, p5
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppSetStream(strCU[5]);
	nppsThreshold_LTVal_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH),imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	
	// s11, p4 (p0.wait)
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[0],0) );
	nppsMul_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	// s11, p3
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f_I(cell_devCUoneOri[1][0][3] + 0*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);
	
	// s12, p4 (p5.wait)
	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[5],0) );
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 0*(imageW*imageH), imageW*imageH);
	// s12, p3
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsDiv_32f_I(cell_devCUoneOri[0][0][0] + 2*(imageW*imageH), cell_devCUoneOri[0][0][0] + 1*(imageW*imageH), imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

void Population::solveApertureOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH))
	b1ZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH))
	b1NonZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH))
	countNonZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH))
	
	dx -> cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH))
	sumX -> cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH))
	dy -> cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH))
	sumY -> cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH))
	dxdy -> cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH))
	sumXY -> cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH))
	dxdx-> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH))
	sumXX -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH))
	dydy -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH))
	sumYY -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH))

	sumXsumYY -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH))
	sumYsumXX -> cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH))
	sumXXsumYY -> cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH))
	sumYsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH))
	sumXsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH))
	sumXYsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH))
	
	dispX -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH))
	dispY -> cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH))
	den -> cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH))

	b1ZeroDen -> cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH))
	b1NonZeroDen -> cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH))
	*/

	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	cv::gpu::magnitudeSqr(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)));

	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)));
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)));
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)));

	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH)),0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	cv::gpu::add(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)));
	// cv::gpu::multiply(dx,dx,dxdx);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)));
	// cv::gpu::multiply(dy,dy,dydy);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)));
	// cv::gpu::multiply(dx,dy,dxdy);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)));

	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)));
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)));
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)));    

	for (int theta=1; theta<nOrient; theta++) {
		// cv::gpu::add(sumX,dx(theta),sumX);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)));
		// cv::gpu::add(sumY,dy(theta),sumY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(theta*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)));
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)));
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)));
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW+theta*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)));
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)));
	}

	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH)));
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)));
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),nMinOri,1,cv::THRESH_BINARY);

	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)));
	
	/*cv::Mat temporanea(imageH,imageW,CV_32FC1);
	cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)).download(temporanea);
	std::cout << "den -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea.at<float>(i,j);*/

	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)),EPS_float,1,cv::THRESH_BINARY);
	
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)));

	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH))); 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)));
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),0,1,cv::THRESH_BINARY_INV);
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)));

	// cv::gpu::divide(dispX,den,dispX);
	cv::gpu::divide(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)));
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)));

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	b1NonZeroDisp -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) 
	countNonZeroDisp -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	
	dx -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	sumX -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dy -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	sumY -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dxdy -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient)
	sumXY -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dxdx-> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient)
	sumXX -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dydy -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient)
	sumYY -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH)

	sumXsumYY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) 
	sumYsumXX -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH)
	sumXXsumYY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH) 
	sumYsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH)
	sumXsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH)
	sumXYsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH)
	
	dispX -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) 
	dispY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH)
	den -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH)

	b1NonZeroDen -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH)
	*/

	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	nppsSqr_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	nppsSqr_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	nppsAdd_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	
	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,nOrient*imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,nOrient*imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient),nOrient*imageW*imageH);

	for (int theta=1; theta<nOrient; theta++) {
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
	}
	
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH);
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH),imageW*imageH);
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH),imageW*imageH);
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH) ,imageW*imageH);
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH,nMinOri,0);
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH,0,1);
	
	/*float *temporanea = (float*) malloc(imageW*imageH*sizeof(float));
	checkCudaErrors( cudaMemcpy(temporanea,cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH) ,imageW*imageH*sizeof(float),cudaMemcpyDeviceToHost) );
	std::cout << "sumXXsumYY -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea[i*imageW+j];*/

	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),imageW*imageH);
	
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH,EPS_float,0);
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH,0,1);
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),imageW*imageH,EPS_float,1);

	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {
	
	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH))
	b1ZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH))
	b1NonZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH))
	countNonZeroDisp -> cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH))
	
	dx -> cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH))
	sumX -> cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH))
	dy -> cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH))
	sumY -> cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH))
	dxdy -> cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH))
	sumXY -> cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH))
	dxdx-> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH))
	sumXX -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH))
	dydy -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH))
	sumYY -> cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH))

	sumXsumYY -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH))
	sumYsumXX -> cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH))
	sumXXsumYY -> cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH))
	sumYsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH))
	sumXsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH))
	sumXYsumXY -> cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH))
	
	dispX -> cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH))
	dispY -> cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH))
	den -> cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH))

	b1ZeroDen -> cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH))
	b1NonZeroDen -> cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH))
	*/

	// s1, p0
	// cv::gpu::magnitudeSqr(dx,dy,sqrModDisp);
	cv::gpu::magnitudeSqr(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),strCV[0]);
	
	// s2, p0
	// cv::gpu::threshold(sqrModDisp,b1NonZeroDisp,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),EPS_float,1,cv::THRESH_BINARY,strCV[0]);

	// s3, p1 (p0.wait)
	// cv::gpu::multiply(dx,b1NonZeroDisp,dx);
	strCV[0].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),1.0,-1,strCV[1]);
	// s3, p2
	// cv::gpu::multiply(dy,b1NonZeroDisp,dy);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),1.0,-1,strCV[2]);
	// s3, p0 
	// cv::gpu::multiply(sqrModDisp,b1NonZeroDisp,sqrModDisp);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),1.0,-1,strCV[0]);

	// s4, p3 (p1.wait)
	// cv::gpu::multiply(dx,dx,dxdx);
	strCV[1].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),1.0,-1,strCV[3]);
	// s4, p4 (p2.wait)
	// cv::gpu::multiply(dy,dy,dydy);
	strCV[2].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)),1.0,-1,strCV[4]);
	// s4, p5 
	// cv::gpu::multiply(dx,dy,dxdy);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)),1.0,-1,strCV[5]);
	// s4, p0
	// cv::gpu::threshold(sqrModDisp,b1ZeroDisp,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH)),0,1,cv::THRESH_BINARY_INV,strCV[0]);
	// s4, p0
	// cv::gpu::add(sqrModDisp, b1ZeroDisp, sqrModDisp);
	cv::gpu::add(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,imageH,nOrient*imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),mask0,-1,strCV[0]);

	// s5, p3 (p0.wait)
	// cv::gpu::divide(dxdx,sqrModDisp,dxdx);
	strCV[0].waitForCompletion();
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,nOrient*imageW,imageH)),1.0,-1,strCV[3]);
	// s5, p4 
	// cv::gpu::divide(dydy,sqrModDisp,dydy);
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,nOrient*imageW,imageH)),1.0,-1,strCV[4]);
	// s5, p5 
	// cv::gpu::divide(dxdy,sqrModDisp,dxdy);    
	cv::gpu::divide(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,nOrient*imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,nOrient*imageW,imageH)),1.0,-1,strCV[5]);    

	for (int theta=1; theta<nOrient; theta++) {
		// s6, p1
		// cv::gpu::add(sumX,dx(theta),sumX);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[1]);
		// s6, p2
		// cv::gpu::add(sumY,dy(theta),sumY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(theta*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)),mask0,-1,strCV[2]);
		// s6, p3
		// cv::gpu::add(sumXX, dxdx(theta),sumXX);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)),mask0,-1,strCV[3]);
		// s6, p4
		// cv::gpu::add(sumYY, dydy(theta),sumYY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW+theta*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)),mask0,-1,strCV[4]);
		// s6, p5
		// cv::gpu::add(sumXY, dxdy(theta),sumXY);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),mask0,-1,strCV[5]);
		// s6, p0
		// cv::gpu::add(countNonZeroDisp, b1NonZeroDisp(theta),countNonZeroDisp);
		cv::gpu::add(cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW+theta*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),mask0,-1,strCV[0]);
	}

	// s7, p1 (p5.wait)
	// cv::gpu::multiply(sumX,sumXY,sumXsumXY);
	strCV[5].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)),1.0,-1,strCV[1]);
	// s7, p2
	// cv::gpu::multiply(sumY, sumXY, sumYsumXY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),1.0,-1,strCV[2]);
	// s7, p3 
	// cv::gpu::multiply(sumY, sumXX, sumYsumXX);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s7, p4 
	// cv::gpu::multiply(sumX,sumYY,sumXsumYY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s7, p5
	// cv::gpu::multiply(sumXY,sumXY,sumXYsumXY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(2*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);
	// s7, p5 
	// cv::gpu::multiply(sumXX, sumYY, sumXXsumYY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(nOrient*imageW,imageH,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);
	// s7, p0
	// cv::gpu::threshold(countNonZeroDisp,countNonZeroDisp,nMinOri,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)),nMinOri,1,cv::THRESH_BINARY,strCV[0]);

	// s8, p5
	// cv::gpu::subtract(sumXXsumYY,sumXYsumXY,den);
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(5*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),mask0,-1,strCV[5]);
	// s8, p4 (p2.wait)
	// cv::gpu::subtract(sumXsumYY,sumYsumXY,dispX);
	strCV[2].waitForCompletion();
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),mask0,-1,strCV[4]);
	// s8, p3 (p1.wait)
	// cv::gpu::subtract(sumYsumXX,sumXsumXY,dispY);
	strCV[1].waitForCompletion();
	cv::gpu::subtract(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),mask0,-1,strCV[3]);
	
	/*cv::Mat temporanea(imageH,imageW,CV_32FC1);
	cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)).download(temporanea);
	std::cout << "den -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea.at<float>(i,j);*/

	// s9, p5
	// cv::gpu::threshold(den, b1NonZeroDen,EPS_float,1,cv::THRESH_BINARY);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)),EPS_float,1,cv::THRESH_BINARY,strCV[5]);
	
	// s10, p4 (p5.wait)
	// cv::gpu::multiply(dispX, b1NonZeroDen, dispX);
	strCV[5].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s10, p3
	// cv::gpu::multiply(dispY, b1NonZeroDen, dispY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s10, p5
	// cv::gpu::multiply(den, b1NonZeroDen, den);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(4*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),1.0,-1,strCV[5]);

	// s11, p4 (p0.wait)
	// cv::gpu::multiply(dispX, countNonZeroDisp, dispX);
	strCV[0].waitForCompletion();
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s11, p3 
	// cv::gpu::multiply(dispY, countNonZeroDisp, dispY);
	cv::gpu::multiply(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[1](cv::Rect(3*nOrient*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);
	// s11, p5
	// cv::gpu::threshold(den, b1ZeroDen,0,1,cv::THRESH_BINARY_INV);
	cv::gpu::threshold(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)), cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),0,1,cv::THRESH_BINARY_INV,strCV[5]);
	// s11, p5
	// cv::gpu::add(den,b1ZeroDen,den);
	cv::gpu::add(cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(3*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),mask0,-1,strCV[5]);

	// s12, p4 (p5.wait)
	// cv::gpu::divide(dispX,den,dispX);
	strCV[5].waitForCompletion();
	cv::gpu::divide(cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(0,0,imageW,imageH)),1.0,-1,strCV[4]);
	// s12, p3
	// cv::gpu::divide(dispY,den,dispY);
	cv::gpu::divide(cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(2*imageW,0,imageW,imageH)),cell_devCVrepOriRepPhase[0](cv::Rect(imageW,0,imageW,imageH)),1.0,-1,strCV[3]);

	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}
void Population::solveApertureCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent)  {

	if (bEvent) 
		checkCudaErrors( cudaEventRecord(start_event, 0) );

	/*
	sqrModDisp -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	b1NonZeroDisp -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) 
	countNonZeroDisp -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	
	dx -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	sumX -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dy -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient)
	sumY -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dxdy -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient)
	sumXY -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dxdx-> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient)
	sumXX -> cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH)
	dydy -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient)
	sumYY -> cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH)

	sumXsumYY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) 
	sumYsumXX -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH)
	sumXXsumYY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH) 
	sumYsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH)
	sumXsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH)
	sumXYsumXY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH)
	
	dispX -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) 
	dispY -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH)
	den -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH)

	b1NonZeroDen -> cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH)
	*/

	// s1, p1
	//nppsSqr_32f(dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	nppsSqr_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s1, p2
	//nppsSqr_32f(dy,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[2]);
	nppsSqr_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s1, p0 (p1.wait, p2.wait)
	//nppsAdd_32f(dxdx,dydy,sqrModDisp,nOrient*imageW*imageH);
	nppSetStream(strCU[0]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[1],0) );
	checkCudaErrors( cudaStreamWaitEvent(strCU[0], kernelEventStrCU[2],0) );
	nppsAdd_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	
	// s2, p0
	//nppsThreshold_LTVal_32f(sqrModDisp,b1NonZeroDisp,nOrient*imageW*imageH,EPS_float,0);
	nppsThreshold_LTVal_32f(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,nOrient*imageW*imageH,EPS_float,0);
	// s2, p0
	//nppsThreshold_GTVal_32f_I(b1NonZeroDisp,nOrient*imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,nOrient*imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );

	// s3, p1 (p0.wait)
	//nppsMul_32f_I(b1NonZeroDisp,dx,nOrient*imageW*imageH);
	nppSetStream(strCU[1]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[1], kernelEventStrCU[0],0) );
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s3, p2
	//nppsMul_32f_I(b1NonZeroDisp,dy,nOrient*imageW*imageH);
	nppSetStream(strCU[2]);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) ,cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s3, p0
	//nppsThreshold_LTVal_32f_I(sqrModDisp,nOrient*imageW*imageH,EPS_float,1);
	nppSetStream(strCU[0]);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),nOrient*imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );

	// s4, p3 (p1.wait)
	//nppsMul_32f(dx,dx,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[1],0) );
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	// s4, p4 (p2.wait)
	//nppsMul_32f(dy,dy,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[2],0) );
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	// s4, p5
	//nppsMul_32f(dx,dy,dxdy,nOrient*imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient),nOrient*imageW*imageH);

	// s5, p3 (p0.wait)
	//nppsDiv_32f_I(sqrModDisp,dxdx,nOrient*imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[0],0) );
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	// s5, p4
	//nppsDiv_32f_I(sqrModDisp,dydy,nOrient*imageW*imageH);
	nppSetStream(strCU[4]);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient),nOrient*imageW*imageH);
	// s5, p5
	//nppsDiv_32f_I(sqrModDisp,dxdy,nOrient*imageW*imageH);
	nppSetStream(strCU[5]);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient),nOrient*imageW*imageH);

	for (int theta=1; theta<nOrient; theta++) {
		// s6, p1
		//nppsAdd_32f_I(dx(theta),sumX,imageW*imageH);
		nppSetStream(strCU[1]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		// s6, p2
		//nppsAdd_32f_I(dy(theta),sumY,imageW*imageH);
		nppSetStream(strCU[2]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		// s6, p3
		//nppsAdd_32f_I(dxdx(theta),sumXX,imageW*imageH);
		nppSetStream(strCU[3]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		// s6, p4
		//nppsAdd_32f_I(dxdy(theta),sumXY,imageW*imageH);
		nppSetStream(strCU[4]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		// s6, p5
		//nppsAdd_32f_I(dydy(theta),sumYY,imageW*imageH);
		nppSetStream(strCU[5]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
		checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
		// s6, p0
		//nppsAdd_32f_I(b1NonZeroDisp(theta),countNonZeroDisp,imageW*imageH);
		nppSetStream(strCU[0]);
		nppsAdd_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + theta*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH);
	}
	
	// s7, p0
	//nppsThreshold_LTVal_32f_I(countNonZeroDisp,imageW*imageH,nMinOri,0);
	nppSetStream(strCU[0]);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH,nMinOri,0);
	// s7, p1 (p5.wait)
	//nppsMul_32f(sumX,sumXY,sumXsumXY,imageW*imageH);
	nppSetStream(strCU[1]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[1], kernelEventStrCU[5],0) );
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[1], streamCU[1]) );
	// s7, p2
	//nppsMul_32f(sumY, sumXY, sumYsumXY,imageW*imageH);
	nppSetStream(strCU[2]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH),imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[2], streamCU[2]) );
	// s7, p3
	//nppsMul_32f(sumY, sumXX, sumYsumXX,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	// s7, p4
	//nppsMul_32f(sumX,sumYY,sumXsumYY,imageW*imageH);
	nppSetStream(strCU[4]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	// s7, p5
	//nppsMul_32f(sumXY,sumXY,sumXYsumXY,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 2*(imageW*imageH*nOrient) + 0*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH),imageW*imageH);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	// s7, p5
	//nppsMul_32f(sumXX, sumYY, sumXXsumYY,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsMul_32f(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[1] + 1*(imageW*imageH*nOrient*nPhase) + 1*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH) ,imageW*imageH);
	// s7, p0
	//nppsThreshold_GTVal_32f_I(countNonZeroDisp,imageW*imageH,0,1);
	nppSetStream(strCU[0]);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH),imageW*imageH,0,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[0], streamCU[0]) );

	// s8, p5
	//nppsSub_32f_I(sumYsumXY,sumXsumYY=dispX,imageW*imageH);
	nppSetStream(strCU[5]);
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 3*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	// s8, p4 (p2.wait)
	//nppsSub_32f_I(sumXsumXY,sumYsumXX=dispY,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[2],0) );
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	// s8, p3 (p1.wait)
	//nppsSub_32f_I(sumXYsumXY,sumXXsumYY=den,imageW*imageH);
	nppSetStream(strCU[3]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[3], kernelEventStrCU[1],0) );
	nppsSub_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 5*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),imageW*imageH);
	
	/*
	float *temporanea = (float*) malloc(imageW*imageH*sizeof(float));
	checkCudaErrors( cudaMemcpy(temporanea,cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),imageW*imageH*sizeof(float),cudaMemcpyDeviceToHost) );
	std::cout << "den -->" << std::endl;
	for (int i=20; i<30; i++)
		for (int j=20; j<30; j++)
			std::cout << temporanea[i*imageW+j];
	*/

	// s9, p5
	//nppsThreshold_LTVal_32f(den,b1NonZeroDen,imageW*imageH,EPS_float,0);
	nppSetStream(strCU[5]);
	nppsThreshold_LTVal_32f(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH,EPS_float,0);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );
	// s9, p5
	//nppsThreshold_GTVal_32f_I(b1NonZeroDen,imageW*imageH,0,1);
	nppsThreshold_GTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH),imageW*imageH,0,1);
	
	// s10, p4 (p5.wait)
	//nppsMul_32f_I(b1NonZeroDen, dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[5],0) );
	nppsMul_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	// s10, p3
	//nppsMul_32f_I(b1NonZeroDen, dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 4*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	// s10, p5
	//nppsThreshold_LTVal_32f_I(den,imageW*imageH,EPS_float,1);
	nppSetStream(strCU[5]);
	nppsThreshold_LTVal_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),imageW*imageH,EPS_float,1);
	checkCudaErrors( cudaEventRecord(kernelEventStreamCU[5], streamCU[5]) );

	// s11, p4 (p0.wait)
	//nppsMul_32f_I(countNonZeroDisp, dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[0],0) );
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	// s11, p3
	//nppsMul_32f_I(countNonZeroDisp, dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsMul_32f_I(cell_devCUrepOriRepPhase[1] + 0*(imageW*imageH*nOrient*nPhase) + 3*(imageW*imageH*nOrient) + 0*(imageW*imageH), cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);
	
	// s12, p4 (p5.wait)
	//nppsDiv_32f_I(den,dispX,imageW*imageH);
	nppSetStream(strCU[4]);
	checkCudaErrors( cudaStreamWaitEvent(strCU[4], kernelEventStrCU[5],0) );
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 0*(imageW*imageH) ,imageW*imageH);
	// s12, p3
	//nppsDiv_32f_I(den,dispY,imageW*imageH);
	nppSetStream(strCU[3]);
	nppsDiv_32f_I(cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 2*(imageW*imageH),cell_devCUrepOriRepPhase[0] + 0*(imageW*imageH*nOrient*nPhase) + 0*(imageW*imageH*nOrient) + 1*(imageW*imageH),imageW*imageH);

	nppSetStream(0);
	
	// wait until the GPU is done
    if (bEvent) {
		checkCudaErrors( cudaEventRecord(stop_event, 0) );
		checkCudaErrors( cudaEventSynchronize(stop_event) );
		checkCudaErrors( cudaEventElapsedTime(elapsed_time, start_event, stop_event) );
	} else 
		*elapsed_time=-1;
}

// Comparing and printing results
/////////////////////////////////////////////////////////////////////////////////////////////

double Population::compareSimpleAnswer(float** result) {

	double delta = 0, sum = 0, deltaTot = 0, sumTot = 0;
	int phase;

    for (int i = 0; i <imageH; i++) 
		for (int j = 0; j < imageW; j++) 
			for (int eye = 0; eye < nEye; eye++) {
				delta = 0;
				sum = 0;
				if (eye == 0) {
					phase = 0;
					for (int theta = 0; theta < nOrient; theta++) {
						float* Re = (float*)(cell_hostCV[eye][0][theta][phase].data + cell_hostCV[eye][0][theta][phase].step*i + j*cell_hostCV[eye][0][theta][phase].elemSize());
						float vRe = *Re;
						float* Im = (float*)(cell_hostCV[eye][1][theta][phase].data + cell_hostCV[eye][1][theta][phase].step*i + j*cell_hostCV[eye][1][theta][phase].elemSize());
						float vIm = *Im;
						delta += (vRe - cell_hostCU[eye][0][theta][phase][i*imageW+j]) * (vRe - cell_hostCU[eye][0][theta][phase][i*imageW+j]);
						delta += (vIm - cell_hostCU[eye][1][theta][phase][i*imageW+j]) * (vIm - cell_hostCU[eye][1][theta][phase][i*imageW+j]);
						sum   += vRe * cell_hostCU[eye][0][theta][phase][i*imageW+j];
						sum   += vIm * cell_hostCU[eye][1][theta][phase][i*imageW+j];
					} 
				} else 
					for (int theta = 0; theta < nOrient; theta++) 
						for (phase = 0; phase<nPhase; phase++) {
							float* Re = (float*)(cell_hostCV[eye][0][theta][phase].data + cell_hostCV[eye][0][theta][phase].step*i + j*cell_hostCV[eye][0][theta][phase].elemSize());
							float vRe = *Re;
							float* Im = (float*)(cell_hostCV[eye][1][theta][phase].data + cell_hostCV[eye][1][theta][phase].step*i + j*cell_hostCV[eye][1][theta][phase].elemSize());
							float vIm = *Im;
							delta += (vRe - cell_hostCU[eye][0][theta][phase][i*imageW+j]) * (vRe - cell_hostCU[eye][0][theta][phase][i*imageW+j]);
							delta += (vIm - cell_hostCU[eye][1][theta][phase][i*imageW+j]) * (vIm - cell_hostCU[eye][1][theta][phase][i*imageW+j]);
							sum   += vRe * cell_hostCU[eye][0][theta][phase][i*imageW+j];
							sum   += vIm * cell_hostCU[eye][1][theta][phase][i*imageW+j];
						}
				result[eye][i*imageW+j] = (float)sqrt(delta / sum) ;
				deltaTot += delta;
				sumTot += sum;
			}
					
        double L2normOutput = sqrt(deltaTot / sumTot);
		return L2normOutput;
}
double Population::compareEnergy(float* result) {

	double delta = 0, sum = 0, deltaTot = 0, sumTot = 0;
	
    for (int i = 0; i <imageH; i++) 
		for (int j = 0; j < imageW; j++) {
			delta = 0;
			sum = 0;
			for (int theta = 0; theta<nOrient; theta++) 
				for (int phase = 0; phase<nPhase; phase++) {
					delta += std::pow((cell_hostCV[1][0][theta][phase].at<float>(i,j) - cell_hostCU[1][0][theta][phase][i*imageW+j]),2);
					sum   += cell_hostCV[1][0][theta][phase].at<float>(i,j) * cell_hostCU[1][0][theta][phase][i*imageW+j];
				}
			result[i*imageW+j] = (float)sqrt(delta / sum) ;
			deltaTot += delta;
			sumTot += sum;
		}
					
    double L2normOutput = sqrt(deltaTot / sumTot);
	return L2normOutput;
}
double Population::compareCenterOfMass(float* result) {

	double delta = 0, sum = 0, deltaTot = 0, sumTot = 0;
	
    for (int i = 0; i <imageH; i++) 
		for (int j = 0; j < imageW; j++) {
			delta = 0;
			sum = 0;
			for (int theta = 0; theta<nOrient; theta++) {
					delta += std::pow((cell_hostCV[1][0][theta][1].at<float>(i,j) - cell_hostCU[1][0][theta][1][i*imageW+j]),2);
					sum   += cell_hostCV[1][0][theta][1].at<float>(i,j) * cell_hostCU[1][0][theta][1][i*imageW+j];
			}
			result[i*imageW+j] = (float)sqrt(delta / sum) ;
			deltaTot += delta;
			sumTot += sum;
		}
					
    double L2normOutput = sqrt(deltaTot / sumTot);
	return L2normOutput;
}
double Population::compareXYCenterOfMass(float* result) {

	double delta = 0, sum = 0, deltaTot = 0, sumTot = 0;
	
    for (int i = 0; i <imageH; i++) 
		for (int j = 0; j < imageW; j++) {
			delta = 0;
			sum = 0;
			for (int theta = 0; theta<nOrient; theta++) {
					delta += std::pow((cell_hostCV[1][0][theta][0].at<float>(i,j) - cell_hostCU[1][0][theta][0][i*imageW+j]),2);
					delta += std::pow((cell_hostCV[1][1][theta][0].at<float>(i,j) - cell_hostCU[1][1][theta][0][i*imageW+j]),2);
					sum   += cell_hostCV[1][0][theta][0].at<float>(i,j) * cell_hostCU[1][0][theta][0][i*imageW+j];
					sum   += cell_hostCV[1][1][theta][0].at<float>(i,j) * cell_hostCU[1][1][theta][0][i*imageW+j];
			}
			result[i*imageW+j] = (float)sqrt(delta / sum) ;
			deltaTot += delta;
			sumTot += sum;
		}
					
    double L2normOutput = sqrt(deltaTot / sumTot);
	return L2normOutput;
}
double Population::compareDisparity(float* result) {

	double delta = 0, sum = 0, deltaTot = 0, sumTot = 0;
	
    for (int i = 0; i <imageH; i++) 
		for (int j = 0; j < imageW; j++) {
			delta = std::pow((cell_hostCV[0][0][0][0].at<float>(i,j) - cell_hostCU[0][0][0][0][i*imageW+j]),2);
			delta += std::pow((cell_hostCV[0][0][1][0].at<float>(i,j) - cell_hostCU[0][0][1][0][i*imageW+j]),2);
			sum   = cell_hostCV[0][0][0][0].at<float>(i,j) * cell_hostCU[0][0][0][0][i*imageW+j];
			sum   += cell_hostCV[0][0][1][0].at<float>(i,j) * cell_hostCU[0][0][1][0][i*imageW+j];
			result[i*imageW+j] = (float)sqrt(delta / sum) ;
			deltaTot += delta;
			sumTot += sum;
		}
			
    double L2normOutput = sqrt(deltaTot / sumTot);
	return L2normOutput;
}

void Population::printFileSimpleAnswerCUDA(std::string filenameRe, std::string filenameIm, int eye, int theta, int phase) {

	std::ofstream f_outputRe, f_outputIm;
    f_outputRe.open(filenameRe);
	f_outputIm.open(filenameIm);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemRe = cell_hostCU[eye][0][theta][phase][i*imageW+j];
			f_outputRe << elemRe << "  ";
			float elemIm = cell_hostCU[eye][1][theta][phase][i*imageW+j];
			f_outputIm << elemIm << "  ";
		}
		f_outputRe << "\n\n";
		f_outputIm << "\n\n";
	}
	f_outputRe.close();
	f_outputIm.close();

}
void Population::printFileSimpleAnswerOPENCV(std::string filenameRe, std::string filenameIm, int eye, int theta, int phase) {

	std::ofstream f_outputRe, f_outputIm;
    f_outputRe.open(filenameRe);
	f_outputIm.open(filenameIm);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemRe = cell_hostCV[eye][0][theta][phase].at<float>(i,j);
			f_outputRe << elemRe << "  ";
			float elemIm = cell_hostCV[eye][1][theta][phase].at<float>(i,j);
			f_outputIm << elemIm << "  ";
		}
		f_outputRe << "\n\n";
		f_outputIm << "\n\n";
	}
	f_outputRe.close();
	f_outputIm.close();

}
void Population::printFileEnergyCUDA(std::string filename, int theta, int phase) {

	std::ofstream f_output;
    f_output.open(filename);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elem = cell_hostCU[1][0][theta][phase][i*imageW+j];
			f_output << elem << "  ";
		}
		f_output << "\n\n";
	}
	f_output.close();

}
void Population::printFileEnergyOPENCV(std::string filename, int theta, int phase){

	std::ofstream f_output;
    f_output.open(filename);

	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elem = cell_hostCV[1][0][theta][phase].at<float>(i,j);
			f_output << elem << "  ";
		}
		f_output << "\n\n";
	}
	f_output.close();
}
void Population::printFileCenterOfMassCUDA(std::string filename, int theta) {

	std::ofstream f_output;
    f_output.open(filename);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elem = cell_hostCU[1][0][theta][1][i*imageW+j];
			f_output << elem << "  ";
		}
		f_output << "\n\n";
	}
	f_output.close();
}
void Population::printFileCenterOfMassOPENCV(std::string filename, int theta){

	std::ofstream f_output;
    f_output.open(filename);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elem = cell_hostCV[1][0][theta][1].at<float>(i,j);
			f_output << elem << "  ";
		}
		f_output << "\n\n";
	}
	f_output.close();
}
void Population::printFileXYCenterOfMassCUDA(std::string filenameX, std::string filenameY, int theta) {

	std::ofstream f_outputX, f_outputY;
    f_outputX.open(filenameX);
	f_outputY.open(filenameY);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemX = cell_hostCU[1][0][theta][0][i*imageW+j];
			float elemY = cell_hostCU[1][1][theta][0][i*imageW+j];
			f_outputX << elemX << "  ";
			f_outputY << elemY << "  ";
		}
		f_outputX << "\n\n";
		f_outputY << "\n\n";
	}
	f_outputX.close();
	f_outputY.close();
}
void Population::printFileXYCenterOfMassOPENCV(std::string filenameX, std::string filenameY, int theta) {

	std::ofstream f_outputX, f_outputY;
    f_outputX.open(filenameX);
	f_outputY.open(filenameY);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemX = cell_hostCV[1][0][theta][0].at<float>(i,j);
			float elemY = cell_hostCV[1][1][theta][0].at<float>(i,j);
			f_outputX << elemX << "  ";
			f_outputY << elemY << "  ";
		}
		f_outputX << "\n\n";
		f_outputY << "\n\n";
	}
	f_outputX.close();
	f_outputY.close();
}
void Population::printFileDisparityCUDA(std::string filenameX, std::string filenameY) {

	std::ofstream f_outputX, f_outputY;
    f_outputX.open(filenameX);
	f_outputY.open(filenameY);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemX = (dst_hostCU + 0*imageW*imageH)[i*imageW+j];
			float elemY = (dst_hostCU + 1*imageW*imageH)[i*imageW+j];
			f_outputX << elemX << "  ";
			f_outputY << elemY << "  ";
		}
		f_outputX << "\n\n";
		f_outputY << "\n\n";
	}
	f_outputX.close();
	f_outputY.close();
}
void Population::printFileDisparityOPENCV(std::string filenameX, std::string filenameY) {

	std::ofstream f_outputX, f_outputY;
    f_outputX.open(filenameX);
	f_outputY.open(filenameY);
			
	for(int i=0; i<imageH; i++) {
		for(int j=0; j<imageW; j++) {
			float elemX = dst_hostCV(cv::Rect(0*imageW,0,imageW,imageH)).at<float>(i,j);
			float elemY = dst_hostCV(cv::Rect(1*imageW,0,imageW,imageH)).at<float>(i,j);
			f_outputX << elemX << "  ";
			f_outputY << elemY << "  ";
		}
	f_outputX << "\n\n";
	f_outputY << "\n\n";
	}
	
	f_outputX.close();
	f_outputY.close();
}