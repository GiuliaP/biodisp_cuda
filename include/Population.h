#ifndef POPULATION_H
#define POPULATION_H

#include <cuda_runtime.h>
//#include <shrQATest.h>
#include <helper_functions.h>
//#include <helper_cuda.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/stream_accessor.hpp"

#include <string>
#include <stdio.h>

#include "Filters.h"

class Population {

public:
	int imageW;
	int imageH;

	cudaEvent_t start_event, stop_event;
	StopWatchInterface *funcTimer;

	// OPENCV data
	/////////////////////////////////////////////////////////////////////////////////////////////

	cv::Mat *src_hostCV;
	cv::Mat dst_hostCV;
	cv::gpu::GpuMat *src_devCV;
	
	cv::gpu::GpuMat *tempCV;
	cv::gpu::GpuMat *tempCVoneOri;
	cv::gpu::GpuMat *tempCVrepOriRepPhase;

	cv::gpu::Stream *strCV;
	cv::gpu::Stream *streamCV;

	cv::gpu::GpuMat mask0;

	cv::gpu::GpuMat ****cell_devCV;
	cv::gpu::GpuMat ***cell_devCVoneOri;
	cv::gpu::GpuMat *cell_devCVrepOriRepPhase;

	cv::Mat ****cell_hostCV;

	// CUDA data
	/////////////////////////////////////////////////////////////////////////////////////////////

	float **src_hostCU;
	float *dst_hostCU;
	float **src_devCU;

	float **tempCU;
	float **tempCUoneOri;
	float **tempCUrepOriRepPhase;

	cudaStream_t* strCU;
	cudaStream_t *streamCU;
	cudaEvent_t *kernelEventStrCU;
	cudaEvent_t *kernelEventStreamCU;
	
	float *****cell_devCU;
	float ****cell_devCUoneOri;
	float **cell_devCUrepOriRepPhase;

	float *****cell_hostCU;

	// Constructor
	/////////////////////////////////////////////////////////////////////////////////////////////
	
	Population();

	// General utility functions
	/////////////////////////////////////////////////////////////////////////////////////////////

	void call(std::string filename, std::string functionName, void (Population::*funcToCall)(Filters *,float *,bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, Filters *f);
	void callAll(std::string filename, std::string *functionName, void (Population::**funcToCall)(Filters *,float *,bool), int _nFunc, int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, bool bTimer, Filters *f);
	
	void call(std::string filename, std::string functionName, void (Population::*funcToCall)(cv::Mat &,float *, bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, cv::Mat &data);
	void call(std::string filename, std::string functionName, void (Population::*funcToCall)(cv::Mat *,float *, bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, cv::Mat *data);
	void call(std::string filename, std::string functionName, void (Population::*funcToCall)(float **,int,int,float *, bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, float **data,int imageW,int imageH);
	void call(std::string filename, std::string functionName, void (Population::*funcToCall)(float *,float *, bool), int iterations, bool warmUpIt,  StopWatchInterface *timer, bool bEvent, float *data);

	// Image pair managing
	/////////////////////////////////////////////////////////////////////////////////////////////

	void mallocCPUImagePairCUDA(int _imageH, int _imageW);
	void mallocCPUImagePairOPENCV(int _imageH, int _imageW);
	void mallocCPUResultCUDA(int _imageH, int _imageW);
	void mallocCPUResultOPENCV(int _imageH, int _imageW);
	void freeCPUImagePairCUDA();
	void freeCPUImagePairOPENCV();
	void freeCPUResultCUDA();
	void freeCPUResultOPENCV();

	void loadImagePairOPENCV(std::string *src);
	void loadImagePairCUDA(char** src, char* argv0);
	void copyCPUImagePairToCUDA(cv::Mat* _src_hostCV);
	
	void mallocGPUImagePairCUDA(int _imageH, int _imageW);
	void mallocGPUImagePairOPENCV(int _imageH, int _imageW);
	void freeGPUImagePairOPENCV();
	void freeGPUImagePairCUDA();

	void uploadImagePairOPENCV(cv::Mat* _src_hostCV, float* elapsed_time, bool bEvent);
	void uploadImagePairCUDA(float **_src_hostCU, int _imageH, int _imageW, float* elapsed_time, bool bEvent);

	// Cells managing
	/////////////////////////////////////////////////////////////////////////////////////////////

	void mallocGPUCellOPENCV();
	void mallocGPUCellCUDA();
	void freeGPUCellOPENCV();
	void freeGPUCellCUDA();

	void mallocGPUCellOPENCVoneOri();
	void mallocGPUCellCUDAoneOri();
	void freeGPUCellOPENCVoneOri();
	void freeGPUCellCUDAoneOri();

	void mallocGPUCellOPENCVrepOriRepPhase();
	void mallocGPUCellCUDArepOriRepPhase();
	void freeGPUCellOPENCVrepOriRepPhase();
	void freeGPUCellCUDArepOriRepPhase();

	void mallocCPUCellOPENCV();
	void mallocCPUCellCUDA();
	void freeCPUCellOPENCV();
	void freeCPUCellCUDA();

	void mallocGPUTempMatricesCUDA();
	void mallocGPUTempMatricesOPENCV();
	void freeGPUTempMatricesCUDA();
	void freeGPUTempMatricesOPENCV();
	void set0GPUTempMatricesCUDA(int *idx, int N);
	void set0GPUTempMatricesOPENCV(int *idx, int N);

	void mallocGPUTempMatricesCUDAoneOri();
	void mallocGPUTempMatricesOPENCVoneOri();
	void freeGPUTempMatricesCUDAoneOri();
	void freeGPUTempMatricesOPENCVoneOri();
	void set0GPUTempMatricesCUDAoneOri(int *idx, int N);
	void set0GPUTempMatricesOPENCVoneOri(int *idx, int N);

	void mallocGPUTempMatricesCUDArepOriRepPhase();
	void mallocGPUTempMatricesOPENCVrepOriRepPhase();
	void freeGPUTempMatricesCUDArepOriRepPhase();
	void freeGPUTempMatricesOPENCVrepOriRepPhase();
	void set0GPUTempMatricesCUDArepOriRepPhase(int *idx, int N);
	void set0GPUTempMatricesOPENCVrepOriRepPhase(int *idx, int N);

	void mallocGPUStreamsCUDA();
	void mallocGPUStreamsOPENCV();
	void freeGPUStreamsCUDA();
	void freeGPUStreamsOPENCV();

	void createStartStopEvents();
	void destroyStartStopEvents();
	void createFuncTimer();
	void destroyFuncTimer();

	void downloadSimpleAnswerOPENCV();
	void downloadSimpleAnswerCUDA();
	void downloadSimpleAnswerOPENCVoneOri();
	void downloadSimpleAnswerCUDAoneOri();
	void downloadSimpleAnswerOPENCVrepOriRepPhase();
	void downloadSimpleAnswerCUDArepOriRepPhase();

	void downloadResultOPENCV(cv::Mat& _dst_hostCV, float* elapsed_time, bool bEvent);
	void downloadResultCUDA(float *_dst_hostCU, float* elapsed_time, bool bEvent);
	void downloadResultOPENCVoneOri(cv::Mat& _dst_hostCV, float* elapsed_time, bool bEvent);
	void downloadResultCUDAoneOri(float *_dst_hostCU, float* elapsed_time, bool bEvent);
	void downloadResultOPENCVrepOriRepPhase(cv::Mat& _dst_hostCV, float* elapsed_time, bool bEvent);
	void downloadResultCUDArepOriRepPhase(float *_dst_hostCU, float* elapsed_time, bool bEvent);

	// Cells' simple answers calculation
	/////////////////////////////////////////////////////////////////////////////////////////////

	void calcSimpleAnswerOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	
	void calcSimpleAnswerOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void calcSimpleAnswerOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcSimpleAnswerCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Cells' simple answer shifting R wrt L
	/////////////////////////////////////////////////////////////////////////////////////////////

	void shiftSimpleAnswerOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void shiftSimpleAnswerOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void shiftSimpleAnswerOPENCV_oneOriOnePhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_oneOriOnePhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);

	void shiftSimpleAnswerOPENCV_repOriRepPhaseBlendLinear_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void shiftSimpleAnswerCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Cells' energy calculation
	/////////////////////////////////////////////////////////////////////////////////////////////

	void calcEnergyOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void calcEnergyOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void calcEnergyOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcEnergyCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Center of mass
	/////////////////////////////////////////////////////////////////////////////////////////////

	void calcCenterOfMassOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	
	void calcCenterOfMassOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void calcCenterOfMassOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void calcCenterOfMassCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Project center of mass to horizontal/vertical axes
	/////////////////////////////////////////////////////////////////////////////////////////////

	void projectXYCenterOfMassOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	
	void projectXYCenterOfMassOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void projectXYCenterOfMassCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Intersection of constraints
	/////////////////////////////////////////////////////////////////////////////////////////////

	void solveApertureOPENCV_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_sepOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureOPENCV_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_sepOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	
	void solveApertureOPENCV_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_oneOriSepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureOPENCV_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_oneOriSepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	void solveApertureOPENCV_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_repOriRepPhase_noStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureOPENCV_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);
	void solveApertureCUDA_repOriRepPhase_withStreams(Filters *f, float *elapsed_time, bool bEvent);

	// Comparing and printing results
	/////////////////////////////////////////////////////////////////////////////////////////////

	double compareSimpleAnswer(float** result);
	double compareEnergy(float* result);
	double compareCenterOfMass(float* result);
	double compareXYCenterOfMass(float* result);
	double compareDisparity(float* result);

	void printFileSimpleAnswerCUDA(std::string filenameRe, std::string filenameIm, int eye, int theta, int phase);
	void printFileSimpleAnswerOPENCV(std::string filenameRe, std::string filenameIm, int eye, int theta, int phase);
	void printFileEnergyCUDA(std::string filename, int theta, int phase);
	void printFileEnergyOPENCV(std::string filenamee, int theta, int phase);
	void printFileCenterOfMassCUDA(std::string filename, int theta);
	void printFileCenterOfMassOPENCV(std::string filenamee, int theta);
	void printFileXYCenterOfMassCUDA(std::string filenameX, std::string filenameY, int theta);
	void printFileXYCenterOfMassOPENCV(std::string filenameX, std::string filenameY, int theta);
	void printFileDisparityCUDA(std::string filenameX, std::string filenameY);
	void printFileDisparityOPENCV(std::string filenameX, std::string filenameY);

};

#endif
