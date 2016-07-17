#ifndef FILTERS_H
#define FILTERS_H

#include "quantities.h"

#include <iostream>
#include <fstream>

#include <ctime>
#include <cstdio>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

//#include <shrQATest.h>
#include <helper_functions.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Filters {

public:
	cv::Mat filters1D_hostCV[nFilters1D];
	float filters1DCV[nFilters1D][taps];
	float filters1DCU[nFilters1D][taps];
	
	cv::Ptr<cv::gpu::BaseColumnFilter_GPU> filters1Dcolumn_devCV[nFilters1D];
	cv::Ptr<cv::gpu::BaseRowFilter_GPU> filters1Drow_devCV[nFilters1D];

	double disp[nPhase];
    double phShift[nPhase];
	float oriTuning[nOrient];
	float cosPhShift[nPhase];
	float sinPhShift[nPhase];
	float cosOriTuning[nOrient];
	float sinOriTuning[nOrient];

	cv::Mat filtGauss_hostCV;
	float filtGauss[filtGaussLength];

	cv::Ptr<cv::gpu::BaseRowFilter_GPU> filtGaussRow_devCV;
	cv::Ptr<cv::gpu::BaseColumnFilter_GPU> filtGaussColumn_devCV;

	cv::gpu::GpuMat cosPhShift_devCVrepOriRepPhase;
	cv::gpu::GpuMat sinPhShift_devCVrepOriRepPhase;
	cv::gpu::GpuMat minusSinPhShift_devCVrepOriRepPhase;
	cv::gpu::GpuMat sumCosSinPhShift_devCVrepOriRepPhase;
	cv::gpu::GpuMat sumCosMinusSinPhShift_devCVrepOriRepPhase;
	float *cosPhShift_devCUrepOriRepPhase;
	float *sinPhShift_devCUrepOriRepPhase;
	float *minusSinPhShift_devCUrepOriRepPhase;

	cv::gpu::GpuMat cosOriTuning_devCVoneOri;
	cv::gpu::GpuMat sinOriTuning_devCVoneOri;
	float *cosOriTuning_devCUoneOri;
	float *sinOriTuning_devCUoneOri;

	cv::gpu::GpuMat cosOriTuning_devCVrepOriRepPhase;
	cv::gpu::GpuMat sinOriTuning_devCVrepOriRepPhase;
	float *cosOriTuning_devCUrepOriRepPhase;
	float *sinOriTuning_devCUrepOriRepPhase;

	Filters(std::string filename, double *disp, float *filtGauss);

	void freeCV();
	void freeCU();

	void uploadCV();
	void uploadCU();

	void mallocGPUphShiftCVrepOriRepPhase(int _imageW, int _imageH);
	void mallocGPUphShiftCUrepOriRepPhase(int _imageW, int _imageH);
	void freeGPUphShiftCVrepOriRepPhase();
	void freeGPUphShiftCUrepOriRepPhase();
	void uploadGPUphShiftCVrepOriRepPhase(int _imageW, int _imageH);
	void uploadGPUphShiftCUrepOriRepPhase(int _imageW, int _imageH);

	void mallocGPUoriTuningCVrepOriRepPhase(int _imageW, int _imageH);
	void mallocGPUoriTuningCUrepOriRepPhase(int _imageW, int _imageH);
	void freeGPUoriTuningCVrepOriRepPhase();
	void freeGPUoriTuningCUrepOriRepPhase();
	void uploadGPUoriTuningCVrepOriRepPhase(int _imageW, int _imageH);
	void uploadGPUoriTuningCUrepOriRepPhase(int _imageW, int _imageH);

	void mallocGPUoriTuningCVoneOriSepPhase(int _imageW, int _imageH);
	void mallocGPUoriTuningCUoneOriSepPhase(int _imageW, int _imageH);
	void freeGPUoriTuningCVoneOriSepPhase();
	void freeGPUoriTuningCUoneOriSepPhase();
	void uploadGPUoriTuningCVoneOriSepPhase(int _imageW, int _imageH);
	void uploadGPUoriTuningCUoneOriSepPhase(int _imageW, int _imageH);
};

#endif

