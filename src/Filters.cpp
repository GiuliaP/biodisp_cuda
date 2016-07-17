#include "Filters.h"
#include "quantities.h"
#include "conv_common.h"

#include <fstream>
#include <iostream>

#include <shrQATest.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <npp.h>
#include <nppdefs.h>

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

Filters::Filters(std::string filename, double* _disp, float *_filtGauss) {

	for (int i=0; i<nPhase; i++) {
		disp[i] = _disp[i];
		phShift[i] = -(2*pigreco*(f0)*disp[i]);
		cosPhShift[i] = (float)std::cos(phShift[i]);
		sinPhShift[i] = (float)std::sin(phShift[i]);
	}

	for (int i = 0; i < nOrient; i++) {
		oriTuning[i] = i*angularBandwidth;
		cosOriTuning[i] = (float)std::cos(oriTuning[i]);
		sinOriTuning[i] = (float)std::sin(oriTuning[i]);
	}

	for (int i=0; i<filtGaussLength; i++)
		filtGauss[i] = _filtGauss[i];
	filtGauss_hostCV = cv::Mat(1,filtGaussLength, CV_32FC1,filtGauss);
	
	std::ifstream f_input(filename,std::ios_base::in);

	for (int i=0; i<nFilters1D; i++) {
		for (int j=0; j<taps; j++) {
			f_input >> filters1DCV[i][j];
			filters1DCU[i][j] = filters1DCV[i][j];
		}
		filters1D_hostCV[i] = cv::Mat(1,taps,CV_32FC1,filters1DCV[i]);
		if (!(i%2) && i>0)
			cv::multiply(filters1D_hostCV[i],-1,filters1D_hostCV[i]);
	}
	f_input.close();  
}

void Filters::freeCV() {

	filtGauss_hostCV.release();
	for (int i=0; i<nFilters1D; i++)
		filters1D_hostCV[i].release();

	filtGaussRow_devCV->~BaseRowFilter_GPU();
	filtGaussColumn_devCV->~BaseColumnFilter_GPU();

	for (int i=0; i<nFilters1D; i++) {
		filters1Drow_devCV[i]->~BaseRowFilter_GPU();
		filters1Dcolumn_devCV[i]->~BaseColumnFilter_GPU();
	}
}
void Filters::freeCU() {
}

void Filters::uploadCV() {

	filtGaussRow_devCV = cv::gpu::getLinearRowFilter_GPU(CV_32FC1,CV_32FC1,filtGauss_hostCV,-1,cv::BORDER_REPLICATE);
	filtGaussColumn_devCV = cv::gpu::getLinearColumnFilter_GPU(CV_32FC1,CV_32FC1,filtGauss_hostCV,-1,cv::BORDER_REPLICATE); 

	for (int i=0; i<nFilters1D; i++) {
		filters1Drow_devCV[i] = cv::gpu::getLinearRowFilter_GPU(CV_32FC1,CV_32FC1,filters1D_hostCV[i],-1,cv::BORDER_REPLICATE);
	    filters1Dcolumn_devCV[i] = cv::gpu::getLinearColumnFilter_GPU(CV_32FC1,CV_32FC1,filters1D_hostCV[i],-1,cv::BORDER_REPLICATE); 
	}  
}
void Filters::uploadCU() {

	/*copyDispToDev(disp);
	copyPhShiftToDev(phShift);
	copyOriTuningToDev(oriTuning);
	copyCosPhShiftToDev(cosPhShift);
	copySinPhShiftToDev(sinPhShift);
	copyCosOriTuningToDev(cosOriTuning);
	copySinOriTuningToDev(sinOriTuning);
	*/
  
	for (int i=0; i<nFilters1D; i++) 
		copyFiltersToDev(filters1DCU[i], i);
	copyFiltGaussToDev((float*)filtGauss);
}

void Filters::mallocGPUphShiftCVrepOriRepPhase(int _imageW, int _imageH) {

	cosPhShift_devCVrepOriRepPhase.create(_imageH,nOrient*nPhase*_imageW,CV_32FC1);
	sinPhShift_devCVrepOriRepPhase.create(_imageH,nOrient*nPhase*_imageW,CV_32FC1);
	minusSinPhShift_devCVrepOriRepPhase.create(_imageH,nOrient*nPhase*_imageW,CV_32FC1);
	sumCosSinPhShift_devCVrepOriRepPhase.create(_imageH,nOrient*nPhase*_imageW,CV_32FC1);
	sumCosMinusSinPhShift_devCVrepOriRepPhase.create(_imageH,nOrient*nPhase*_imageW,CV_32FC1);
}
void Filters::mallocGPUphShiftCUrepOriRepPhase(int _imageW, int _imageH) {

	checkCudaErrors( cudaMalloc((void **)&cosPhShift_devCUrepOriRepPhase, _imageW * _imageH * nPhase * nOrient * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void **)&sinPhShift_devCUrepOriRepPhase, _imageW * _imageH * nPhase * nOrient * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void **)&minusSinPhShift_devCUrepOriRepPhase, _imageW * _imageH * nPhase * nOrient * sizeof(float)) );
}
void Filters::freeGPUphShiftCVrepOriRepPhase() {

	cosPhShift_devCVrepOriRepPhase.release();
	sinPhShift_devCVrepOriRepPhase.release();
	minusSinPhShift_devCVrepOriRepPhase.release();
	sumCosSinPhShift_devCVrepOriRepPhase.release();
	sumCosMinusSinPhShift_devCVrepOriRepPhase.release();
}
void Filters::freeGPUphShiftCUrepOriRepPhase() {

	checkCudaErrors( cudaFree(cosPhShift_devCUrepOriRepPhase) );
	checkCudaErrors( cudaFree(sinPhShift_devCUrepOriRepPhase) );
	checkCudaErrors( cudaFree(minusSinPhShift_devCUrepOriRepPhase) );
}
void Filters::uploadGPUphShiftCVrepOriRepPhase(int _imageW, int _imageH) {

	float ofs = 0.00001;

	sinPhShift_devCVrepOriRepPhase.setTo(0);
	cosPhShift_devCVrepOriRepPhase.setTo(0);
	for (int phase=0; phase<nPhase; phase++) {
			cv::gpu::add(sinPhShift_devCVrepOriRepPhase(cv::Rect(phase*nOrient*_imageW,0,_imageW*nOrient,_imageH)),sinPhShift[phase],sinPhShift_devCVrepOriRepPhase(cv::Rect(phase*nOrient*_imageW,0,_imageW*nOrient,_imageH)));
			cv::gpu::add(cosPhShift_devCVrepOriRepPhase(cv::Rect(phase*nOrient*_imageW,0,_imageW*nOrient,_imageH)),cosPhShift[phase],cosPhShift_devCVrepOriRepPhase(cv::Rect(phase*nOrient*_imageW,0,_imageW*nOrient,_imageH)));
		}
	cv::gpu::multiply(sinPhShift_devCVrepOriRepPhase,-1,minusSinPhShift_devCVrepOriRepPhase,1,-1);
	cv::gpu::add(cosPhShift_devCVrepOriRepPhase,sinPhShift_devCVrepOriRepPhase,sumCosSinPhShift_devCVrepOriRepPhase);
	cv::gpu::add(sumCosSinPhShift_devCVrepOriRepPhase,ofs,sumCosSinPhShift_devCVrepOriRepPhase);
	cv::gpu::add(cosPhShift_devCVrepOriRepPhase,minusSinPhShift_devCVrepOriRepPhase,sumCosMinusSinPhShift_devCVrepOriRepPhase);
	cv::gpu::add(sumCosMinusSinPhShift_devCVrepOriRepPhase,ofs,sumCosMinusSinPhShift_devCVrepOriRepPhase);
}
void Filters::uploadGPUphShiftCUrepOriRepPhase(int _imageW, int _imageH) {

	for (int phase=0; phase<nPhase; phase++) {
			nppsSet_32f(cosPhShift[phase],cosPhShift_devCUrepOriRepPhase + phase * (_imageW * _imageH * nOrient),_imageW * _imageH * nOrient);
			nppsSet_32f(sinPhShift[phase],sinPhShift_devCUrepOriRepPhase + phase * (_imageW * _imageH * nOrient),_imageW * _imageH * nOrient);
			nppsSet_32f(-sinPhShift[phase],minusSinPhShift_devCUrepOriRepPhase + phase * (_imageW * _imageH * nOrient),_imageW * _imageH * nOrient);
		}
}

void Filters::mallocGPUoriTuningCVoneOriSepPhase(int _imageW, int _imageH) {

	cosOriTuning_devCVoneOri.create(_imageH,nOrient*_imageW,CV_32FC1);
	sinOriTuning_devCVoneOri.create(_imageH,nOrient*_imageW,CV_32FC1);
}
void Filters::mallocGPUoriTuningCUoneOriSepPhase(int _imageW, int _imageH) {

	checkCudaErrors( cudaMalloc((void **)&cosOriTuning_devCUoneOri, _imageW * _imageH * nOrient * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void **)&sinOriTuning_devCUoneOri, _imageW * _imageH * nOrient * sizeof(float)) );
}
void Filters::freeGPUoriTuningCVoneOriSepPhase() {

	cosOriTuning_devCVoneOri.release();
	sinOriTuning_devCVoneOri.release();
}
void Filters::freeGPUoriTuningCUoneOriSepPhase() {

	checkCudaErrors( cudaFree(cosOriTuning_devCUoneOri) );
	checkCudaErrors( cudaFree(sinOriTuning_devCUoneOri) );
}
void Filters::uploadGPUoriTuningCVoneOriSepPhase(int _imageW, int _imageH) {

	cosOriTuning_devCVoneOri.setTo(0);
	sinOriTuning_devCVoneOri.setTo(0);

	for (int theta=0; theta<nOrient; theta++) {
		cv::gpu::add(cosOriTuning_devCVoneOri(cv::Rect(theta*_imageW,0,_imageW,_imageH)),cosOriTuning[theta],cosOriTuning_devCVoneOri(cv::Rect(theta*_imageW,0,_imageW,_imageH)));
		cv::gpu::add(sinOriTuning_devCVoneOri(cv::Rect(theta*_imageW,0,_imageW,_imageH)),sinOriTuning[theta],sinOriTuning_devCVoneOri(cv::Rect(theta*_imageW,0,_imageW,_imageH)));
	}
}
void Filters::uploadGPUoriTuningCUoneOriSepPhase(int _imageW, int _imageH) {

	for (int theta=0; theta<nOrient; theta++) {
		nppsSet_32f(cosOriTuning[theta],cosOriTuning_devCUoneOri + theta * _imageW * _imageH, _imageW * _imageH);
		nppsSet_32f(sinOriTuning[theta],sinOriTuning_devCUoneOri + theta * _imageW * _imageH, _imageW * _imageH);
	}
}

void Filters::mallocGPUoriTuningCVrepOriRepPhase(int _imageW, int _imageH) {
	
	cosOriTuning_devCVrepOriRepPhase.create(_imageH,nOrient*_imageW,CV_32FC1);
	sinOriTuning_devCVrepOriRepPhase.create(_imageH,nOrient*_imageW,CV_32FC1);
}
void Filters::mallocGPUoriTuningCUrepOriRepPhase(int _imageW, int _imageH) {

	checkCudaErrors( cudaMalloc((void **)&cosOriTuning_devCUrepOriRepPhase, _imageW * _imageH * nOrient * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void **)&sinOriTuning_devCUrepOriRepPhase, _imageW * _imageH * nOrient * sizeof(float)) );
}
void Filters::freeGPUoriTuningCVrepOriRepPhase() {

	cosOriTuning_devCVrepOriRepPhase.release();
	sinOriTuning_devCVrepOriRepPhase.release();
}
void Filters::freeGPUoriTuningCUrepOriRepPhase() {

	checkCudaErrors( cudaFree(cosOriTuning_devCUrepOriRepPhase) );
	checkCudaErrors( cudaFree(sinOriTuning_devCUrepOriRepPhase) );
}
void Filters::uploadGPUoriTuningCVrepOriRepPhase(int _imageW, int _imageH) {

	cosOriTuning_devCVrepOriRepPhase.setTo(0);
	sinOriTuning_devCVrepOriRepPhase.setTo(0);

	for (int theta=0; theta<nOrient; theta++) {
		cv::gpu::add(cosOriTuning_devCVrepOriRepPhase(cv::Rect(theta*_imageW,0,_imageW,_imageH)),cosOriTuning[theta],cosOriTuning_devCVrepOriRepPhase(cv::Rect(theta*_imageW,0,_imageW,_imageH)));
		cv::gpu::add(sinOriTuning_devCVrepOriRepPhase(cv::Rect(theta*_imageW,0,_imageW,_imageH)),sinOriTuning[theta],sinOriTuning_devCVrepOriRepPhase(cv::Rect(theta*_imageW,0,_imageW,_imageH)));
	}
}
void Filters::uploadGPUoriTuningCUrepOriRepPhase(int _imageW, int _imageH) {

	for (int theta=0; theta<nOrient; theta++) {
		nppsSet_32f(cosOriTuning[theta],cosOriTuning_devCUrepOriRepPhase + theta * _imageW * _imageH, _imageW * _imageH);
		nppsSet_32f(sinOriTuning[theta],sinOriTuning_devCUrepOriRepPhase + theta * _imageW * _imageH, _imageW * _imageH);
	}
}
