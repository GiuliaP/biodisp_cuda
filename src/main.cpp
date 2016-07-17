// Project files includes
////////////////////////////////////////////////////////////////////////////////
#include "Population.h"
#include "Filters.h"
#include "quantities.h"

// Utilities and system includes
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>

#include <ctime>
#include <cstdio>
#include <time.h>
#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <string>

// CUDA includes
////////////////////////////////////////////////////////////////////////////////
#include <cuda_runtime.h>
#include <shrQATest.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// Global variables
////////////////////////////////////////////////////////////////////////////////

double disp[] = {-1.5,-1,-0.5,0,0.5,1,1.5};
float filtGauss[] = {0.0625, 0.2500, 0.3750, 0.2500, 0.0625};

// Program main 
/////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
    
// Device properties
/////////////////////////////////////////////////////////////////////////////////
	
	cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int cuda_device = 0;

    shrQAStart(argc, argv);

    // Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    cuda_device = findCudaDevice(argc, (const char **)argv);

	checkCudaErrors(cudaGetDevice(&cuda_device));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, cuda_device));

    printf("GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x20) {
        printf("%s: requires a minimum CUDA compute 2.0 capability\n\n");
        shrQAFinishExit(argc, (const char **)argv, QA_PASSED);
    }
	
	if ((deviceProp.concurrentKernels == 0)) {
        printf("GPU does not support concurrent kernel execution\n\n");
        printf("CUDA kernel runs will be serialized\n\n");
    }

// Logfile and files for the time measures
/////////////////////////////////////////////////////////////////////////////////
	
	char prompt[100];
	char timeUpload[100], timeDownload[100], timeAllFunc[100];

// Intances of Filters and Population classes
/////////////////////////////////////////////////////////////////////////////////

    Population cells;
	Filters filters("filters.txt", disp, filtGauss);

	filters.uploadCV();
	filters.uploadCU();

// Indexes that are necessary for the CUDA part 
/////////////////////////////////////////////////////////////////////////////////
	
	int nTempIdx = nOrient;
	int *tempIdx = (int*) malloc(nTempIdx * sizeof(int));
	for (int idx=0; idx<nTempIdx; idx++)
		tempIdx[idx] = idx;

	int nTempIdx_oneOri = 1;
	int *tempIdx_oneOri = (int*) malloc(nTempIdx_oneOri * sizeof(int));
	for (int idx=0; idx<nTempIdx_oneOri; idx++)
		tempIdx_oneOri[idx] = idx;

	int nTempIdx_repOriRepPhase = 4;
	int *tempIdx_repOriRepPhase = (int*) malloc(nTempIdx_repOriRepPhase * sizeof(int));
	for (int idx=0; idx<nTempIdx_repOriRepPhase; idx++)
		tempIdx_repOriRepPhase[idx] = idx;

// Function pointers
/////////////////////////////////////////////////////////////////////////////////

	std::string funcName[nFunc];
	void (Population::*func[nFunc])(Filters *,float *,bool);
	
// Timer
/////////////////////////////////////////////////////////////////////////////////

	StopWatchInterface *extTimer = NULL;
    sdkCreateTimer(&extTimer);
	int iterations = 50;
	bool bWarmUpIteration = true;
	bool bEventCV = false;


	bool bEventCU = true;
	bool bTimer = false;


// Images
/////////////////////////////////////////////////////////////////////////////////

	char ***allImNames_char, **imName_char;
	std::string *imName_str;
	int dims[] = {704};
	int nDims = 1;
	int dim;

	allImNames_char = (char***) malloc (nDims * sizeof(char**));
	for (int i=0; i<nDims; i++)
		allImNames_char[i] = (char**) malloc (nEye * sizeof(char*));
	imName_char = (char**) malloc (nEye * sizeof(char*));
	imName_str = new std::string[nEye];
	
	allImNames_char[0][0] = "TsukubaLeft704.png";
	allImNames_char[0][1] = "TsukubaRight704.png";

// For-cycle for all the image pairs listed
/////////////////////////////////////////////////////////////////////////////////

for (int i=0; i<nDims; i++) {
		
	imName_char[0] = allImNames_char[i][0]; 
	imName_char[1] = allImNames_char[i][1]; 
	imName_str[0] = imName_char[0];
	imName_str[1] = imName_char[1];
	dim = dims[i];
	
	// if uncomment this instruction, set '-noprompt' in the project properties (Configuration Properties, Debugging)
	// in this case the output is redirected to file
	// otherwise set '-prompt'
	sprintf(prompt,"results/timing/prompt%d_bEventCU%d_bTimer%d.txt",dim,bEventCU,bTimer);
	std::freopen (prompt,"w",stdout);

	printf("Allocating OPENCV resources...\n\n");

	cells.mallocCPUImagePairOPENCV(dim, dim);
	cells.mallocCPUResultOPENCV(dim, dim);
	cells.mallocGPUImagePairOPENCV(cells.imageH, cells.imageW);
    
	cells.mallocGPUStreamsOPENCV();
	cells.createStartStopEvents();
	cells.createFuncTimer();

	cells.mallocCPUCellOPENCV();
	
	cells.loadImagePairOPENCV(imName_str);

	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV NO STREAMS separate Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	cells.mallocGPUCellOPENCV();
	cells.mallocGPUTempMatricesOPENCV();
	
	funcName[0] = "calcSimpleAnswerOPENCV_sepOriSepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_sepOriSepPhase_noStreams";
	funcName[2] = "calcEnergyOPENCV_sepOriSepPhase_noStreams";
	funcName[3] = "calcCenterOfMassOPENCV_sepOriSepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_sepOriSepPhase_noStreams";
	funcName[5] = "solveApertureOPENCV_sepOriSepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_sepOriSepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_sepOriSepPhase_noStreams;
	func[2] = &Population::calcEnergyOPENCV_sepOriSepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_sepOriSepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_sepOriSepPhase_noStreams;
	func[5] = &Population::solveApertureOPENCV_sepOriSepPhase_noStreams;

	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCV%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCV%d_bEvent%d.txt",dim,bEventCV);

	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload, "downloadResultOPENCV", &Population::downloadResultOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCV();
	cells.freeGPUCellOPENCV();
	cells.freeGPUTempMatricesOPENCV();


	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV NO STREAMS united Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUoriTuningCVoneOriSepPhase(dim, dim);
	filters.uploadGPUoriTuningCVoneOriSepPhase(dim, dim);
	cells.mallocGPUCellOPENCVoneOri();
	cells.mallocGPUTempMatricesOPENCVoneOri();
	
	funcName[0] = "calcSimpleAnswerOPENCV_oneOriSepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_oneOriSepPhase_noStreams";
	funcName[2] = "calcEnergyOPENCV_oneOriSepPhase_noStreams";
	funcName[3] = "calcCenterOfMassOPENCV_oneOriSepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_oneOriSepPhase_noStreams";
	funcName[5] = "solveApertureOPENCV_oneOriSepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_oneOriSepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_oneOriSepPhase_noStreams;
	func[2] = &Population::calcEnergyOPENCV_oneOriSepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_oneOriSepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_oneOriSepPhase_noStreams;
	func[5] = &Population::solveApertureOPENCV_oneOriSepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCVoneOri%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCVoneOri%d_bEvent%d.txt",dim,bEventCV);
	
	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration,extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload,"downloadResultOPENCVoneOri", &Population::downloadResultOPENCVoneOri,iterations,bWarmUpIteration,extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCVoneOri();
	filters.freeGPUoriTuningCVoneOriSepPhase();
	cells.freeGPUCellOPENCVoneOri();
	cells.freeGPUTempMatricesOPENCVoneOri();

/*
	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV NO STREAMS repeated Ori repeated Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.uploadGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.mallocGPUoriTuningCVrepOriRepPhase(dim, dim);
	filters.uploadGPUoriTuningCVrepOriRepPhase(dim, dim);
	cells.mallocGPUCellOPENCVrepOriRepPhase();
	cells.mallocGPUTempMatricesOPENCVrepOriRepPhase();
	
	funcName[0] = "calcSimpleAnswerOPENCVrepOriRepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_repOriRepPhase_noStreams";
	funcName[2] = "calcEnergyOPENCV_repOriRepPhase_noStreams";
	funcName[3] = "calcCenterOfMassOPENCV_repOriRepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams";
	funcName[5] = "solveApertureOPENCV_repOriRepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_repOriRepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_repOriRepPhase_noStreams;
	func[2] = &Population::calcEnergyOPENCV_repOriRepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_repOriRepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams;
	func[5] = &Population::solveApertureOPENCV_repOriRepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCVrepOriRepPhase%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCVrepOriRepPhase%d_bEvent%d.txt",dim,bEventCV);
	
	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload,"downloadResultOPENCVrepOriRepPhase", &Population::downloadResultOPENCVrepOriRepPhase,iterations,bWarmUpIteration,extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCVrepOriRepPhase();
	filters.freeGPUphShiftCVrepOriRepPhase();
	filters.freeGPUoriTuningCVrepOriRepPhase();
	cells.freeGPUCellOPENCVrepOriRepPhase();
	cells.freeGPUTempMatricesOPENCVrepOriRepPhase();
*/
/*
	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV NO STREAMS repeated Ori repeated Phase - using BlendLinear:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.uploadGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.mallocGPUoriTuningCVrepOriRepPhase(dim, dim);
	filters.uploadGPUoriTuningCVrepOriRepPhase(dim, dim);
	cells.mallocGPUCellOPENCVrepOriRepPhase();
	cells.mallocGPUTempMatricesOPENCVrepOriRepPhase();
	
	funcName[0] = "calcSimpleAnswerOPENCVrepOriRepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_repOriRepPhaseBlendLinear_noStreams";
	funcName[2] = "calcEnergyOPENCV_repOriRepPhase_noStreams";
	funcName[3] = "calcCenterOfMassOPENCV_repOriRepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams";
	funcName[5] = "solveApertureOPENCV_repOriRepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_repOriRepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_repOriRepPhaseBlendLinear_noStreams;
	func[2] = &Population::calcEnergyOPENCV_repOriRepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_repOriRepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_repOriRepPhase_noStreams;
	func[5] = &Population::solveApertureOPENCV_repOriRepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCVrepOriRepPhaseBlendLinear%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCVrepOriRepPhase%d_bEvent%d.txt",dim,bEventCV);
	
	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload,"downloadResultOPENCVrepOriRepPhase", &Population::downloadResultOPENCVrepOriRepPhase,iterations,bWarmUpIteration,extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCVrepOriRepPhase();
	filters.freeGPUphShiftCVrepOriRepPhase();
	filters.freeGPUoriTuningCVrepOriRepPhase();
	cells.freeGPUCellOPENCVrepOriRepPhase();
	cells.freeGPUTempMatricesOPENCVrepOriRepPhase();
*/

	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV WITH STREAMS separate Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	cells.mallocGPUCellOPENCV();
	cells.mallocGPUTempMatricesOPENCV();
	
	funcName[0] = "calcSimpleAnswerOPENCV_sepOriSepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_sepOriSepPhase_withStreams";
	funcName[2] = "calcEnergyOPENCV_sepOriSepPhase_withStreams";
	funcName[3] = "calcCenterOfMassOPENCV_sepOriSepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_sepOriSepPhase_withStreams";
	funcName[5] = "solveApertureOPENCV_sepOriSepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_sepOriSepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_sepOriSepPhase_withStreams;
	func[2] = &Population::calcEnergyOPENCV_sepOriSepPhase_withStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_sepOriSepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_sepOriSepPhase_withStreams;
	func[5] = &Population::solveApertureOPENCV_sepOriSepPhase_withStreams;

	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCV%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCV%d_bEvent%d.txt",dim,bEventCV);

	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload, "downloadResultOPENCV", &Population::downloadResultOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCV();
	cells.freeGPUCellOPENCV();
	cells.freeGPUTempMatricesOPENCV();


	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV WITH STREAMS united Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUoriTuningCVoneOriSepPhase(dim, dim);
	filters.uploadGPUoriTuningCVoneOriSepPhase(dim, dim);
	cells.mallocGPUCellOPENCVoneOri();
	cells.mallocGPUTempMatricesOPENCVoneOri();
	
	funcName[0] = "calcSimpleAnswerOPENCV_oneOriSepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_oneOriSepPhase_withStreams";
	funcName[2] = "calcEnergyOPENCV_oneOriSepPhase_withStreams";
	funcName[3] = "calcCenterOfMassOPENCV_oneOriSepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_oneOriSepPhase_withStreams";
	funcName[5] = "solveApertureOPENCV_oneOriSepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_oneOriSepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_oneOriSepPhase_withStreams;
	func[2] = &Population::calcEnergyOPENCV_oneOriSepPhase_withStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_oneOriSepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_oneOriSepPhase_withStreams;
	func[5] = &Population::solveApertureOPENCV_oneOriSepPhase_withStreams;
	
	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCVoneOri%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCVoneOri%d_bEvent%d.txt",dim,bEventCV);
	
	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration,extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload,"downloadResultOPENCVoneOri", &Population::downloadResultOPENCVoneOri,iterations,bWarmUpIteration,extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCVoneOri();
	filters.freeGPUoriTuningCVoneOriSepPhase();
	cells.freeGPUCellOPENCVoneOri();
	cells.freeGPUTempMatricesOPENCVoneOri();

/*
	printf("////////////////////////////////////////////////////////////////\n");
	printf("OPENCV WITH STREAMS repeated Ori repeated Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.uploadGPUphShiftCVrepOriRepPhase(dim, dim);
	filters.mallocGPUoriTuningCVrepOriRepPhase(dim, dim);
	filters.uploadGPUoriTuningCVrepOriRepPhase(dim, dim);
	cells.mallocGPUCellOPENCVrepOriRepPhase();
	cells.mallocGPUTempMatricesOPENCVrepOriRepPhase();
	
	funcName[0] = "calcSimpleAnswerOPENCV_repOriRepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerOPENCV_repOriRepPhase_withStreams";
	funcName[2] = "calcEnergyOPENCV_repOriRepPhase_noStreams";
	funcName[3] = "calcCenterOfMassOPENCV_repOriRepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassOPENCV_repOriRepPhase_withStreams";
	funcName[5] = "solveApertureOPENCV_repOriRepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerOPENCV_repOriRepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerOPENCV_repOriRepPhase_withStreams;
	func[2] = &Population::calcEnergyOPENCV_repOriRepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassOPENCV_repOriRepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassOPENCV_repOriRepPhase_withStreams;
	func[5] = &Population::solveApertureOPENCV_repOriRepPhase_withStreams;
	
	sprintf(timeUpload,"results/timing/uploadCV%d_bEvent%d.txt",dim,bEventCV);
	sprintf(timeAllFunc,"results/timing/allFuncCVrepOriRepPhase%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCV,bTimer);
	sprintf(timeDownload,"results/timing/downloadCVrepOriRepPhase%d_bEvent%d.txt",dim,bEventCV);
	
	cells.call(timeUpload, "uploadImagePairOPENCV", &Population::uploadImagePairOPENCV,iterations,bWarmUpIteration, extTimer,bEventCV,cells.src_hostCV);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCV, bTimer, &filters);
	cells.call(timeDownload,"downloadResultOPENCVrepOriRepPhase", &Population::downloadResultOPENCVrepOriRepPhase,iterations,bWarmUpIteration,extTimer,bEventCV,cells.dst_hostCV);
	
	//cells.downloadSimpleAnswerOPENCVrepOriRepPhase();
	filters.freeGPUphShiftCVrepOriRepPhase();
	filters.freeGPUoriTuningCVrepOriRepPhase();
	cells.freeGPUCellOPENCVrepOriRepPhase();
	cells.freeGPUTempMatricesOPENCVrepOriRepPhase();
*/

	printf("Deallocating OPENCV resources...\n\n");

	cells.freeGPUImagePairOPENCV();
	cells.freeCPUImagePairOPENCV();

	cells.freeGPUStreamsOPENCV();
	cells.destroyStartStopEvents();
	cells.destroyFuncTimer();

	/*
	printf("Printing simple answers OPENCV...\n\n");

	char filenameCVRe[100], filenameCVIm[100];
	int eyeCV;

	eyeCV=0; 
		for (int theta=0; theta<nOrient; theta++) {
			sprintf(filenameCVRe, "results/simpleRe_%d_%d_%d_CV.txt", eyeCV, theta, centralPhaseIdx);
			sprintf(filenameCVIm, "results/simpleIm_%d_%d_%d_CV.txt", eyeCV, theta, centralPhaseIdx);
			cells.printFileSimpleAnswerOPENCV(filenameCVRe, filenameCVIm, eyeCV, theta, 0);
		}
	eyeCV=1;
		for (int theta=0; theta<nOrient; theta++) 
			for (int phase=0; phase<nPhase; phase++) {
				sprintf(filenameCVRe, "results/simpleRe_%d_%d_%d_CV.txt", eyeCV, theta, phase);
				sprintf(filenameCVIm, "results/simpleIm_%d_%d_%d_CV.txt", eyeCV, theta, phase);
				cells.printFileSimpleAnswerOPENCV(filenameCVRe, filenameCVIm, eyeCV, theta, phase);
			}
	*/
	/*
	printf("Printing energy OPENCV...\n\n");

	char filenameCVen[100];

	for (int theta=0; theta<nOrient; theta++) 
		for (int phase=0; phase<nPhase; phase++) {
			sprintf(filenameCVen, "results/energy_%d_%d_CV.txt", theta, phase);
			cells.printFileEnergyOPENCV(filenameCVen, theta, phase);
		}
	*/
	/*
	printf("Printing centers of mass OPENCV...\n\n");
	
	char filenameCVcom[100];

	for (int theta=0; theta<nOrient; theta++) {
		sprintf(filenameCVcom, "results/com_%d_CV.txt", theta);
		cells.printFileCenterOfMassOPENCV(filenameCVcom, theta);
	}
	*/
	/*
	printf("Printing X and Y components of centers of mass OPENCV...\n\n");
	
	char filenameCVcomX[100], filenameCVcomY[100];

	for (int theta=0; theta<nOrient; theta++) {
		sprintf(filenameCVcomX, "results/comX_%d_CV.txt", theta);
		sprintf(filenameCVcomY, "results/comY_%d_CV.txt", theta);
		cells.printFileXYCenterOfMassOPENCV(filenameCVcomX, filenameCVcomY, theta);
	}
	*/
	/*
	printf("Printing X and Y components of disparities OPENCV...\n\n");
	
	char filenameCVdispX[100], filenameCVdispY[100];

	sprintf(filenameCVdispX, "results/dispX_CV.txt");
	sprintf(filenameCVdispY, "results/dispY_CV.txt");
	cells.printFileDisparityOPENCV(filenameCVdispX, filenameCVdispY);
	*/

	printf("Allocating CUDA resources...\n\n");
	
	cells.mallocCPUImagePairOPENCV(dim,dim);
	cells.mallocCPUImagePairCUDA(dim,dim);
	cells.mallocCPUResultCUDA(dim, dim);
	cells.mallocGPUImagePairCUDA(cells.imageH, cells.imageH);
   
	cells.mallocGPUStreamsCUDA();
	cells.createStartStopEvents();
	cells.createFuncTimer();

	cells.mallocCPUCellCUDA();

	//cells.loadImagePairCUDA(imName_char, argv[0]);
	cells.loadImagePairOPENCV(imName_str);
	cells.copyCPUImagePairToCUDA(cells.src_hostCV);

	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA NO STREAMS separate Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	cells.mallocGPUCellCUDA();
	cells.mallocGPUTempMatricesCUDA();
	cells.set0GPUTempMatricesCUDA(tempIdx, nTempIdx); // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
   
	funcName[0] = "calcSimpleAnswerCUDA_sepOriSepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_sepOriSepPhase_noStreams";
	funcName[2] = "calcEnergyCUDA_sepOriSepPhase_noStreams";
	funcName[3] = "calcCenterOfMassCUDA_sepOriSepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_sepOriSepPhase_noStreams";
	funcName[5] = "solveApertureCUDA_sepOriSepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_sepOriSepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_sepOriSepPhase_noStreams;
	func[2] = &Population::calcEnergyCUDA_sepOriSepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassCUDA_sepOriSepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_sepOriSepPhase_noStreams;
	func[5] = &Population::solveApertureCUDA_sepOriSepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCU%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCU%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload,"downloadResultCUDA", &Population::downloadResultCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.dst_hostCU);
	
	//cells.downloadSimpleAnswerCUDA();
	cells.freeGPUCellCUDA();
	cells.freeGPUTempMatricesCUDA();


	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA NO STREAMS united Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUoriTuningCUoneOriSepPhase(dim, dim);
	filters.uploadGPUoriTuningCUoneOriSepPhase(dim, dim);
	cells.mallocGPUCellCUDAoneOri();
	cells.mallocGPUTempMatricesCUDAoneOri();
	cells.set0GPUTempMatricesCUDAoneOri(tempIdx_oneOri, nTempIdx_oneOri);  // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
   	
	funcName[0] = "calcSimpleAnswerCUDA_oneOriSepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_oneOriSepPhase_noStreams";
	funcName[2] = "calcEnergyCUDA_oneOriSepPhase_noStreams";
	funcName[3] = "calcCenterOfMassCUDA_oneOriSepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_oneOriSepPhase_noStreams";
	funcName[5] = "solveApertureCUDA_oneOriSepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_oneOriSepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_oneOriSepPhase_noStreams;
	func[2] = &Population::calcEnergyCUDA_oneOriSepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassCUDA_oneOriSepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_oneOriSepPhase_noStreams;
	func[5] = &Population::solveApertureCUDA_oneOriSepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCUoneOri%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCUoneOri%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload,"downloadResultCUDAoneOri", &Population::downloadResultCUDAoneOri,iterations,bWarmUpIteration,extTimer,bEventCU,cells.dst_hostCU);
	
	//cells.downloadSimpleAnswerCUDAoneOri();
	filters.freeGPUoriTuningCUoneOriSepPhase();
	cells.freeGPUCellCUDAoneOri();
	cells.freeGPUTempMatricesCUDAoneOri();

/*
	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA NO STREAMS repeated Ori repeated Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUphShiftCUrepOriRepPhase(dim, dim);
	filters.uploadGPUphShiftCUrepOriRepPhase(dim, dim);
	filters.mallocGPUoriTuningCUrepOriRepPhase(dim, dim);
	filters.uploadGPUoriTuningCUrepOriRepPhase(dim, dim);
	cells.mallocGPUCellCUDArepOriRepPhase();
	cells.mallocGPUTempMatricesCUDArepOriRepPhase();
	cells.set0GPUTempMatricesCUDArepOriRepPhase(tempIdx_repOriRepPhase, nTempIdx_repOriRepPhase); // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
   
	funcName[0] = "calcSimpleAnswerCUDA_repOriRepPhase_noStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_repOriRepPhase_noStreams";
	funcName[2] = "calcEnergyCUDA_repOriRepPhase_noStreams";
	funcName[3] = "calcCenterOfMassCUDA_repOriRepPhase_noStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_repOriRepPhase_noStreams";
	funcName[5] = "solveApertureCUDA_repOriRepPhase_noStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_repOriRepPhase_noStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_repOriRepPhase_noStreams;
	func[2] = &Population::calcEnergyCUDA_repOriRepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassCUDA_repOriRepPhase_noStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_repOriRepPhase_noStreams;
	func[5] = &Population::solveApertureCUDA_repOriRepPhase_noStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCUrepOriRepPhase%d_noStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCUrepOriRepPhase%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload, "downloadResultCUDArepOriRepPhase", &Population::downloadResultCUDArepOriRepPhase,iterations,bWarmUpIteration, extTimer,bEventCU,cells.dst_hostCU);

	//cells.downloadSimpleAnswerCUDArepOriRepPhase();
	filters.freeGPUphShiftCUrepOriRepPhase();
	filters.freeGPUoriTuningCUrepOriRepPhase();
	cells.freeGPUCellCUDArepOriRepPhase();
	cells.freeGPUTempMatricesCUDArepOriRepPhase();
*/

	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA WITH STREAMS separate Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	cells.mallocGPUCellCUDA();
	cells.mallocGPUTempMatricesCUDA();
	cells.set0GPUTempMatricesCUDA(tempIdx, nTempIdx); // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
	
	funcName[0] = "calcSimpleAnswerCUDA_sepOriSepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_sepOriSepPhase_withStreams";
	funcName[2] = "calcEnergyCUDA_sepOriSepPhase_withStreams";
	funcName[3] = "calcCenterOfMassCUDA_sepOriSepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_sepOriSepPhase_withStreams";
	funcName[5] = "solveApertureCUDA_sepOriSepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_sepOriSepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_sepOriSepPhase_withStreams;
	func[2] = &Population::calcEnergyCUDA_sepOriSepPhase_withStreams;
	func[3] = &Population::calcCenterOfMassCUDA_sepOriSepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_sepOriSepPhase_withStreams;
	func[5] = &Population::solveApertureCUDA_sepOriSepPhase_withStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCU%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCU%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload,"downloadResultCUDA", &Population::downloadResultCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.dst_hostCU);
	
	//cells.downloadSimpleAnswerCUDA();
	cells.freeGPUCellCUDA();
	cells.freeGPUTempMatricesCUDA();


	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA WITH STREAMS united Ori separate Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUoriTuningCUoneOriSepPhase(dim, dim);
	filters.uploadGPUoriTuningCUoneOriSepPhase(dim, dim);
	cells.mallocGPUCellCUDAoneOri();
	cells.mallocGPUTempMatricesCUDAoneOri();
	cells.set0GPUTempMatricesCUDAoneOri(tempIdx_oneOri, nTempIdx_oneOri); // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
	
	funcName[0] = "calcSimpleAnswerCUDA_oneOriSepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_oneOriSepPhase_withStreams";
	funcName[2] = "calcEnergyCUDA_oneOriSepPhase_withStreams";
	funcName[3] = "calcCenterOfMassCUDA_oneOriSepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_oneOriSepPhase_withStreams";
	funcName[5] = "solveApertureCUDA_oneOriSepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_oneOriSepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_oneOriSepPhase_withStreams;
	func[2] = &Population::calcEnergyCUDA_oneOriSepPhase_withStreams;
	func[3] = &Population::calcCenterOfMassCUDA_oneOriSepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_oneOriSepPhase_withStreams;
	func[5] = &Population::solveApertureCUDA_oneOriSepPhase_withStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCUoneOri%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCUoneOri%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload,"downloadResultCUDAoneOri", &Population::downloadResultCUDAoneOri,iterations,bWarmUpIteration,extTimer,bEventCU,cells.dst_hostCU);
	
	//cells.downloadSimpleAnswerCUDAoneOri();
	filters.freeGPUoriTuningCUoneOriSepPhase();
	cells.freeGPUCellCUDAoneOri();
	cells.freeGPUTempMatricesCUDAoneOri();

/*
	printf("////////////////////////////////////////////////////////////////\n");
	printf("CUDA WITH STREAMS repeated Ori repeated Phase:\n");
	printf("////////////////////////////////////////////////////////////////\n\n");
	filters.mallocGPUphShiftCUrepOriRepPhase(dim, dim);
	filters.uploadGPUphShiftCUrepOriRepPhase(dim, dim);
	filters.mallocGPUoriTuningCUrepOriRepPhase(dim, dim);
	filters.uploadGPUoriTuningCUrepOriRepPhase(dim, dim);
	cells.mallocGPUCellCUDArepOriRepPhase();
	cells.mallocGPUTempMatricesCUDArepOriRepPhase();
	cells.set0GPUTempMatricesCUDArepOriRepPhase(tempIdx_repOriRepPhase, nTempIdx_repOriRepPhase); // deve restare l'azzeramento perche' nppsMaxEvery_32f_I esiste solo in-place (per ora)
	
	funcName[0] = "calcSimpleAnswerCUDA_repOriRepPhase_withStreams";
	funcName[1] = "shiftSimpleAnswerCUDA_repOriRepPhase_withStreams";
	funcName[2] = "calcEnergyCUDA_repOriRepPhase_withStreams";
	funcName[3] = "calcCenterOfMassCUDA_repOriRepPhase_withStreams";
	funcName[4] = "projectXYCenterOfMassCUDA_repOriRepPhase_withStreams";
	funcName[5] = "solveApertureCUDA_repOriRepPhase_withStreams";
	func[0] = &Population::calcSimpleAnswerCUDA_repOriRepPhase_withStreams;
	func[1] = &Population::shiftSimpleAnswerCUDA_repOriRepPhase_withStreams;
	func[2] = &Population::calcEnergyCUDA_repOriRepPhase_noStreams;
	func[3] = &Population::calcCenterOfMassCUDA_repOriRepPhase_withStreams;
	func[4] = &Population::projectXYCenterOfMassCUDA_repOriRepPhase_withStreams;
	func[5] = &Population::solveApertureCUDA_repOriRepPhase_withStreams;
	
	sprintf(timeUpload,"results/timing/uploadCU%d_bEvent%d.txt",dim,bEventCU);
	sprintf(timeAllFunc,"results/timing/allFuncCUrepOriRepPhase%d_withStreams_bEvent%d_bTimer%d.txt",dim,bEventCU,bTimer);
	sprintf(timeDownload,"results/timing/downloadCUrepOriRepPhase%d_bEvent%d.txt",dim,bEventCU);
	
	cells.call(timeUpload,"uploadImagePairCUDA", &Population::uploadImagePairCUDA,iterations,bWarmUpIteration, extTimer,bEventCU,cells.src_hostCU,dim,dim);
	cells.callAll(timeAllFunc,funcName,func,nFunc,iterations, bWarmUpIteration, extTimer, bEventCU, bTimer, &filters);
	cells.call(timeDownload, "downloadResultCUDArepOriRepPhase", &Population::downloadResultCUDArepOriRepPhase,iterations,bWarmUpIteration, extTimer,bEventCU,cells.dst_hostCU);
	
	//cells.downloadSimpleAnswerCUDArepOriRepPhase();
	filters.freeGPUphShiftCUrepOriRepPhase();
	filters.freeGPUoriTuningCUrepOriRepPhase();
	cells.freeGPUCellCUDArepOriRepPhase();
	cells.freeGPUTempMatricesCUDArepOriRepPhase();
*/
	
	printf("Deallocating CUDA resources...\n\n");

	cells.freeGPUImagePairCUDA();
	cells.freeCPUImagePairCUDA();
	cells.freeCPUImagePairOPENCV();

	cells.freeGPUStreamsCUDA();
	cells.destroyStartStopEvents();
	cells.destroyFuncTimer();

	/*
	printf("Printing simple answers CUDA...\n\n");

	char filenameCURe[100], filenameCUIm[100];
	int eyeCU;

	eyeCU=0; 
		for (int theta=0; theta<nOrient; theta++) {
			sprintf(filenameCURe, "results/simpleRe_%d_%d_%d_CU.txt", eyeCU, theta, centralPhaseIdx);
			sprintf(filenameCUIm, "results/simpleIm_%d_%d_%d_CU.txt", eyeCU, theta, centralPhaseIdx);
			cells.printFileSimpleAnswerCUDA(filenameCURe, filenameCUIm, eyeCU, theta, 0);
		}
	eyeCU=1;
		for (int theta=0; theta<nOrient; theta++) 
			for (int phase=0; phase<nPhase; phase++) {
				sprintf(filenameCURe, "results/simpleRe_%d_%d_%d_CU.txt", eyeCU, theta, phase);
				sprintf(filenameCUIm, "results/simpleIm_%d_%d_%d_CU.txt", eyeCU, theta, phase);
				cells.printFileSimpleAnswerCUDA(filenameCURe, filenameCUIm, eyeCU, theta, phase);
			}
	*/
	/*
	printf("Printing energy CUDA...\n\n");

	char filenameCUen[100];

	for (int theta=0; theta<nOrient; theta++) 
		for (int phase=0; phase<nPhase; phase++) {
			sprintf(filenameCUen, "results/energy_%d_%d_CU.txt", theta, phase);
			cells.printFileEnergyCUDA(filenameCUen, theta, phase);
		}
	*/
	/*
	printf("Printing centers of mass CUDA...\n\n");

	char filenameCUcom[100];

	for (int theta=0; theta<nOrient; theta++) {
		sprintf(filenameCUcom, "results/com_%d_CU.txt", theta);
		cells.printFileCenterOfMassCUDA(filenameCUcom, theta);
	}
	*/
	/*
	printf("Printing X and Y components of centers of mass CUDA...\n\n");

	char filenameCUcomX[100], filenameCUcomY[100];

	for (int theta=0; theta<nOrient; theta++) {
		sprintf(filenameCUcomX, "results/comX_%d_CU.txt", theta);
		sprintf(filenameCUcomY, "results/comY_%d_CU.txt", theta);
		cells.printFileXYCenterOfMassCUDA(filenameCUcomX, filenameCUcomY, theta);
	}
	*/
	/*
	printf("Printing X and Y components of disparities CUDA...\n\n");
	
	char filenameCUdispX[100], filenameCUdispY[100];

	sprintf(filenameCUdispX, "results/dispX_CU.txt");
	sprintf(filenameCUdispY, "results/dispY_CU.txt");
	cells.printFileDisparityCUDA(filenameCUdispX, filenameCUdispY);
	*/
	/*
	printf("Comparing simple answers CUDA/OPENCV...\n\n");

    float **diffSimpleAnswer = (float**) malloc(nEye * sizeof(float*));
	for (int eye = 0; eye < nEye ; eye++)
	    diffSimpleAnswer[eye] = (float*) malloc(cells.imageW * cells.imageH * sizeof(float));
	double L2normOutputSimpleAnswer;

	L2normOutputSimpleAnswer = cells.compareSimpleAnswer(diffSimpleAnswer);
	printf("Relative L2normOutputSimpleAnswer: %f\n\n", L2normOutputSimpleAnswer);
   
	for (int eye = 0; eye < nEye ; eye++)
		free(diffSimpleAnswer[eye]);
	free(diffSimpleAnswer);
	*/
	/*
	printf("Comparing energy CUDA/OPENCV...\n\n");

	float *diffEnergy = new float[cells.imageW * cells.imageH];
	double L2normOutputEnergy;

	L2normOutputEnergy = cells.compareEnergy(diffEnergy);
	printf("Relative L2normOutputEnergy: %f\n\n", L2normOutputEnergy);

	delete [] diffEnergy;
	*/
	/*
	printf("Comparing centers of mass CUDA/OPENCV...\n\n");

	float *diffCenterOfMass = new float[cells.imageW * cells.imageH];
	double L2normOutputCenterOfMass;

	L2normOutputCenterOfMass = cells.compareCenterOfMass(diffCenterOfMass);
	printf("Relative L2normOutputCenterOfMass: %f\n\n", L2normOutputCenterOfMass);

	delete [] diffCenterOfMass;
	*/
	/*
	printf("Comparing X and Y components of centers of mass CUDA/OPENCV...\n\n");

	float *diffXYCenterOfMass = new float[cells.imageW * cells.imageH];
	double L2normOutputXYCenterOfMass;

	L2normOutputXYCenterOfMass = cells.compareXYCenterOfMass(diffXYCenterOfMass);
	printf("Relative L2normOutputXYCenterOfMass: %f\n\n", L2normOutputXYCenterOfMass);

	delete [] diffXYCenterOfMass;
	*/
	
	printf("Comparing X and Y components of disparities CUDA/OPENCV...\n\n");

	float *diffDisparity = new float[cells.imageW * cells.imageH];
	double L2normOutputDisparity;

	L2normOutputDisparity = cells.compareDisparity(diffDisparity);
	printf("Relative L2normOutputDisparity: %f\n\n", L2normOutputDisparity);

	delete [] diffDisparity;
	

	printf("Deallocating other resources...\n\n");
    	
	cells.freeCPUCellOPENCV();
	cells.freeCPUResultOPENCV();
	cells.freeCPUCellCUDA();
	cells.freeCPUResultCUDA();

	fclose(stdout);

	}
	
	filters.freeCV();
	filters.freeCU();

	sdkDeleteTimer(&extTimer);
	
	free(tempIdx);
	free(tempIdx_oneOri);
	free(tempIdx_repOriRepPhase);
	
	free(allImNames_char);
	free(imName_char);
	delete [] imName_str;

	cudaDeviceReset();
	
	shrQAFinishExit(argc, (const char **)argv, QA_PASSED); 
	
	return 0;
}
