#ifndef CONV_COMMON_H
#define CONV_COMMON_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////

extern "C" void copyFiltGaussToDev(float *host_ptr);
extern "C" void copyFiltGaussToHost(float *host_ptr);

extern "C" void copyFiltersToDev(float *host_ptr, const int index);
extern "C" void copyFiltersToHost(float *host_ptr, const int index);

/*
extern "C" void copyDispToDev(float *host_ptr);
extern "C" void copyDispToHost(float *host_ptr);
extern "C" void copyPhShiftToDev(float *host_ptr);
extern "C" void copyPhShiftToHost(float *host_ptr);
extern "C" void copyOriTuningToDev(float *host_ptr);
extern "C" void copyOriTuningToHost(float *host_ptr);

extern "C" void copyCosPhShiftToDev(float *host_ptr);
extern "C" void copyCosPhShiftToHost(float *host_ptr);
extern "C" void copySinPhShiftToDev(float *host_ptr);
extern "C" void copySinPhShiftToHost(float *host_ptr);
extern "C" void copyCosOriTuningToDev(float *host_ptr);
extern "C" void copyCosOriTuningToHost(float *host_ptr);
extern "C" void copySinOriTuningToDev(float *host_ptr);
extern "C" void copySinOriTuningToHost(float *host_ptr);
*/

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int KERNEL_LENGTH,
	int kernelIdx,
	cudaStream_t stream = 0
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int KERNEL_LENGTH,
	int kernelIdx,
	cudaStream_t stream = 0
);

#endif
