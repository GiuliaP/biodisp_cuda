#include <assert.h>
//#include <helper_cuda.h>
//#include <cuda_runtime.h>
#include "conv_common.h"
#include "quantities.h"

__constant__ float dev_filters1D[nFilters1D*taps+filtGaussLength];

/*
__constant__ float dev_Disp[nPhase];
__constant__ float dev_phShift[nPhase];
__constant__ float dev_oriTuning[nOrient];

__constant__ float dev_cosPhShift[nPhase];
__constant__ float dev_sinPhShift[nPhase];
__constant__ float dev_cosOriTuning[nOrient];
__constant__ float dev_sinOriTuning[nOrient];
*/

extern "C" void copyFiltGaussToDev(float *host_ptr){
	    cudaError_t err = cudaMemcpyToSymbol(dev_filters1D, host_ptr, filtGaussLength * sizeof(float), nFilters1D*taps*sizeof(float));
}
extern "C" void copyFiltGaussToHost(float *host_ptr){
        cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_filters1D, filtGaussLength * sizeof(float), nFilters1D*taps*sizeof(float));
}
extern "C" void copyFiltersToDev(float *host_ptr, const int index){
		cudaError_t err = cudaMemcpyToSymbol(dev_filters1D, host_ptr, taps * sizeof(float), index*taps*sizeof(float));
}
extern "C" void copyFiltersToHost(float *host_ptr, const int index){
		cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_filters1D, taps * sizeof(float), index*taps*sizeof(float));
}

/*
extern "C" void copyPhShiftToDev(float *host_ptr){
		cudaError_t err = cudaMemcpyToSymbol(dev_phShift, host_ptr, nPhase * sizeof(float));
}
extern "C" void copyPhShiftToHost(float *host_ptr) {
		cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_phShift, nPhase * sizeof(float));
}
extern "C" void copyDispToDev(float *host_ptr){
		cudaError_t err = cudaMemcpyToSymbol(dev_Disp, host_ptr, nPhase * sizeof(float));
}
extern "C" void copyDispToHost(float *host_ptr) {
		cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_Disp, nPhase * sizeof(float));
}
extern "C" void copyOriTuningToDev(float *host_ptr) {
	cudaError_t err = cudaMemcpyToSymbol(dev_oriTuning, host_ptr, nOrient * sizeof(float));
}
extern "C" void copyOriTuningToHost(float *host_ptr){
	cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_oriTuning, nOrient * sizeof(float));
}

extern "C" void copyCosPhShiftToDev(float *host_ptr){
	cudaError_t err = cudaMemcpyToSymbol(dev_cosPhShift, host_ptr, nPhase * sizeof(float));
}
extern "C" void copyCosPhShiftToHost(float *host_ptr){
	cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_cosPhShift, nPhase * sizeof(float));
}
extern "C" void copySinPhShiftToDev(float *host_ptr){
	cudaError_t err = cudaMemcpyToSymbol(dev_sinPhShift, host_ptr, nPhase * sizeof(float));
}
extern "C" void copySinPhShiftToHost(float *host_ptr){
	cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_sinPhShift, nPhase * sizeof(float));
}
extern "C" void copyCosOriTuningToDev(float *host_ptr){
	cudaError_t err = cudaMemcpyToSymbol(dev_cosOriTuning, host_ptr, nOrient * sizeof(float));
}
extern "C" void copyCosOriTuningToHost(float *host_ptr){
	cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_cosOriTuning, nOrient * sizeof(float));
}
extern "C" void copySinOriTuningToDev(float *host_ptr){
	cudaError_t err = cudaMemcpyToSymbol(dev_sinOriTuning, host_ptr, nOrient * sizeof(float));
}
extern "C" void copySinOriTuningToHost(float *host_ptr){
	cudaError_t err = cudaMemcpyFromSymbol(host_ptr, dev_sinOriTuning, nOrient * sizeof(float));
}
*/
////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////

#define   ROWS_BLOCKDIM_X 32
#define   ROWS_BLOCKDIM_Y 4
#define   ROWS_HALO_STEPS 1
#define   COLUMNS_BLOCKDIM_X 32
#define   COLUMNS_BLOCKDIM_Y 8
#define   COLUMNS_HALO_STEPS 1

#define   ROWS_RESULT_STEPS 11
#define COLUMNS_RESULT_STEPS 8

__global__ void convolutionRowsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch,
	int KERNEL_LENGTH,
	const int kernelIdx
){
    int KERNEL_RADIUS = (KERNEL_LENGTH -1)*0.5f;
		
	__shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

    //Load left halo
    #pragma unroll
    for(int i = 0; i < ROWS_HALO_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Load right halo
    #pragma unroll
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        #pragma unroll
        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += dev_filters1D[taps*kernelIdx+KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int KERNEL_LENGTH,
	int kernelIdx,
	cudaStream_t stream
){
    int KERNEL_RADIUS = (KERNEL_LENGTH -1)*0.5f;
	assert( ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS );
    assert( imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0 );
    assert( imageH % ROWS_BLOCKDIM_Y == 0 );

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

   convolutionRowsKernel<<<blocks, threads, 0, stream>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW,
		KERNEL_LENGTH,
		kernelIdx
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////

__global__ void convolutionColumnsKernel(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int pitch,
	int KERNEL_LENGTH,
	const int kernelIdx
){
     int KERNEL_RADIUS = (KERNEL_LENGTH -1)*0.5f;
	 
	 __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Upper halo
    #pragma unroll
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Lower halo
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += dev_filters1D[taps*kernelIdx+KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
	int KERNEL_LENGTH,
	int kernelIdx,
	cudaStream_t stream
){
    int KERNEL_RADIUS = (KERNEL_LENGTH -1)*0.5f;
	assert( COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS );
    assert( imageW % COLUMNS_BLOCKDIM_X == 0 );
    assert( imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0 );

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    convolutionColumnsKernel<<<blocks, threads, 0, stream>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW,
		KERNEL_LENGTH,
		kernelIdx
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

