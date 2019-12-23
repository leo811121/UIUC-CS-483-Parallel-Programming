#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 4
#define MASK_WIDTH 3
#define RADIUS 1

//@@ Define constant memory for device kernel here
__constant__ float MASK[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  
  int x_o = bx*TILE_WIDTH + tx;
  int y_o = by*TILE_WIDTH + ty;
  int z_o = bz*TILE_WIDTH + tz;
  
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];
  
  if (x_o >=0 && x_o < x_size && y_o >=0 && y_o < y_size && z_o >=0 && z_o < z_size)
    N_ds[tz][ty][tx] = input[x_size*y_size*z_o + x_size*y_o + x_o];
  else
    N_ds[tz][ty][tx] = 0;
  __syncthreads();
  
  int x_N_ds_Start = tx - RADIUS;
  int y_N_ds_Start = ty - RADIUS;
  int z_N_ds_Start = tz - RADIUS;
  float Pvalue = 0;
  
  for (int i = 0; i < MASK_WIDTH; i++)
    for (int j = 0; j < MASK_WIDTH; j++)
      for (int k = 0; k < MASK_WIDTH; k++){
        
        int x_ds_index = x_N_ds_Start + i;
        int y_ds_index = y_N_ds_Start + j;
        int z_ds_index = z_N_ds_Start + k;
        
        if (x_ds_index >= 0 && x_ds_index < TILE_WIDTH &&
            y_ds_index >= 0 && y_ds_index < TILE_WIDTH &&
            z_ds_index >= 0 && z_ds_index < TILE_WIDTH
           )
          Pvalue += N_ds[z_ds_index][y_ds_index][x_ds_index]*MASK[MASK_WIDTH*MASK_WIDTH*k + MASK_WIDTH*j + i];
        else{
          int x_Global = bx*TILE_WIDTH + x_ds_index;
          int y_Global = by*TILE_WIDTH + y_ds_index;
          int z_Global = bz*TILE_WIDTH + z_ds_index;
          
          if (x_Global >= 0 && x_Global < x_size &&
              y_Global >= 0 && y_Global < y_size &&
              z_Global >= 0 && z_Global < z_size)
            Pvalue += input[x_size*y_size*z_Global + x_size*y_Global + x_Global]*MASK[MASK_WIDTH*MASK_WIDTH*k + MASK_WIDTH*j + i];
        }
      }

  if (x_o < x_size && y_o < y_size && z_o < z_size)
     output[x_size*y_size*z_o + x_size*y_o + x_o] = Pvalue;

   __syncthreads();
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel = (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**)&deviceInput, x_size*y_size*z_size*sizeof(float));
  cudaMalloc((void**)&deviceOutput, x_size*y_size*z_size*sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, x_size*y_size*z_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(MASK, hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil((1.0*x_size)/TILE_WIDTH), ceil((1.0*y_size)/TILE_WIDTH), ceil((1.0*z_size)/TILE_WIDTH));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size,
                                y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3 , deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
