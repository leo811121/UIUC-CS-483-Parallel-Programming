// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void float2Char(float *input, unsigned char *output, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int x = bx*blockDim.x + tx;
  int y = by*blockDim.y + ty;
  if(y < height && x < width){
    int idx = (width * height)*bz + (width)*y + x;
    output[idx] = (unsigned char) (255 * input[idx]);
  }
}

__global__ void rgb2Gray(unsigned char *input, unsigned char *output, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y;  
  int tx = threadIdx.x; int ty = threadIdx.y;
  int x = bx*blockDim.x + tx;
  int y = by*blockDim.y + ty;
  if(y < height && x < width){
    int idx = y * (width) + x;
    uint8_t r = input[3 * idx];
    uint8_t g = input[3 * idx + 1];
    uint8_t b = input[3 * idx + 2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void gray2Hist(unsigned char *input, unsigned int *output, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int x = bx*blockDim.x + tx;
  int y = by*blockDim.y + ty;
  
  __shared__ unsigned int hist[HISTOGRAM_LENGTH];
  int tIdx = blockDim.x*ty + tx;
  if (tIdx < HISTOGRAM_LENGTH) 
    hist[tIdx] = 0;
  
  __syncthreads();
  
  if (x < width && y < height) {
    int idx = y * (width) + x;
    unsigned char val = input[idx];
    atomicAdd(&(hist[val]), 1);
  }

  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) 
    atomicAdd(&(output[tIdx]), hist[tIdx]);
  
}

__global__ void hist2CDF(unsigned int *input, float *output, int width, int height){
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int id = threadIdx.x;
  
  if(id < HISTOGRAM_LENGTH)
    cdf[id] = input[id];
  __syncthreads();
  
  //reduction
  int stride = 1;
  while(stride < HISTOGRAM_LENGTH)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < HISTOGRAM_LENGTH && (index-stride) >= 0)
            cdf[index] += cdf[index-stride];
        stride = stride*2;
    }
  
  stride = HISTOGRAM_LENGTH/4;
  while(stride > 0)
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if((index+stride) < HISTOGRAM_LENGTH)
        {
	       cdf[index+stride] += cdf[index];
        }				
        stride = stride / 2;
    }
  __syncthreads();
  output[id] = cdf[id] / ((float)(width * height));
}


__global__ void equal(unsigned char *img, float *cdf, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 
  int tx = threadIdx.x; int ty = threadIdx.y;
  int x = bx*blockDim.x + tx;
  int y = by*blockDim.y + ty;
  
  if(x < width && y < height){
    int idx = (width * height)*bz + (width)*y + x;
    float v = 255*(cdf[img[idx]] - cdf[0])/(1.0 - cdf[0]);
    img[idx] = (unsigned char) min(max(v, 0.0), 255.0);
  }
}

__global__ void uint2float(unsigned char *input, float *output, int width, int height){
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; 
  int tx = threadIdx.x; int ty = threadIdx.y;
  int x = bx*blockDim.x + tx;
  int y = by*blockDim.y + ty;
  if(x < width && y < height){
    int idx = (width * height)*bz + (width)*y + x;
    output[idx] = (float) (input[idx] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceFloat;
  unsigned char *deviceUint;
  unsigned char *deviceGray;
  unsigned int *deviceHist;
  float *deviceCDF; 

  args = wbArg_read(argc, argv); /* parse the input arguments */
  inputImageFile = wbArg_getInputFile(args, 0);
  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);//get image data 
  hostOutputImageData = wbImage_getData(outputImage); 
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  //@@ insert code here
  cudaMalloc((void**) &deviceFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceUint, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
  cudaMalloc((void**) &deviceGray, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceHist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) deviceHist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  
  cudaMemcpy(deviceFloat, hostInputImageData, 
             imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyHostToDevice);
  
  
  dim3 dimGrid;
  dim3 dimBlock;
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  float2Char<<<dimGrid, dimBlock>>>(deviceFloat, deviceUint, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  rgb2Gray<<<dimGrid, dimBlock>>>(deviceUint, deviceGray, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  gray2Hist<<<dimGrid, dimBlock>>>(deviceGray, deviceHist, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  hist2CDF<<<dimGrid, dimBlock>>>(deviceHist, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  equal<<<dimGrid, dimBlock>>>(deviceUint, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  uint2float<<<dimGrid, dimBlock>>>(deviceUint, deviceFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hostOutputImageData, deviceFloat,
             imageWidth*imageHeight*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceFloat);
  cudaFree(deviceUint);
  cudaFree(deviceGray);
  cudaFree(deviceHist);
  cudaFree(deviceCDF);

  return 0;
}
