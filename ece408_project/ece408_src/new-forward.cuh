
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 5
#define TILE_WIDTH1 16 
#define TILE_WIDTH2 24

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define X_unroll(i2, i1, i0) X_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define k_unroll(i1, i0) k_unroll[i1 * (C * K * K) + i0]
#define xunroll3d(i2, i1, i0) x_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define kunroll2d(i1, i0) k_unroll[(i1) * (C * K * K) + i0]

#define TILE_SZ_A 128
#define TILE_SZ_B 8

#define kernel4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


namespace mxnet
{
namespace op
{
__constant__ float kernel[7200];

//baseline
__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    if(h < H_out && w < W_out){
        float val = 0;
		//#pragma unroll 2 
        for(int c = 0; c < C; c++){
			//#pragma unroll 5
            for(int p = 0; p < K; p++){
				//#pragma unroll 5
                for(int q = 0; q < K; q++){
                    //val += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
					val += x4d(b, c, h+p, w+q) * kernel4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = val;
    }   
}


//kernel for first convolution layer 
__global__ void forward_kernel1(float * __restrict__ y, const float * __restrict__ x, const float * __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	
    //int W_grid = ceil((float)W_out / TILE_WIDTH);
	int W_grid = ceil((float)W_out / 16);
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    //const int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    //const int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
	const int h = blockIdx.z / W_grid * 16 + threadIdx.y;
    const int w = blockIdx.z % W_grid * 16 + threadIdx.x;
    if(h < H_out && w < W_out){
        float val = 0;
		//#pragma unroll 2 
        //for(int c = 0; c < C; c++){
			#pragma unroll 5
            for(int p = 0; p < K; p++){
				#pragma unroll 5
                for(int q = 0; q < K; q++){
                    //val += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
					//val += x4d(b, c, h+p, w+q) * kernel4d(m, c, p, q);
					val += x4d(b, 0, h+p, w+q) * kernel4d(m, 0, p, q);
                }
            }
        //}
        y4d(b, m, h, w) = val;
    }   
}


//kernel for second convolution layer
__global__ void forward_kernel2(float* /*__restrict__*/  y, const float* /*__restrict__*/  x, const float* __restrict__ k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	
    int W_grid = ceil((float)W_out / TILE_WIDTH);
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    const int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
    if(h < H_out && w < W_out){
        float val = 0;
		#pragma unroll 12 
        for(int c = 0; c < C; c++){
			#pragma unroll 5
            for(int p = 0; p < K; p++){
				#pragma unroll 5
                for(int q = 0; q < K; q++){
                    //val += x4d(b, c, h+p, w+q) * k4d(m, c, p, q);
					val += x4d(b, c, h+p, w+q) * kernel4d(m, c, p, q);
                }
            }
        }
        y4d(b, m, h, w) = val;
    }   
}


__global__ void unroll_x(const int C, const int H, const int W, const int K, const float* x, float* X_unroll, const int B, int B_pre)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int c, s, h_out, w_out, w_unroll, h_unroll, h_base, p, q;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int out_number = H_out * W_out;
	if (t < C * out_number && b < (B-B_pre)) {
		c = t / out_number;
		s = t % out_number;
		h_out = s / W_out;
		w_out = s % W_out;
		w_unroll = h_out * W_out + w_out;
		h_base = c * K * K;
		
		
		for(p = 0; p < K; p++){
			for(q = 0; q < K; q++) {
				h_unroll = h_base + p * K + q; 
				X_unroll(b, h_unroll, w_unroll) = x4d(b+B_pre, c, h_out + p, w_out + q);				
			}
		}
	
	}		
}

__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int batch, int B_pre) {
  
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Layer = blockIdx.z * blockDim.z + threadIdx.z;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
  	__shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
	
	int numTiles = numAColumns/TILE_WIDTH;
    if (numAColumns%TILE_WIDTH) numTiles++;
	
	float CVal = 0.0;

	
	for(int a = 0; a < numTiles; ++a){
		
		if(Row < numARows && a * TILE_WIDTH + tx < numAColumns){
			tileA[ty][tx] = A[Row * numAColumns + a * TILE_WIDTH + tx];
		}
	
		else tileA[ty][tx] = 0.0;
	
		if(a * TILE_WIDTH + ty < numBRows && Col < numBColumns){
			tileB[ty][tx] = B[(Layer) * numBColumns * numBRows + (a * TILE_WIDTH + ty) * numBColumns + Col];
		}
		else tileB[ty][tx] = 0.0;
		__syncthreads();
		
			for(int k = 0; k < TILE_WIDTH; ++k){
				CVal += tileA[ty][k] * tileB[k][tx];
			}
			__syncthreads(); 		
				
	}
	if(Row < numCRows && Col < numCColumns && Layer < (batch-B_pre)){
		C[(Layer+B_pre) * numCColumns * numCRows + Row * numCColumns + Col] = CVal;
	}		
}

__global__ void reg_matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int batch, int B_pre) {
  
	__shared__ float tile[TILE_SZ_A];

	int ty = threadIdx.y;
	int row = ty + blockIdx.y*blockDim.y;
	int Layer = blockIdx.z * blockDim.z + threadIdx.z;
	float reg = 0.0;
	float Pvalues[TILE_SZ_B]={0};
	
	for(int ph=0;ph<numAColumns;ph++)
	{	
		
		if(TILE_SZ_B*blockIdx.x+ty < numBColumns && Layer < (batch-B_pre) && ty < TILE_SZ_B){
			//tile[ty] = B(ph,TILE_SZ_B*blockIdx.x+ty);
			tile[ty] = B[(Layer) * numBColumns * numBRows + ph*numBColumns +TILE_SZ_B*blockIdx.x+ty];			
		}
		
		//else if (ty < TILE_SZ_B){
		else 		
		tile[ty] = 0.0;
		//}
    	__syncthreads();
		
    	if(row < numARows){
        	//reg = A(row,ph);
			reg = A[row*numAColumns + ph];
		}
      	else
       		reg = 0.0;
 
    	for(int step=0;step<TILE_SZ_B;step++)
    	{
  
			Pvalues[step]+=reg*tile[step];
      
		}
		__syncthreads();
	}
 
    for(int x=0;x<TILE_SZ_B;x++){
		if(row< numCRows &&  TILE_SZ_B*blockIdx.x+x < numCColumns && Layer< (batch-B_pre)){
       		//C(row,TILE_SZ_B*blockIdx.x+x) = Pvalues[x];
			C[(Layer+B_pre) * numCColumns * numCRows + row*numCColumns + TILE_SZ_B*blockIdx.x+x] = Pvalues[x]; 
     	}
	}		
}

__global__ void fusion(const int C, const int H, const int W, const int K, const int M, const int B, int B_pre, const float* x, float* y, float *w, float* X_unroll, int numARows, int numAColumns, int numBRows
						,int numBColumns, int numCRows, int numCColumns) {
  	
	
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	int b = blockIdx.y * blockDim.y + threadIdx.y;
	int c, s, h_out, w_out, w_unroll, h_unroll, h_base, p, q;
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	int out_number = H_out * W_out;
	int Layer = blockIdx.z * blockDim.z + threadIdx.z;
	
	
	if (t < C * out_number && b < (B-B_pre) && Layer == 0) {
		c = t / out_number;
		s = t % out_number;
		h_out = s / W_out;
		w_out = s % W_out;
		w_unroll = h_out * W_out + w_out;
		h_base = c * K * K;
				
		for(p = 0; p < K; p++){
			for(q = 0; q < K; q++) {
				h_unroll = h_base + p * K + q; 
				X_unroll(b, h_unroll, w_unroll) = x4d(b+B_pre, c, h_out + p, w_out + q);				
			}
		}
	}
	__syncthreads();

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	//int Layer = blockIdx.z * blockDim.z + threadIdx.z;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	__shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileB[TILE_WIDTH][TILE_WIDTH];
	
	int numTiles = numAColumns/TILE_WIDTH;
    if (numAColumns%TILE_WIDTH) numTiles++;
	
	float CVal = 0.0;
	for(int a = 0; a < numTiles; ++a){
		if(Row < numARows && a * TILE_WIDTH + tx < numAColumns){
			tileA[ty][tx] = w[Row * numAColumns + a * TILE_WIDTH + tx];
		}
		else tileA[ty][tx] = 0.0;
		if(a * TILE_WIDTH + ty < numBRows && Col < numBColumns){
			tileB[ty][tx] = X_unroll[Layer * numBColumns * numBRows + (a * TILE_WIDTH + ty) * numBColumns + Col];
		}
		else tileB[ty][tx] = 0.0;
		__syncthreads();
			for(int k = 0; k < TILE_WIDTH; ++k){
				CVal += tileA[ty][k] * tileB[k][tx];
			}
		__syncthreads(); 		
		
	}
	if(Row < numCRows && Col < numCColumns && Layer < (B-B_pre)){
		y[(Layer+B_pre) * numCColumns * numCRows + Row * numCColumns + Col] = CVal;
	}
}

__global__ void unrolltoshare(int C, int K, int M,int H, int W, int W_out, int H_out, float* __restrict__  x, float* __restrict__  k, float* __restrict__ y) {
	__shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];

	int b = blockIdx.z;
	int tx = threadIdx.x, ty = threadIdx.y;
	int row = blockIdx.y * TILE_WIDTH + ty;
	int column = blockIdx.x * TILE_WIDTH + tx;
	int numMatAColumns = C*K*K; // This is the same as numMatBRows.

	float acc = 0.0;
	int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH));
	//#pragma unroll
	for (int i = 0; i < num_iterations; i++) { 
		int temp_col = i*TILE_WIDTH + tx, temp_row = i*TILE_WIDTH + ty;
		tileMatA[ty][tx] = 0;
		tileMatB[ty][tx] = 0;

		// Original indices in the filter tensor.
		int W_m = row;
		int W_c = temp_col/(K*K);
		int W_h = (temp_col%(K*K))/K, W_w = (temp_col%(K*K))%K;

		if (temp_col < numMatAColumns && row < M)
			tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
		else
			tileMatA[ty][tx] = 0;
		// Original indices in the input tensor.
				
		int X_b = b;				
		int X_c = temp_row/(K*K);
		int X_p = temp_row%(K*K)/K , X_q = (temp_row%(K*K))%K;
		int X_h = column/W_out, X_w =column%W_out;
		

		if (temp_row < numMatAColumns && column < H_out*W_out)
			tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
		else
			tileMatB[ty][tx] = 0;

		__syncthreads();
		//#pragma unroll 23
		for (int q = 0; q < TILE_WIDTH; q++)
			acc += tileMatA[ty][q] * tileMatB[q][tx];
		__syncthreads();
	}

	// Original indices in the output tensor.
	int Y_b = b;
	int Y_m = row;
	int Y_h = column / W_out, Y_w = column % W_out;

	if (row < M && column < W_out*H_out)
		y4d(Y_b, Y_m, Y_h, Y_w) = acc;
}


__global__ void unrolltoshare1(int C, int K, int M,int H, int W, int W_out, int H_out, float* __restrict__  x, float* __restrict__  k, float* __restrict__ y) {
	__shared__ float tileMatA[TILE_WIDTH1][TILE_WIDTH1];
	__shared__ float tileMatB[TILE_WIDTH1][TILE_WIDTH1];

	int b = blockIdx.z;
	int tx = threadIdx.x, ty = threadIdx.y;
	int row = blockIdx.y * TILE_WIDTH1 + ty;
	int column = blockIdx.x * TILE_WIDTH1 + tx;
	int numMatAColumns = C*K*K; // This is the same as numMatBRows.

	float acc = 0.0;
	int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH1));
	//#pragma unroll
	for (int i = 0; i < num_iterations; i++) { 
		int temp_col = i*TILE_WIDTH1 + tx, temp_row = i*TILE_WIDTH1 + ty;
		tileMatA[ty][tx] = 0;
		tileMatB[ty][tx] = 0;

		// Original indices in the filter tensor.
		int W_m = row;
		int W_c = temp_col/(K*K);
		int W_h = (temp_col%(K*K))/K, W_w = (temp_col%(K*K))%K;

		if (temp_col < numMatAColumns && row < M)
			tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
		else
			tileMatA[ty][tx] = 0;
		// Original indices in the input tensor.
				
		int X_b = b;				
		int X_c = temp_row/(K*K);
		int X_p = temp_row%(K*K)/K , X_q = (temp_row%(K*K))%K;
		int X_h = column/W_out, X_w =column%W_out;
		

		if (temp_row < numMatAColumns && column < H_out*W_out)
			tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
		else
			tileMatB[ty][tx] = 0;

		__syncthreads();
		//#pragma unroll 23
		for (int q = 0; q < TILE_WIDTH1; q++){
			acc += tileMatA[ty][q] * tileMatB[q][tx];
		}
		__syncthreads();
	}

	// Original indices in the output tensor.
	int Y_b = b;
	int Y_m = row;
	int Y_h = column / W_out, Y_w = column % W_out;

	if (row < M && column < W_out*H_out)
		y4d(Y_b, Y_m, Y_h, Y_w) = acc;
}


__global__ void unrolltoshare2(int C, int K, int M,int H, int W, int W_out, int H_out, float* x, float* __restrict__ k, float*  y) {
	__shared__ float tileMatA[TILE_WIDTH2][TILE_WIDTH2];
	__shared__ float tileMatB[TILE_WIDTH2][TILE_WIDTH2];

	int b = blockIdx.z;
	int tx = threadIdx.x, ty = threadIdx.y;
	int row = blockIdx.y * TILE_WIDTH2 + ty;
	int column = blockIdx.x * TILE_WIDTH2 + tx;
	int numMatAColumns = C*K*K; // This is the same as numMatBRows.

	float acc = 0.0;
	int num_iterations = ceil(numMatAColumns/(1.0*TILE_WIDTH2));
	//#pragma unroll
	for (int i = 0; i < num_iterations; i++) { 
		int temp_col = i*TILE_WIDTH2 + tx, temp_row = i*TILE_WIDTH2 + ty;
		tileMatA[ty][tx] = 0;
		tileMatB[ty][tx] = 0;

		// Original indices in the filter tensor.
		int W_m = row;
		int W_c = temp_col/(K*K);
		int W_h = (temp_col%(K*K))/K, W_w = (temp_col%(K*K))%K;

		if (temp_col < numMatAColumns && row < M)
			tileMatA[ty][tx] = k4d(W_m, W_c, W_h, W_w);
		else
			tileMatA[ty][tx] = 0;
		// Original indices in the input tensor.
				
		int X_b = b;				
		int X_c = temp_row/(K*K);
		int X_p = temp_row%(K*K)/K , X_q = (temp_row%(K*K))%K;
		int X_h = column/W_out, X_w =column%W_out;
		

		if (temp_row < numMatAColumns && column < H_out*W_out)
			tileMatB[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
		else
			tileMatB[ty][tx] = 0;

		__syncthreads();
		//#pragma unroll 24
		for (int q = 0; q < TILE_WIDTH2; q++){		
			acc += tileMatA[ty][q] * tileMatB[q][tx];
		}
		__syncthreads();
	}

	// Original indices in the output tensor.
	int Y_b = b;
	int Y_m = row;
	int Y_h = column / W_out, Y_w = column % W_out;

	if (row < M && column < W_out*H_out)
		y4d(Y_b, Y_m, Y_h, Y_w) = acc;
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
	
    const int B = x.shape_[0];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];    
    
    const int M = y.shape_[1];
    const int K = w.shape_[3];
	
    int H_out = H - K + 1;
    int W_out = W - K + 1;

	//printf("Hout= %d, WOUT = %d, K = %d , M = %d\n", H_out, W_out, K, M);

	
/*********************************different layer ***********************************************************/ 

	if(C==1){
		
	//unroll directly to share memory			
	dim3 gridDim1(ceil(H_out*W_out/(1.0*TILE_WIDTH1)),ceil(M/(1.0*TILE_WIDTH1)),B);
	dim3 blockDim1(TILE_WIDTH1,TILE_WIDTH1,1);
	unrolltoshare1<<<gridDim1, blockDim1>>>(C, K, M, H, W ,W_out, H_out, x.dptr_, w.dptr_, y.dptr_);	
		
	/*	
	int W_grid = ceil(1.0*W_out / 16);
    int H_grid = ceil(1.0*H_out / 16);

    int Z = H_grid * W_grid;

	cudaMemcpyToSymbol(kernel, w.dptr_, w.shape_[0]*w.shape_[1]*w.shape_[2]*w.shape_[3]* sizeof(float), 0, cudaMemcpyDeviceToDevice);		
    // Set the kernel dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(B, M, Z);

    // Call the kernel
	forward_kernel1<<<gridDim, blockDim,0,0>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K);
	*/

	}
	
	else if(C==12){
			
	//unroll directly to share memory	
	dim3 gridDim2(ceil(H_out*W_out/(1.0*TILE_WIDTH2)),ceil(M/(1.0*TILE_WIDTH2)),B);
	dim3 blockDim2(TILE_WIDTH2,TILE_WIDTH2,1);
	unrolltoshare2<<<gridDim2, blockDim2>>>(C, K, M, H, W ,W_out, H_out, x.dptr_, w.dptr_, y.dptr_);
	
	/*
	int W_grid = ceil(1.0*W_out / 32);
    int H_grid = ceil(1.0*H_out / 32);

    int Z = H_grid * W_grid;

	//constant memory
	cudaMemcpyToSymbol(kernel, w.dptr_, w.shape_[0]*w.shape_[1]*w.shape_[2]*w.shape_[3]* sizeof(float), 0, cudaMemcpyDeviceToDevice);	
    // Set the kernel dimensions
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(B, M, Z);
	
    // Call the kernel
	forward_kernel2<<<gridDim, blockDim,0,0>>>(y.dptr_,x.dptr_,w.dptr_,B,M,C,H,W,K);
	*/
	}
	
	

/*********************************unroll + share memory multiplication***********************************************************/ 
/*		
	//parameters for matrix multiply 		
	int numARows;    
	int numAColumns; 
	int numBRows;    
	int numBColumns; 
	int numCRows;   
	int numCColumns;
	
	numARows = M;
	numAColumns = C *K *K; 
	
	numBRows = C *K * K;
	numBColumns = H_out * W_out; 
	
	numCRows = numARows;
  	numCColumns = numBColumns;

    //unroll x & matrix multi.
    //part1	 
    int B_curr = 4000;
    int B_pre = 0;
	float *x_unroll;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid, unrollBlock>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);
    
	dim3 matrixGrid(numCColumns/TILE_WIDTH,numCRows/TILE_WIDTH,(B_curr-B_pre));
	if (numCColumns%TILE_WIDTH) matrixGrid.x++;
	if (numCRows%TILE_WIDTH) matrixGrid.y++;
	dim3 matrixBlock(TILE_WIDTH,TILE_WIDTH,1);
	matrixMultiply<<<matrixGrid, matrixBlock>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);    

    //part2
    B_curr = 8000;
    B_pre = 4000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid_2(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock_2(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid_2, unrollBlock_2>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);
    
    dim3 matrixGrid2(numCColumns/TILE_WIDTH,numCRows/TILE_WIDTH,(B_curr-B_pre));
	if (numCColumns%TILE_WIDTH) matrixGrid2.x++;
	if (numCRows%TILE_WIDTH) matrixGrid2.y++;
    //dim3 matrixGrid(ceil((float)numCColumns/TILE_WIDTH),ceil((float)numCRows/TILE_WIDTH,(B_curr-B_pre)));
	dim3 matrixBlock2(TILE_WIDTH,TILE_WIDTH,1);
	matrixMultiply<<<matrixGrid2, matrixBlock2>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);
    
    //part3
    B_curr = 10000;
    B_pre = 8000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid_3(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock_3(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid_3, unrollBlock_3>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);
    
    dim3 matrixGrid3(numCColumns/TILE_WIDTH,numCRows/TILE_WIDTH,(B_curr-B_pre));
	if (numCColumns%TILE_WIDTH) matrixGrid3.x++;
	if (numCRows%TILE_WIDTH) matrixGrid3.y++;
    //dim3 matrixGrid(ceil((float)numCColumns/TILE_WIDTH),ceil((float)numCRows/TILE_WIDTH,(B_curr-B_pre)));
	dim3 matrixBlock3(TILE_WIDTH,TILE_WIDTH,1);
	matrixMultiply<<<matrixGrid3, matrixBlock3>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);
*/
/**********************************************unroll + register tiling multiplication*****************************************************************************************************************/

/*
	//parameters for matrix multiply 		
	int numARows;    
	int numAColumns; 
	int numBRows;    
	int numBColumns; 
	int numCRows;   
	int numCColumns;
	
	numARows = M;
	numAColumns = C *K *K; 
	
	numBRows = C *K * K;
	numBColumns = H_out * W_out; 
	
	numCRows = numARows;
  	numCColumns = numBColumns;

    //reg tiling matrix multiply
    //part1
    
    int B_curr = 4000;
    int B_pre = 0;
	float *x_unroll;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid, unrollBlock>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);

	dim3 tileGrid((numCColumns-1)/TILE_SZ_B +1,(numCRows-1)/TILE_SZ_A+1,(B_curr-B_pre));
	dim3 tileBlock(1,TILE_SZ_A,1);
	reg_matrixMultiply<<<tileGrid, tileBlock>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);
    
    //part2
    B_curr = 8000;
    B_pre = 4000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid2(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock2(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid2, unrollBlock2>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);
    
    dim3 tileGrid2((numCColumns-1)/TILE_SZ_B +1,(numCRows-1)/TILE_SZ_A+1,(B_curr-B_pre));
	dim3 tileBlock2(1,TILE_SZ_A,1);
	reg_matrixMultiply<<<tileGrid2, tileBlock2>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);

    //part3
    B_curr = 10000;
    B_pre = 8000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid3(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock3(TILE_WIDTH, TILE_WIDTH, 1);
	unroll_x<<<unrollGrid3, unrollBlock3>>>(C, H, W, K, x.dptr_, x_unroll, B_curr, B_pre);
    
    dim3 tileGrid3((numCColumns-1)/TILE_SZ_B +1,(numCRows-1)/TILE_SZ_A+1,(B_curr-B_pre));
	dim3 tileBlock3(1,TILE_SZ_A,1);
	reg_matrixMultiply<<<tileGrid3, tileBlock3>>>(w.dptr_, x_unroll, y.dptr_, numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns,B_curr, B_pre);
    cudaFree(x_unroll);
*/	
/************************************************fusion unroll sharmemory multiplication*************************************************************************/

/*
	//parameters for matrix multiply 		
	int numARows;    
	int numAColumns; 
	int numBRows;    
	int numBColumns; 
	int numCRows;   
	int numCColumns;
	
	numARows = M;
	numAColumns = C *K *K; 
	
	numBRows = C *K * K;
	numBColumns = H_out * W_out; 
	
	numCRows = numARows;
  	numCColumns = numBColumns;

	
    //fusion
    //part1
    float *x_unroll;
    int B_curr = 4000;
    int B_pre = 0;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 fusionGrid(unrollGrid.x,unrollGrid.y, B_curr-B_pre);
	dim3 fusionBlock(TILE_WIDTH,TILE_WIDTH, 1);
	fusion<<<fusionGrid, fusionBlock>>>(C, H, W, K, M, B_curr, B_pre, x.dptr_, y.dptr_, w.dptr_, x_unroll,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
    cudaFree(x_unroll);

    //part2
    B_curr = 8000;
    B_pre = 4000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid2(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock2(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 fusionGrid2(unrollGrid2.x,unrollGrid2.y, B_curr-B_pre);
	dim3 fusionBlock2(TILE_WIDTH,TILE_WIDTH, 1);
	fusion<<<fusionGrid2, fusionBlock2>>>(C, H, W, K, M, B_curr, B_pre, x.dptr_, y.dptr_, w.dptr_, x_unroll,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
    cudaFree(x_unroll);

    //part3
    //part3
    B_curr = 10000;
    B_pre = 8000;	
	cudaMalloc((void **)&x_unroll, (B_curr-B_pre) * K * K * C * H_out* W_out*sizeof(float));
	dim3 unrollGrid3(ceil((float)C*H_out*W_out/TILE_WIDTH), ceil((float)(B_curr-B_pre)/TILE_WIDTH), 1);
	dim3 unrollBlock3(TILE_WIDTH, TILE_WIDTH, 1);
	dim3 fusionGrid3(unrollGrid3.x,unrollGrid3.y, B_curr-B_pre);
	dim3 fusionBlock3(TILE_WIDTH,TILE_WIDTH, 1);
	fusion<<<fusionGrid3, fusionBlock3>>>(C, H, W, K, M, B_curr, B_pre, x.dptr_, y.dptr_, w.dptr_, x_unroll,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
    cudaFree(x_unroll);
	MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
*/
}


template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    //CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#undef y4d
#undef x4d
#undef k4d
#undef X_unroll
#undef funroll2d
#endif

