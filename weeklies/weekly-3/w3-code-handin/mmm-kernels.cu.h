#ifndef MULT_KERNELS
#define MULT_KERNELS

// widthA = heightB
template <class ElTp> 
__global__ void matMultKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  if( (gidx >= widthB) || (gidy >= heightA) ) return;

  for(int k = 0; k < widthA; k ++) {
      accum += A[gidy*widthA + k] * B[k*widthB + gidx];
  }

  C[gidy*widthB + gidx] = accum;
}


// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  __shared__ ElTp Ash[T][T];
  __shared__ ElTp Bsh[T][T];

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      Ash[threadIdx.y][threadIdx.x] = ((gidy < heightA) && (kk+threadIdx.x < widthA)) ?
            A[gidy*widthA + kk + threadIdx.x] : 0.0;
      Bsh[threadIdx.y][threadIdx.x] = ((gidx < widthB)  && (kk+threadIdx.y < widthA)) ?
            B[(threadIdx.y+kk)*widthB + gidx] : 0.0;
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < T; k++)
          accum += Ash[threadIdx.y][k] * Bsh[k][threadIdx.x];
      __syncthreads();
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}

template <class ElTp, int T>
__global__ void matMultRegTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    // ToDo: fill in the kernel implementation of register+block tiled
    //       matrix-matrix multiplication here
    ElTp cs[T];

    __shared__ ElTp Ash[T][T];

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int i;
    int ii = blockIdx.y * T;
    int jjj = blockIdx.x * T * T;

    int jj = jjj + (tidy * T);
    int j = jjj + tidx;

    int gidy = blockIdx.y*blockDim.y + threadIdx.y;

    for(int i =0; i<T; i++){
      cs[i] = 0.0;
    }
    __syncthreads();

    for(int kk = 0; kk < widthA; kk += T){
      // here we will  insert a collective  copy to  shared
      //  memory  of the  slice: A[ii : ii+T, kk : kk+T]
      // taken from matMultTiledKer
      Ash[threadIdx.y][threadIdx.x] = ((gidy < heightA) && (kk+threadIdx.x < widthA)) ?
            A[gidy*widthA + kk + threadIdx.x] : 0.0;
      __syncthreads();
      for(int k = 0; k < T; k++){
        ElTp b = ((j<widthB) && ((kk+k) < widthA))? B[(kk+k)*widthB + j]: 0.0;

        #pragma unroll
        for(i = 0; i<T; i++){
          cs[i] += Ash[i+ii][k+kk] * b;
        }
	__syncthreads();
      }
    }
    #pragma unroll
    for(int i = 0; i < T; i++){
      if ((i + ii < heightA) && (j < widthB)) {
        C[(i+ii)*widthB +j] = cs[i];
      } 
    }
}

// widthA = heightB
template <class ElTp, int T> 
__global__ void matMultCacheKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
  ElTp accum = 0.0f;

  int gidx = blockIdx.x*blockDim.x + threadIdx.x;
  int gidy = blockIdx.y*blockDim.y + threadIdx.y; 

  for(int kk = 0; kk < widthA; kk += T) {
      __syncthreads();
      #pragma unroll
      for(int k = 0; k < T; k++)
        accum += A[gidy*widthA + kk + k] * B[gidy*widthB + (kk+k)];
  }

  if( (gidx < widthB) && (gidy < heightA) )
    C[gidy*widthB + gidx] = accum;
}

/*
template <class ElTp, int T> 
__global__ void matMultRegTiledKer(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    // ToDo: fill in the kernel implementation of register+block tiled 
    //       matrix-matrix multiplication here

    __shared__ ElTp Ash[T][T];

    int ii = blockIdx.y * T; 
    int jjj = blockIdx.x * T*T;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int jj = tidy + ii;
    int j = tidx + jj;

    ElTp cs[T][T][T];

    for (int i = 0; i < heightA; i++) { //check the upper bound here
      cs[(jj-jjj)/T][j-jj][i-ii] = 0.0;
    } 

    for (int kk = 0; kk < widthA; kk += T) {
      // populate shared memory
      Ash[tidy][tidx] = ((jj < heightA) && (kk+tidx < widthA)) ?
            A[jj*widthA + kk + tidx] : 0.0;
      for (int k=0; k < widthA; k++) {
        ElTp b = B[k,j];
        #pragma unroll
        for (int i = ii; i < heightA; i++) {
          cs[(jj-jjj)/T][j-jj][i-ii] += (Ash[i,k] * b);
        }
      }
      
    }

    for(int i = 0; i < heightA; i++) {
      C[i,j] = cs[(jj-jjj)/T][j-jj][i-ii];
    }

}
*/

#endif
