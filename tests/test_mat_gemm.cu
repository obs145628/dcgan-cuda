#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>
#include "../src/memory/alloc.hh"

template<int width, int height, int K, int blockDimX, int blockDimY>
__global__
void mat_mul_cuda(float *A, float *B, float *C)
{
  __shared__ float A_tile[blockDimX * blockDimX];
  float cRes[blockDimX] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  const int aStart = width * blockDimX * blockIdx.y;
  const int aEnd = aStart + width - 1;
  const int bStart = blockDimX * 4 * blockIdx.x;
  const int bStep = blockDimX * width;

  for (int aIdx = aStart, bIdx = bStart;
       aIdx <= aEnd;
       aIdx += blockDimX, bIdx += bStep)
  {

    #pragma unroll
    for (int i = 0; i < 16; i += 4)
      A_tile[i + threadIdx.y + blockDimX * threadIdx.x] = A[aIdx + width * (i + threadIdx.y) + threadIdx.x];

    __syncthreads();

    float *bPartial = &B[bIdx + blockDimX * threadIdx.y + threadIdx.x];
    int indexPartial = 0;
    int tileIndex = 0;

    #pragma unroll
    for (int i = 0; i < blockDimX; ++i)
    {
      const float bVal = bPartial[indexPartial];
      cRes[0] += A_tile[tileIndex] * bVal;
      cRes[1] += A_tile[tileIndex + 1] * bVal;
      cRes[2] += A_tile[tileIndex + 2] * bVal;
      cRes[3] += A_tile[tileIndex + 3] * bVal;
      cRes[4] += A_tile[tileIndex + 4] * bVal;
      cRes[5] += A_tile[tileIndex + 5] * bVal;
      cRes[6] += A_tile[tileIndex + 6] * bVal;
      cRes[7] += A_tile[tileIndex + 7] * bVal;
      cRes[8] += A_tile[tileIndex + 8] * bVal;
      cRes[9] += A_tile[tileIndex + 9] * bVal;
      cRes[10] += A_tile[tileIndex + 10] * bVal;
      cRes[11] += A_tile[tileIndex + 11] * bVal;
      cRes[12] += A_tile[tileIndex + 12] * bVal;
      cRes[13] += A_tile[tileIndex + 13] * bVal;
      cRes[14] += A_tile[tileIndex + 14] * bVal;
      cRes[15] += A_tile[tileIndex + 15] * bVal;
      tileIndex += blockDimX;
      indexPartial += width;
    }

    __syncthreads();
  }

  int cIdx = width * blockDimX * blockIdx.y + blockDimX * blockIdx.x * 4
                  + blockDimX * threadIdx.y + threadIdx.x;
   #pragma unroll
  for (int i = 0; i < blockDimX; ++i)
  {
    C[cIdx] = cRes[i];
    cIdx += width;
  }
}

int main(int argc, char** argv)
{

    if (argc != 3)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    auto mats = tocha::Tensors::load(argv[1]);
    dbl_t* a = reinterpret_cast<dbl_t*>(mats.arr()[0].data);
    dbl_t* b = reinterpret_cast<dbl_t*>(mats.arr()[1].data);

    int height = 512;
    int width = 512;
    float *aCuda, *bCuda, *cCuda;
    cudaMalloc((void**)&aCuda, sizeof(float) * height * width);
    cudaMalloc((void**)&bCuda, sizeof(float) * height * width);
    cudaMalloc((void**)&cCuda, sizeof(float) * height * width);

    cudaMemcpy(aCuda, a, sizeof(float) * height * width, cudaMemcpyHostToDevice);
    cudaMemcpy(bCuda, b, sizeof(float) * height * width, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 dimBlock(16, 4);
    dim3 dimGrid(8, 32);
    mat_mul_cuda<512, 512, 512, 16, 4><<<dimGrid, dimBlock>>>(aCuda, bCuda, cCuda);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milli = 0;
    cudaEventElapsedTime(&milli, start, stop);
    std::cout << "Timer : " << milli << " ms" << std::endl;

    //cudaDeviceSynchronize();

    tocha::Tensors out;
    out.add(tocha::Tensor::f32(512, 512));
    dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

    cudaMemcpy(res, cCuda, sizeof(float) * height * width, cudaMemcpyDeviceToHost);

    out.save(argv[2]);
    cudaFree(aCuda);
    cudaFree(bCuda);
    cudaFree(cCuda);
}
