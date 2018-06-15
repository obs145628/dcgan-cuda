#include <iostream>
#include <cmath>

#include <tocha/tensor.hh>
#include "../src/memory/alloc.hh"

template<int widthA, int widthB, int blockDimX>
__global__
void mat_mul_cuda(const dbl_t *A, const dbl_t *B, dbl_t *C)
{
  __shared__ dbl_t A_tile[blockDimX * blockDimX];
  dbl_t cRes[blockDimX] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  const int aStart = widthA * blockDimX * blockIdx.y;
  const int aEnd = aStart + widthA - 1;
  const int bStart = blockDimX * 4 * blockIdx.x;
  const int bStep = blockDimX * widthB;

  int cIdx = widthB * blockDimX * blockIdx.y + blockDimX * blockIdx.x * 4
                  + blockDimX * threadIdx.y + threadIdx.x;

  for (int aIdx = aStart, bIdx = bStart;
       aIdx <= aEnd;
       aIdx += blockDimX, bIdx += bStep)
  {

    #pragma unroll
    for (int i = 0; i < 16; i += 4)
    {
      const int aIndex = aIdx + widthA * (i + threadIdx.y) + threadIdx.x;
      if (aIndex < 4800)
        A_tile[i + threadIdx.y + blockDimX * threadIdx.x] = A[aIndex];
    }

    __syncthreads();

    const int bPIndex = bIdx + blockDimX * threadIdx.y + threadIdx.x;
    const dbl_t *bPartial = &B[bPIndex];
    int indexPartial = 0;
    int tileIndex = 0;

    #pragma unroll
    for (int i = 0; i < blockDimX; ++i)
    {
      if (bPIndex + indexPartial < 4915200)
      {
        const dbl_t bVal = bPartial[indexPartial];

        #pragma unroll
        for (int j = 0; j < 16; ++j)
          cRes[j] += A_tile[tileIndex + j] * bVal;
        tileIndex += blockDimX;
        indexPartial += widthB;
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < blockDimX; ++i)
  {
    C[cIdx] = cRes[i];
    cIdx += widthB;
  }
}

template<int hSize, int wSize, int chSize, int nbFilter>
__global__
void ker_transform_cuda(const dbl_t *ker, dbl_t *res)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int filterIdx = index % nbFilter;
  const int chIdx = ((index - filterIdx) % (chSize * nbFilter))/nbFilter;
  const int wIdx = ((index - filterIdx - (chIdx * nbFilter))
                  % (chSize * nbFilter * wSize))/(chSize * nbFilter);
  const int hIdx = (index - wIdx * chSize * nbFilter - chIdx * nbFilter - filterIdx)/(wSize * chSize * nbFilter);
  const int nIndex = filterIdx * hSize * wSize * chSize + chIdx * wSize * hSize + hIdx * wSize + wIdx;
  res[nIndex] = ker[index];
}

template<int numPatchesX,
         int numPatchesY,
         int numPatchSlices,
         int width,
         int height,
         int windowWidth,
         int windowHeight,
         int strideX,
         int strideY,
         int padLeft,
         int padTop,
         int batchSize,
         int chSize>
__global__ void
im2col_gpu_kernel(float* stacked,
                  float const* data,
                  const int batchIdx)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {

    const int x = index % numPatchesX;
    int y = index / numPatchesX;
    const int z = y / numPatchesY;
    y %= numPatchesY;

    const int x_data = x * strideX - padLeft ;
    const int y_data = y * strideY - padTop;
    data += batchIdx * (chSize * width * height) + (y_data * chSize * width) + x_data * chSize + z;

    const int patchSliceOffset = (numPatchesX*numPatchesY*batchSize*windowWidth*windowHeight) * z + batchIdx * (numPatchesX*numPatchesY) + y * numPatchesX + x;
    stacked += patchSliceOffset;

    #pragma unroll
    for (int v = 0; v < windowHeight; ++v)
    {
      #pragma unroll
      for (int u = 0; u < windowWidth; ++u)
      {
        if (y_data + v >= 0 &&
            y_data + v < height &&
            x_data + u >= 0 &&
            x_data + u < width)
        {
          *stacked = data[v * chSize * width + u * chSize];
        } else {
          *stacked = 0 ;
        }
        stacked += (numPatchesX*numPatchesY*batchSize);
      }
    }
  }
}

__global__
void transform_res(const dbl_t *resConv, dbl_t *transf)
{
  int blkIdx = blockIdx.y * 65536 * 4 + blockIdx.x * 16;
  const int thIdx = threadIdx.y * 65536 + threadIdx.x;
  const int realIdx = blkIdx + thIdx;
  const int nbFIdx = realIdx / 65536;
  const int tmp0 = nbFIdx * 65536;
  const int batchIdx = (realIdx - tmp0) / 1024;
  const int tmp1 = batchIdx * 1024;
  const int tmp2 = realIdx - tmp0 - tmp1;
  const int hIdx = tmp2 / 32;
  const int wIdx = tmp2 % 32;

  transf[batchIdx * 65536 + hIdx * 32 * 64 + wIdx * 64 + nbFIdx] = resConv[realIdx];
}

void make_conv(char **argv)
{
  auto mats = tocha::Tensors::load(argv[1]);
  const dbl_t* inputHost = reinterpret_cast<dbl_t*>(mats.arr()[0].data);
  const dbl_t* kernelHost = reinterpret_cast<dbl_t*>(mats.arr()[1].data);

  const int batchSize = 64;
  const int height = 67; // 64 + 3 pad
  const int width = 67; // 64 + 3 pad
  const int chSize = 3;
  const int totalInputSize = batchSize * height * width * chSize;
  const int realTotalInputSize = batchSize * 64 * 64 * chSize;

  const int heightKer = 5;
  const int widthKer = 5;
  const int nbFilter = 64;
  const int totalKernelSize = heightKer * widthKer * chSize * nbFilter;

  const int stride = 2;
  const int padTop = 1;
  const int padBottom = 2;
  const int padL = 1;
  const int padR = 2;
  const int P = (64 + 3 - heightKer) / stride + 1;
  const int Q = (64 + 3 - widthKer) / stride + 1;
  const int newInputSize = chSize * heightKer * widthKer * batchSize * P * Q;

  float milli = 0;

  dbl_t *kernelCuda, *newKernelCuda;

  cudaEvent_t start, stop, startConv, stopKer, startPatch, stopPatch, startImg, stopImg;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventCreate(&startConv);
  cudaEventCreate(&startPatch);
  cudaEventCreate(&startImg);
  cudaEventCreate(&stopKer);
  cudaEventCreate(&stopPatch);
  cudaEventCreate(&stopImg);

  cudaEventRecord(start);
  cudaMalloc((void**)&kernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMalloc((void**)&newKernelCuda, sizeof(dbl_t) * totalKernelSize);
  cudaMemcpy(kernelCuda, kernelHost, sizeof(dbl_t) * totalKernelSize,
                  cudaMemcpyHostToDevice);
  dim3 dimGrid(totalKernelSize / 32);
  dim3 dimBlock(32);

  ker_transform_cuda<5, 5, 3, 64><<<dimGrid, dimBlock>>>(kernelCuda, newKernelCuda);
  cudaEventRecord(stopKer);
  //cudaDeviceSynchronize();
  cudaEventSynchronize(stopKer);

  cudaEventElapsedTime(&milli, start, stopKer);
  std::cout << "Timer Ker: " << milli << " ms" << std::endl;

  cudaEventRecord(startImg);
  dbl_t *inputCuda, *newInputCuda;
  cudaMalloc((void**)&inputCuda, sizeof(dbl_t) * realTotalInputSize);
  cudaMalloc((void**)&newInputCuda, sizeof(dbl_t) * newInputSize);
  cudaMemcpy(inputCuda, inputHost, sizeof(dbl_t) * realTotalInputSize,
                  cudaMemcpyHostToDevice);

  #pragma unroll
  for (int b = 0; b < 64; ++b)
  {
    im2col_gpu_kernel<32, 32, 3072, 64, 64, 5, 5, 2, 2, 1, 1, 64, 3>
            <<<3, 1024>>>(newInputCuda, inputCuda, b);
  }

  cudaEventRecord(stopImg);
  //cudaDeviceSynchronize();
  cudaEventSynchronize(stopImg);

  cudaEventElapsedTime(&milli, startImg, stopImg);
  std::cout << "Timer Img: " << milli << " ms" << std::endl;

  dbl_t *resConvCuda;
  const int resSize = nbFilter * batchSize * P * Q;
  cudaMalloc((void**)&resConvCuda, sizeof(dbl_t) * resSize);
  dim3 dimBlockConv(16, 4);
  dim3 dimGridConv(1024, 4);

  cudaEventRecord(startConv);
  mat_mul_cuda<75, 65536, 16><<<dimGridConv, dimBlockConv>>>(newKernelCuda, newInputCuda, resConvCuda);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&milli, startConv, stop);
  std::cout << "Timer conv: " << milli << " ms" << std::endl;
  cudaEventElapsedTime(&milli, start, stop);
  std::cout << "Timer all: " << milli << " ms" << std::endl;

  dbl_t *transfCuda;
  cudaMalloc((void**)&transfCuda, sizeof(dbl_t) * resSize);
  dim3 dimGridTransf(4096, 16);
  dim3 dimBlockTransf(16, 4);
  transform_res<<<dimGridTransf, dimBlockTransf>>>(resConvCuda, transfCuda);

  cudaDeviceSynchronize();

  tocha::Tensors out;
  out.add(tocha::Tensor::f32(64, 32, 32, 64));
  dbl_t* res = reinterpret_cast<dbl_t*>(out.arr()[0].data);

  cudaMemcpy(res, transfCuda, sizeof(dbl_t) * resSize, cudaMemcpyDeviceToHost);

  out.save(argv[2]);

  cudaFree(newInputCuda);
  cudaFree(newKernelCuda);
  cudaFree(resConvCuda);
  cudaFree(inputCuda);
  cudaFree(kernelCuda);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid number of arguments\n";
        return 1;
    }

    make_conv(argv);

    return 0;
}
